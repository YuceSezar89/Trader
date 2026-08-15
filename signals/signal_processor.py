import asyncio
import json
import time
from typing import Awaitable, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

from config import Config
from database.engine import get_session, run_with_db_timeout
from database.models import Signal
from indicators.core import (
    calculate_adx,
    calculate_atr,
    calculate_macd,
    calculate_mfi,
    calculate_obv,
    calculate_rsi,
    truncate_after_gap,
)
from indicators.financial_metrics import calculate_metrics
from signals import ta_kovalama_gate, tf_alignment_gate
from signals.market_context import (
    compute_cvd_slope,
    compute_smc_market_structure,
    compute_vp_score,
)
from signals.paper_trade_manager import (
    ha_cross_manager,
    paper_trade_manager,
    rsi_15m_manager,
    rsi_cross_live_manager,
)
from signals.risk_manager import risk_manager
from signals.signal_engine import signal_engine
from signals.signal_lifecycle_manager import signal_lifecycle_manager
from signals.vpm_calculator import VPMCalculator
from utils.logger import get_logger
from utils.preprocessing import (
    normalize_volatility_0_100,
)
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient
from utils.vpmv import compute_components, compute_pre, compute_raw_components

logger = get_logger(__name__)

_PT_FLAG_CACHE: dict = {"value": "1", "ts": 0.0}
_PT_FLAG_TTL = 30.0


def trim_to_closed_bar(df: pd.DataFrame, closed_open_time: int) -> pd.DataFrame:
    """df'yi kapanış event'inin kendi taşıdığı open_time'a göre kırpar — bar
    kapandıktan birkaç saniye sonra canlı tick akışı bir sonraki barın forming
    satırını Redis buffer'ına eklemiş olabilir. Bu olmadan farklı modüller
    df'nin son satırının kapanmış mı forming mi olduğunu birbirinden bağımsız
    ve çelişkili şekilde tahmin ediyordu (12 Tem 2026 denetimi — sinyal
    zamanlama/VPMV skor bug'ı). Kırpma sonrası df'nin SON satırı her zaman
    kesin olarak kapanmış barın kendisidir, çağıran taraflar artık iloc[:-1]
    ile ayrıca "forming" varsaymamalı."""
    if df is None or df.empty or "open_time" not in df.columns:
        return df
    mask = df["open_time"] <= closed_open_time
    if not mask.any():
        return df.iloc[:0]
    return df.loc[mask]


async def _compute_all_up(
    symbol: str,
    interval: str,
    indicators_name: str,
    sig_type: str,
    vol_s: Optional[float],
    mom_s: Optional[float],
    vlt_s: Optional[float],
    prc_s: Optional[float],
) -> Optional[bool]:
    """Aynı symbol+interval+indicators+signal_type için bir önceki sinyale göre
    hacim/momentum/volatilite/fiyat skorlarının HEPSİ arttı mı?"""
    if None in (vol_s, mom_s, vlt_s, prc_s):
        return None

    async def _do_query() -> Optional[bool]:
        async with get_session() as session:
            result = await session.execute(
                select(Signal.vol_score, Signal.mom_score, Signal.volat_score, Signal.price_score)
                .where(
                    Signal.symbol == symbol,
                    Signal.interval == interval,
                    Signal.indicators == indicators_name,
                    Signal.signal_type == sig_type,
                    Signal.vol_score.isnot(None),
                )
                .order_by(Signal.opened_at.desc())
                .limit(1)
            )
            row = result.first()
            if row is None:
                return None
            prev_vol, prev_mom, prev_vlt, prev_prc = row
            if None in (prev_vol, prev_mom, prev_vlt, prev_prc):
                return None
            return vol_s > prev_vol and mom_s > prev_mom and vlt_s > prev_vlt and prc_s > prev_prc

    try:
        return await run_with_db_timeout(_do_query())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] all_up hesaplanamadı: %s", symbol, exc)
        return None


_EMPTY_COMPONENT_PCT = {"vol": None, "mom": None, "vlt": None, "prc": None}


async def _compute_component_change_pct(
    symbol: str,
    interval: str,
    indicators_name: str,
    sig_type: str,
    raw: Optional[dict],
) -> dict:
    """4 ham bileşenin (vol/mom/vlt/prc — utils/vpmv.py::compute_raw_components),
    aynı symbol+interval+indicators+signal_type için BİR ÖNCEKİ sinyale göre
    yüzdesel değişimi (Sinyal Mumu Ratioları'nın `volume/lastSignalVolume-1`
    mantığıyla aynı baseline — 5-bar pencere ortalaması DEĞİL)."""
    if raw is None:
        return dict(_EMPTY_COMPONENT_PCT)

    async def _do_query() -> dict:
        async with get_session() as session:
            result = await session.execute(
                select(Signal.vol_raw, Signal.mom_raw, Signal.volat_raw, Signal.price_raw)
                .where(
                    Signal.symbol == symbol,
                    Signal.interval == interval,
                    Signal.indicators == indicators_name,
                    Signal.signal_type == sig_type,
                    Signal.vol_raw.isnot(None),
                )
                .order_by(Signal.opened_at.desc())
                .limit(1)
            )
            row = result.first()
            if row is None:
                return dict(_EMPTY_COMPONENT_PCT)
            prev_vol, prev_mom, prev_vlt, prev_prc = row
            out: dict = {}
            for name, current, previous in (
                ("vol", raw["vol"], prev_vol),
                ("mom", raw["mom"], prev_mom),
                ("vlt", raw["vlt"], prev_vlt),
                ("prc", raw["prc"], prev_prc),
            ):
                if previous is None or abs(previous) < 1e-8:
                    out[name] = None
                else:
                    out[name] = (current - previous) / abs(previous) * 100.0
            return out

    try:
        return await run_with_db_timeout(_do_query())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] component change_pct hesaplanamadı: %s", symbol, exc)
        return dict(_EMPTY_COMPONENT_PCT)


async def _get_pt_flag() -> str:
    now = time.monotonic()
    if now - _PT_FLAG_CACHE["ts"] < _PT_FLAG_TTL:
        return _PT_FLAG_CACHE["value"]
    try:
        val = await asyncio.wait_for(
            RedisClient.get_client().get("settings:paper_trade_enabled"),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
        _PT_FLAG_CACHE["value"] = str(val) if val is not None else "1"
        _PT_FLAG_CACHE["ts"] = now
    except Exception as exc:
        logger.warning("paper_trade_enabled bayrağı okunamadı, önbellek kullanılıyor: %s", exc)
    return _PT_FLAG_CACHE["value"]


# 18 Tem 2026: research/pattern_lab/rsi_cross_combined_score_bt.py üst tercile
# (top %33) 4 kapılı doğrulandı (korelasyon+gerçek $+placebo+split-period,
# bkz. [[project_rsi_cross_combined_score]]) — ama backtest'in SABİT eşiği
# (66.28) canlı evrenin o anki dağılımına göre kalibre değil: piyasa durgunken
# evren ortalaması düşüyor, sabit sayı neredeyse hiçbir sinyali geçirmiyordu
# (canlıda ölçüldü: %12.7 geçiş, beklenen %33). Bunun yerine DİNAMİK yüzdelik
# dilim kullanılıyor — panelin rank_score'unun zaten yaptığı gibi, o anki
# CANLI evrene göre üst tercile mi diye bakılıyor, piyasa rejimine göre
# kendini otomatik ayarlıyor.
_RSI_CROSS_GATE_PERCENTILE = 0.667
_RSI_CROSS_GATE_MIN_UNIVERSE = 30
_RSI_CROSS_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}
_RSI_CROSS_SNAPSHOT_TTL = 30.0


async def _get_ranking_snapshot() -> Optional[dict]:
    """ranking:snapshot'ı (symbol -> row) sözlüğe çevirip 30sn cache'ler —
    backend (live_data_manager.py::_ranking_publish_loop) zaten ~90sn'de bir
    yeniliyor, her sinyalde ayrı Redis GET+json.loads gereksiz."""
    now = time.monotonic()
    if (
        _RSI_CROSS_SNAPSHOT_CACHE["data"] is not None
        and now - _RSI_CROSS_SNAPSHOT_CACHE["ts"] < _RSI_CROSS_SNAPSHOT_TTL
    ):
        return _RSI_CROSS_SNAPSHOT_CACHE["data"]
    try:
        raw = await asyncio.wait_for(
            RedisClient.get_client().get("ranking:snapshot"),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
        if not raw:
            return _RSI_CROSS_SNAPSHOT_CACHE["data"]
        rows = json.loads(raw)
        by_symbol = {row["symbol"]: row for row in rows if "symbol" in row}
        _RSI_CROSS_SNAPSHOT_CACHE["data"] = by_symbol
        _RSI_CROSS_SNAPSHOT_CACHE["ts"] = now
        return by_symbol
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("ranking:snapshot okunamadı, önbellek kullanılıyor: %s", exc)
        return _RSI_CROSS_SNAPSHOT_CACHE["data"]


async def _rsi_cross_gate_passed(symbol: str, signal_type: str) -> bool:
    """RSI Cross combined skorunun, O ANKİ CANLI EVRENE göre üst tercile'de
    olup olmadığını kontrol eder — SADECE rsi_cross_live stratejisi için
    (diğer paper trade yöneticileri etkilenmez). Yön-ayarlama (adj = raw
    Long için, 100-raw Short için) TÜM evrene aynı sinyalin yönüymüş gibi
    uygulanıp yüzdelik dilim hesaplanıyor — böylece Long ve Short için ayrı
    ama tutarlı, simetrik bir karşılaştırma evreni oluşuyor. Veri yoksa/
    sembol snapshot'ta yoksa/evren çok küçükse FAIL-OPEN: gate atlanır,
    sinyal normal açılır — Redis/ranking aksaklığı stratejiyi sessizce
    tamamen durdurmasın (kullanıcı kararı, 18 Tem 2026)."""
    snapshot = await _get_ranking_snapshot()
    if not snapshot or symbol not in snapshot:
        return True
    raw = snapshot[symbol].get("rsi_cross_combined")
    if raw is None:
        return True

    universe = [
        row["rsi_cross_combined"]
        for row in snapshot.values()
        if row.get("rsi_cross_combined") is not None
    ]
    if len(universe) < _RSI_CROSS_GATE_MIN_UNIVERSE:
        return True

    if signal_type == "Long":
        my_adj = raw
        universe_adj = universe
    else:
        my_adj = 100.0 - raw
        universe_adj = [100.0 - v for v in universe]

    percentile = sum(1 for v in universe_adj if v < my_adj) / len(universe_adj)
    return percentile >= _RSI_CROSS_GATE_PERCENTILE


_MIN_BARS: dict[str, int] = {
    "1m": 300,
    "5m": 100,
    "15m": 50,
    "30m": 50,
    "1h": 50,
    "4h": 50,
    "6h": 50,
    "8h": 50,
    "12h": 50,
    "1d": 50,
}

_MTF_HIGHER: dict[str, list[str]] = {
    "1m": ["5m", "15m"],
    "5m": ["15m", "1h"],
    "15m": ["1h", "4h"],
    "1h": ["4h", "1d"],
    "4h": ["1d"],
    "1d": [],
}

_SIGNAL_GENERATION_TFS = {"5m", "15m"}

_HTF_CONFIRM_TFS = ["4h", "6h", "8h", "12h", "1d"]

# "Ultra hizalama" (research/pattern_lab/ha_cross_mtf_alignment_bt.py, v2-14/v2-15,
# 10-11 Tem 2026) — YUKARIDAKİ _HTF_CONFIRM_TFS'ten (15m HA_Cross + sadece Long +
# 4h/6h/8h/12h/1d, hiç backtest edilmemiş) FARKLI, bağımsız ve doğrulanmış bir
# katman: 5m HA_Cross sinyali + 15m/1h/4h'nin TAMAMININ (3/3) sinyalle aynı yönde
# olması. Split-period'da sağlam çıktı (PF 1.45→2.35 iki yarıda da baseline üstü,
# OOS $1683/ay). Sadece ETİKETLEME amaçlı — hiçbir sinyali engellemez/değiştirmez.
_HA_ULTRA_CONFIRM_TFS = ["15m", "1h", "4h"]


async def _count_ha_ultra_confirm(
    symbol: str,
    signal_type: str,
    symbol_buffers: Optional[dict] = None,
) -> Optional[int]:
    """15m/1h/4h'nin SON KAPANMIŞ barının HA yönü sinyal yönüyle kaçının
    uyuştuğunu sayar (0-3). Son satır tick güncellemesiyle henüz kapanmamış
    (forming) olabileceği için atılır — research/pattern_lab/mtf_helpers.py
    ile aynı disiplin. Herhangi bir TF için veri yetersizse None (belirsiz,
    0'dan ayrı tutulur)."""
    count = 0
    for tf in _HA_ULTRA_CONFIRM_TFS:
        try:
            if symbol_buffers is not None:
                df = symbol_buffers.get(tf)
            else:
                df = await RedisClient.get_mtf_klines(symbol, tf)
            if df is None or len(df) < 2:
                return None
            closed = df.iloc[:-1]
            if closed.empty or "ha_open" not in closed.columns or "ha_close" not in closed.columns:
                return None
            last = closed.iloc[-1]
            ha_open, ha_close = last["ha_open"], last["ha_close"]
            if pd.isna(ha_open) or pd.isna(ha_close):
                return None
            ha_bull = float(ha_close) > float(ha_open)
            if (signal_type == "Long" and ha_bull) or (signal_type == "Short" and not ha_bull):
                count += 1
        except Exception as exc:
            logger.debug("HA ULTRA konfirmasyonu [%s %s] atlandı: %s", symbol, tf, exc)
            return None
    return count


async def _count_htf_ha_bullish(
    symbol: str,
    signal_type: str,
    symbol_buffers: Optional[dict] = None,
) -> int:
    """HTF buffer'larında kaç TF'nin HA yönü sinyal yönüyle uyuştuğunu sayar."""
    count = 0
    for tf in _HTF_CONFIRM_TFS:
        try:
            if symbol_buffers is not None:
                df = symbol_buffers.get(tf)
            else:
                df = await RedisClient.get_mtf_klines(symbol, tf)
            if df is None or df.empty:
                continue
            if "ha_open" not in df.columns or "ha_close" not in df.columns:
                continue
            last = df.iloc[-1]
            ha_bull = float(last["ha_close"]) > float(last["ha_open"])
            if (signal_type == "Long" and ha_bull) or (signal_type == "Short" and not ha_bull):
                count += 1
        except Exception as exc:
            logger.debug("HA HTF sayımı [%s %s] atlandı: %s", symbol, tf, exc)
    return count


async def _compute_mtf_score(
    symbol: str,
    interval: str,
    signal_type: str,
    symbol_buffers: Optional[dict] = None,
) -> float:
    """
    Üst TF'lerin ST yönüne göre MTF konfirmasyon skoru döner (0 / 50 / 100).

    Her üst TF için sinyal yönüyle ST direction uyuşuyorsa +1 puan.
    Toplam puan normalize edilerek 0-100 arasında döner.
    """
    higher_tfs = _MTF_HIGHER.get(interval, [])
    if not higher_tfs:
        return 100.0

    confirmed = 0
    checked = 0
    for tf in higher_tfs:
        try:
            if symbol_buffers is not None:
                df = symbol_buffers.get(tf)
            else:
                df = await RedisClient.get_mtf_klines(symbol, tf)
            if df is None or df.empty or "st_direction" not in df.columns:
                continue
            valid = df["st_direction"].dropna()
            if valid.empty:
                continue
            st_bullish = float(valid.iloc[-1]) == -1
            checked += 1
            if (signal_type == "Long" and st_bullish) or (
                signal_type == "Short" and not st_bullish
            ):
                confirmed += 1
        except Exception as exc:
            logger.debug("MTF ST konfirmasyonu [%s %s] atlandı: %s", symbol, tf, exc)

    if checked == 0:
        return 100.0

    return round(confirmed / checked * 100)


def _compute_devisso_score(df: pd.DataFrame) -> Optional[float]:
    """
    Δprice% / ΔRSI(14) → EMA(7) → percentile rank (son 100 bar) → 0-100.
    ERSI (RSI Verimliliği): fiyat birim RSI başına ne kadar hareket etti.
    Yüksek → verimli (fiyat az RSI ile çok hareket etti, trend sağlıklı).
    Düşük  → verimsiz (aynı hareket için RSI çok yoruldu, trend zorlanıyor).
    """
    try:
        if len(df) < 30:
            return None
        df = truncate_after_gap(df)
        if len(df) < 30:
            return None
        close = df["close"].astype(float)
        rsi = calculate_rsi(df, period=14)
        price_pct = close.pct_change() * 100.0
        raw = price_pct / rsi.diff().replace(0.0, np.nan)
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.dropna()
        if len(valid) < 20:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        rank = float((recent < current).sum()) / len(recent)
        return round(rank * 100.0, 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _compute_candle_pattern(df: pd.DataFrame) -> str:
    """
    Son mumda tespit edilen candlestick pattern(ler)i döner.
    Örnek: "+HAMMER,-SHOOTINGSTAR" ya da "-"
    Değer: 100 = bullish (+), -100 = bearish (-)
    """
    try:
        import pandas_ta_classic as _pta  # pylint: disable=import-outside-toplevel

        if len(df) < 5:
            return "-"
        df_cdl = df[["open", "high", "low", "close"]].copy().astype(float)
        df_cdl.ta.cores = 0
        result = df_cdl.ta.cdl_pattern(name="all")
        if result is None or result.empty:
            return "-"
        last = result.iloc[-1]
        found = last[last != 0]
        if found.empty:
            return "-"
        parts = []
        for col, val in found.items():
            name = str(col).replace("CDL_", "").split("_")[0]
            sign = "+" if val > 0 else "-"
            parts.append(f"{sign}{name}")
        return ",".join(parts)
    except Exception:  # pylint: disable=broad-exception-caught
        return "-"


def _classify_candle(last: pd.Series) -> Tuple[str, Optional[float], Optional[float]]:
    """Son mumun gövde/üst fitil/alt fitil oranına göre baskın kısmını VE
    sayısal gövde/fitil yüzdelerini döner: (kategori, body_pct, wick_pct).
    wick_pct = üst+alt fitil toplamı (=100-body_pct) — Hoca'nın Sinyal Mumu
    Ratioları'ndaki Body Ratio/Wick Length metriklerinin sayısal karşılığı
    (13 Ağu 2026 — önceden sadece kategori tutulup sayı atılıyordu)."""
    rng = last["high"] - last["low"]
    if rng <= 0:
        return "belirsiz", None, None
    upper = max(last["open"], last["close"])
    lower = min(last["open"], last["close"])
    body = abs(last["close"] - last["open"]) / rng * 100
    upper_wick = (last["high"] - upper) / rng * 100
    lower_wick = (lower - last["low"]) / rng * 100
    parts = {"govde": body, "ust_fitil": upper_wick, "alt_fitil": lower_wick}
    kategori = max(parts, key=parts.get)
    return kategori, round(body, 2), round(upper_wick + lower_wick, 2)


def _compute_signal_extras(df: pd.DataFrame) -> Optional[dict]:
    """Sinyal Mumu Ratioları'nın (Hoca Telegram külliyatı, S29 — orijinal
    Pine dosyası) kalan metrikleri: RSI(14)/MFI(14)/RSI-of-MACD(12,26,9,14)
    bar-to-bar değişimi + OBV seviyesi, sinyal barında. VPMV'nin 4
    bileşeninden (vol/mom/vlt/prc) BAĞIMSIZ, kendi ham hesabı — isim
    çakışmasın diye "signal_" önekiyle (13 Ağu 2026)."""
    if len(df) < 35:  # MACD(26) + üstüne RSI(14) ısınma payı
        return None

    out: dict = {"rsi_change": None, "mfi_change": None, "macd_change": None, "obv": None}
    try:
        rsi = calculate_rsi(df, period=14)
        out["rsi_change"] = float(rsi.diff().iloc[-1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("signal_extras rsi_change hesaplanamadı: %s", exc)
    try:
        mfi = calculate_mfi(df, period=14)
        out["mfi_change"] = float(mfi.diff().iloc[-1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("signal_extras mfi_change hesaplanamadı: %s", exc)
    try:
        macd_line, _, _ = calculate_macd(df, fast=12, slow=26, signal=9)
        macd_rsi = calculate_rsi(pd.DataFrame({"macd": macd_line}), period=14, price_col="macd")
        out["macd_change"] = float(macd_rsi.diff().iloc[-1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("signal_extras macd_change hesaplanamadı: %s", exc)
    try:
        obv = calculate_obv(df)
        out["obv"] = float(obv.iloc[-1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("signal_extras obv hesaplanamadı: %s", exc)
    return out


_FVG_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_FVG_LOOKBACK = 30


def _detect_fvg_in_df(df: pd.DataFrame, sig_type: str, entry_price: float) -> bool:
    """Entry fiyatının içinde kaldığı aktif bir FVG var mı? (retest senaryosu)

    Not: Kütüphanenin MitigatedIndex'i fiyat gap kenarına DEĞDİĞİ an dolar sayar;
    o tanımla "dolmamış + fiyat içinde" imkânsızdır. Burada gap, fiyat içinden
    TAMAMEN geçince (boğa: low < Bottom, ayı: high > Top) geçersiz sayılır.
    """
    if len(df) < 3:
        return False
    try:
        from smartmoneyconcepts import smc as _smc_lib  # pylint: disable=import-outside-toplevel

        df_smc = df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
        for col in df_smc.columns:
            df_smc[col] = df_smc[col].astype(float)
        fvg_df = _smc_lib.fvg(df_smc)
        direction = 1 if sig_type == "Long" else -1
        lows = df_smc["low"].to_numpy()
        highs = df_smc["high"].to_numpy()
        for i in range(len(fvg_df)):
            fvg_val = fvg_df["FVG"].iloc[i]
            if np.isnan(fvg_val) or fvg_val != direction:
                continue
            top = float(fvg_df["Top"].iloc[i])
            bot = float(fvg_df["Bottom"].iloc[i])
            if not bot <= entry_price <= top:
                continue
            if direction == 1 and np.any(lows[i + 2 :] < bot):
                continue  # boğa gap'i tamamen dolmuş
            if direction == -1 and np.any(highs[i + 2 :] > top):
                continue  # ayı gap'i tamamen dolmuş
            return True
        return False
    except Exception:  # pylint: disable=broad-exception-caught
        return False


async def _compute_fvg(symbol: str, sig_type: str, entry_price: float) -> str:
    """Tüm TF'lerde aktif FVG var mı? Sonuç: '1h,4h' veya '-'."""
    matched: list[str] = []
    try:
        for tf in _FVG_TIMEFRAMES:
            try:
                df_tf = await RedisClient.get_mtf_klines(symbol, tf)
                if df_tf is None or df_tf.empty:
                    continue
                if not {"open", "high", "low", "close", "volume"}.issubset(df_tf.columns):
                    continue
                if _detect_fvg_in_df(df_tf, sig_type, entry_price):
                    matched.append(tf)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("FVG hesabı [%s] atlandı: %s", symbol, exc)
    return ",".join(matched) if matched else "-"


async def process_and_enrich_signals(
    symbol: str,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    interval: str,
    oi_data: Optional[str] = None,
    symbol_buffers: Optional[dict] = None,
    metrics_calculator: Optional[
        Callable[[pd.DataFrame, pd.DataFrame, str], Awaitable[pd.DataFrame]]
    ] = None,
    dry_run: bool = False,
) -> None:
    """
    Bir sembol için teknik sinyalleri hesaplar, finansal metriklerle zenginleştirir
    ve veritabanına kaydeder.

    metrics_calculator: verilirse calculate_metrics yerine bu async çağrılır (ör.
    signal_service.py'nin ProcessPoolExecutor sarmalayıcısı). None ise davranış
    mevcut senkron çağrıyla birebir aynıdır — live_data_manager.py bu parametreyi
    hiç geçmez, dolayısıyla etkilenmez.

    dry_run: True ise sinyal sonuna kadar hesaplanır/loglanır ama DB'ye yazılmaz ve
    paper trade tetiklenmez (signal_service.py'nin Faz 4 dry-run gözlem penceresi
    için — varsayılan False, live_data_manager.py etkilenmez).
    """
    logger.info(
        f"[{symbol}] Sinyal işleme başlatıldı - DataFrame boyutu: {len(df)}, Ref boyutu: {len(ref_df)}"
    )

    if df.empty or ref_df.empty:
        logger.warning(f"[{symbol}] Veri çerçevelerinden biri boş, atlanıyor.")
        return

    min_bars = _MIN_BARS.get(interval, 50)
    if len(df) < min_bars:
        logger.warning(
            "[%s] %s: yetersiz bar (%d/%d), sinyal üretilmiyor",
            symbol,
            interval,
            len(df),
            min_bars,
        )
        return

    if interval not in _SIGNAL_GENERATION_TFS:
        return

    # 1. Finansal Metrikleri Hesapla
    alpha, beta, latest_metrics = None, None, {}
    try:
        if len(df) >= 50 and len(ref_df) >= 50:
            df_prepared = df.copy()
            df_prepared.index = pd.Index(pd.to_datetime(df_prepared["open_time"], unit="ms"))
            ref_df_prepared = ref_df.copy()
            ref_df_prepared.index = pd.Index(
                pd.to_datetime(ref_df_prepared["open_time"], unit="ms")
            )

            if metrics_calculator is not None:
                df_with_metrics = await metrics_calculator(df_prepared, ref_df_prepared, interval)
            else:
                df_with_metrics = calculate_metrics(df_prepared, ref_df_prepared, interval=interval)
            latest_metrics = df_with_metrics.iloc[-1].to_dict()
            alpha = latest_metrics.get("alpha")
            beta = latest_metrics.get("beta")

            alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
            beta_str = f"{beta:.4f}" if beta is not None else "N/A"
            logger.info(f"[{symbol}] Finansal metrikler -> Alpha: {alpha_str}, Beta: {beta_str}")
        else:
            logger.warning(f"[{symbol}] Metrik için yeterli veri yok (gerekli: 50), atlanıyor.")
    except Exception as e:
        logger.error(f"[{symbol}] Finansal metrik hatası: {e}", exc_info=False)

    # 2. Teknik Sinyalleri Hesapla
    try:
        logger.info(f"[{symbol}] Teknik sinyal hesaplama başlatılıyor...")
        technical_signals = await signal_engine.calculate_all_signals(
            df, symbol=symbol, interval=interval
        )
        logger.info(
            f"[{symbol}] Teknik sinyal hesaplama tamamlandı - {len(technical_signals) if technical_signals else 0} tür"
        )
    except Exception as e:
        logger.error(f"[{symbol}] Teknik sinyal hatası: {e}", exc_info=True)
        return

    if not technical_signals:
        logger.info(f"[{symbol}] Teknik sinyal bulunamadı.")
        return

    min_vpmv = float(Config.VPM.get("MIN_SCORE", 40.0))
    vpm_weights = Config.VPM.get("WEIGHTS")

    # Mevcut ST yönünü bir kez oku (st_confirmed hesabı için)
    st_direction = None
    if "st_direction" in df.columns and len(df) >= 2:
        st_dir_val = df["st_direction"].iloc[-2]
        if st_dir_val is not None and not (
            isinstance(st_dir_val, float) and st_dir_val != st_dir_val
        ):
            st_direction = float(st_dir_val)  # -1=bullish, 1=bearish

    # 2 Ağu 2026 (Fable 5 performans denetimi): z_score/regime/ADX/BTC-z
    # SADECE df/ref_df'e bağlı, sig_type'a bağlı DEĞİL (yön-bağımsız fiyat
    # hesapları) — aynı bar'da birden fazla gösterge (ör. RSI_Cross VE
    # HA_Cross) aynı anda tetiklenirse eskiden aşağıdaki döngü İÇİNDE her
    # signal_data için tekrar tekrar hesaplanıyordu (canlı log kanıtı:
    # aynı sembol+zaman için ort. 2x tekrar). st_direction ile aynı desende
    # döngü dışına, bir kez hesaplanıyor.
    z_score_entry = None
    try:
        df_g = truncate_after_gap(df)
        if len(df_g) >= 210 and "close" in df_g.columns:
            closes = df_g["close"].astype(float)
            ema200 = closes.ewm(span=200, adjust=False).mean()
            std200 = closes.rolling(200).std()
            z_score_entry = round(
                float((closes.iloc[-1] - ema200.iloc[-1]) / (std200.iloc[-1] + 1e-12)), 3
            )
    except Exception as exc:
        logger.debug("z_score_entry hesaplanamadı [%s]: %s", symbol, exc)

    regime_trend: Optional[str] = None
    volatility_regime: Optional[str] = None
    try:
        if len(df) >= 28:
            adx_series, _, _ = calculate_adx(df)
            adx_val = float(adx_series.iloc[-1])
            regime_trend = "trending" if adx_val > 25 else "ranging" if adx_val < 20 else "neutral"

            atr_series = calculate_atr(df, period=Config.ATR_PERIOD)
            atr_pct = float(normalize_volatility_0_100(atr_series).iloc[-1])
            volatility_regime = "high" if atr_pct > 70 else "low" if atr_pct < 30 else "normal"
    except Exception as exc:
        logger.debug("volatility_regime hesaplanamadı [%s]: %s", symbol, exc)

    btc_z: Optional[float] = None
    btc_trend_str: Optional[str] = None
    try:
        btc_df = ref_df if not ref_df.empty else None
        if btc_df is not None and len(btc_df) >= 210:
            btc_closes = btc_df["close"].astype(float)
            btc_ema = btc_closes.ewm(span=200, adjust=False).mean()
            btc_std = btc_closes.rolling(200).std()
            btc_z = round(
                float((btc_closes.iloc[-1] - btc_ema.iloc[-1]) / (btc_std.iloc[-1] + 1e-12)), 3
            )
            btc_trend_str = "bullish" if btc_z > 0.5 else "bearish" if btc_z < -0.5 else "neutral"
    except Exception as exc:
        logger.debug("BTC trend hesaplanamadı: %s", exc)

    for signal_name, signal_list in technical_signals.items():
        if not isinstance(signal_list, list) or not signal_list:
            continue

        for signal_data in signal_list:
            try:
                sig_type = signal_data.get("signal_type", "Long")

                # st_confirmed: ST yönüyle uyumlu mu? (supertrend kendi sinyali için her zaman True)
                if signal_name == "supertrend" or st_direction is None:
                    st_confirmed = True
                else:
                    st_bullish = st_direction == -1
                    st_confirmed = (sig_type == "Long" and st_bullish) or (
                        sig_type == "Short" and not st_bullish
                    )
                    if not st_confirmed:
                        logger.info(
                            f"[{symbol}] {signal_name} {sig_type} ST onaysız "
                            f"(ST {'bullish' if st_bullish else 'bearish'}) — yine de kaydediliyor"
                        )

                # 3. VPMV Hesapla
                vpms_score: Optional[float] = None
                vol_s = mom_s = vlt_s = prc_s = None
                if len(df) >= 1:
                    candle_kategori, body_pct, wick_pct = _classify_candle(df.iloc[-1])
                else:
                    candle_kategori, body_pct, wick_pct = None, None, None
                try:
                    vol_s, mom_s, vlt_s, prc_s = compute_components(df, sig_type)
                    vpms_score = VPMCalculator.calculate(
                        vol_score=vol_s,
                        momentum_score=mom_s,
                        vlt_score=vlt_s,
                        price_score=prc_s,
                        weights=vpm_weights,
                    )
                    logger.info(
                        f"[{symbol}] VPMV | type={sig_type} "
                        f"V={vol_s:.1f} M={mom_s:.1f} Vlt={vlt_s:.1f} P={prc_s:.1f} "
                        f"→ score={vpms_score:.1f}"
                    )
                except Exception as vpm_err:
                    logger.warning(f"[{symbol}] VPMV hesaplama atlandı: {vpm_err}")

                # Pre-signal directional volume log (test)
                try:
                    _pre = df.iloc[-6:-1]
                    _hl = (_pre["high"] - _pre["low"]).clip(lower=1e-8)
                    _bv = (_pre["volume"] * (_pre["close"] - _pre["low"]) / _hl).sum()
                    _sv = (_pre["volume"] * (_pre["high"] - _pre["close"]) / _hl).sum()
                    _tot = _bv + _sv
                    _buy_pct = _bv / _tot * 100 if _tot > 0 else 50.0
                    logger.info(
                        "PREVOL | %s | %s | %s | buy_pct=%.1f",
                        symbol,
                        sig_type,
                        interval,
                        _buy_pct,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

                # CVD slope (normalize, -1..+1)
                _cvd_slope = compute_cvd_slope(df)
                if _cvd_slope is not None:
                    logger.info(
                        "CVD | %s | %s | %s | slope=%.4f", symbol, sig_type, interval, _cvd_slope
                    )

                # 4. Minimum skor filtresi (B kapısı)
                if vpms_score is not None and vpms_score < min_vpmv:
                    logger.info(
                        f"[{symbol}] VPMV={vpms_score:.1f} < {min_vpmv} — sinyal atlandı ({signal_name})"
                    )
                    continue

                # 5. MTF konfirmasyon skoru hesapla
                mtf_score = await _compute_mtf_score(symbol, interval, sig_type, symbol_buffers)
                logger.info(
                    f"[{symbol}] MTF konfirmasyon | {interval} {sig_type} "
                    f"→ score={mtf_score:.0f} "
                    f"(higher TFs: {_MTF_HIGHER.get(interval, [])})"
                )

                # 6. Z-score filtresi — ham değer artık döngü dışında (yukarıda)
                # bir kez hesaplanıyor, burada sadece sig_type'a bağlı eşik
                # kontrolü kalıyor.
                if z_score_entry is not None:
                    _is_long = sig_type == "Long"
                    _z_min = Config.VPM.get("LONG_Z_MIN" if _is_long else "SHORT_Z_MIN")
                    _z_max = Config.VPM.get("LONG_Z_MAX" if _is_long else "SHORT_Z_MAX")
                    if (_z_min is not None and z_score_entry < _z_min) or (
                        _z_max is not None and z_score_entry > _z_max
                    ):
                        logger.info(
                            "[%s] Z-score=%.3f filtre dışı (%s, min=%s max=%s) — sinyal atlandı",
                            symbol,
                            z_score_entry,
                            sig_type,
                            _z_min,
                            _z_max,
                        )
                        continue

                indicators_name = signal_data.get("indicators", "")

                # HTF HA hizalanması (sadece HA_Cross sinyalleri için)
                htf_bull_count = 0
                ha_ultra_confirm: Optional[int] = None
                if indicators_name == "HA_Cross":
                    htf_bull_count = await _count_htf_ha_bullish(symbol, sig_type, symbol_buffers)
                    if interval == "5m":
                        ha_ultra_confirm = await _count_ha_ultra_confirm(
                            symbol, sig_type, symbol_buffers
                        )

                is_confluence = (
                    indicators_name == "HA_Cross"
                    and sig_type == "Long"
                    and interval == "15m"
                    and htf_bull_count >= Config.HA_HTF_MIN_COUNT
                )

                if is_confluence:
                    logger.info(
                        "[%s] ★ KONFLUANS SİNYALİ %s %s | HTF=%d/%d",
                        symbol,
                        interval,
                        sig_type,
                        htf_bull_count,
                        len(_HTF_CONFIRM_TFS),
                    )

                # 7. Sinyali işle
                current_price = float(df["close"].iloc[-1]) if len(df) >= 1 else None
                enriched_signal = {
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": signal_data.get("indicators"),
                    "signal_type": sig_type,
                    "opened_at": signal_data.get("timestamp"),
                    "open_price": signal_data.get("price"),
                    "vpms_score": float(vpms_score) if vpms_score is not None else None,
                    "mtf_score": mtf_score,
                    "st_confirmed": st_confirmed,
                    "rsi": signal_data.get("rsi"),
                    "strength": signal_data.get("strength"),
                    "atr": signal_data.get("atr"),
                    "alpha": alpha,
                    "beta": beta,
                    "sharpe_ratio": latest_metrics.get("sharpe_ratio"),
                    "sortino_ratio": latest_metrics.get("sortino_ratio"),
                    "calmar_ratio": latest_metrics.get("calmar_ratio"),
                    "information_ratio": latest_metrics.get("information_ratio"),
                    "oi_data": oi_data,
                    "z_score_entry": z_score_entry,
                    "is_confluence": is_confluence,
                    "htf_bull_count": htf_bull_count,
                    "ha_ultra_confirm": ha_ultra_confirm,
                    "vol_score": vol_s,
                    "mom_score": mom_s,
                    "volat_score": vlt_s,
                    "price_score": prc_s,
                    "candle_kategori": candle_kategori,
                    "body_pct": body_pct,
                    "wick_pct": wick_pct,
                }

                enriched_signal["all_up"] = await _compute_all_up(
                    symbol, interval, indicators_name, sig_type, vol_s, mom_s, vlt_s, prc_s
                )

                _vpmv_pre_avg = _vpmv_slope = _vpmv_ratio = None
                try:
                    _pre_avg, _slope, _vpmv_sig = compute_pre(df, sig_type)
                    logger.info(
                        "[%s] VPMV pre raw: df_len=%d pre_avg=%s vpmv_sig=%s",
                        symbol,
                        len(df),
                        _pre_avg,
                        _vpmv_sig,
                    )
                    if _pre_avg is not None and _vpmv_sig is not None:
                        _vpmv_pre_avg = round(_pre_avg, 2)
                        _vpmv_slope = round(_slope, 2)
                        _vpmv_ratio = round(_vpmv_sig / _pre_avg, 4) if _pre_avg > 0 else None
                except Exception as _vpe:
                    logger.warning("[%s] VPMV pre hesaplama hatası: %s", symbol, _vpe)

                enriched_signal["vpmv_pre_avg"] = _vpmv_pre_avg
                enriched_signal["vpmv_slope"] = _vpmv_slope
                enriched_signal["vpmv_ratio"] = _vpmv_ratio

                # 13 Ağu 2026: 4 HAM bileşenin (hacim/RSI seviyesi/ATR/kapanış
                # fiyatı) BU sinyaldeki değeri + BİR ÖNCEKİ aynı (symbol,
                # interval, indicators, signal_type) sinyaline göre yüzdesel
                # değişimi — Sinyal Mumu Ratioları'nın `volume/lastSignalVolume-1`
                # mantığıyla aynı baseline.
                _raw_components_map = {
                    "vol": ("vol_raw", "vol_change_pct"),
                    "mom": ("mom_raw", "mom_change_pct"),
                    "vlt": ("volat_raw", "volat_change_pct"),
                    "prc": ("price_raw", "price_change_pct"),
                }
                for _raw_key, _pct_key in _raw_components_map.values():
                    enriched_signal[_raw_key] = None
                    enriched_signal[_pct_key] = None
                try:
                    _raw_comp = compute_raw_components(df, sig_type)
                    if _raw_comp is not None:
                        for _name, (_raw_key, _) in _raw_components_map.items():
                            enriched_signal[_raw_key] = round(_raw_comp[_name], 4)
                        _pct_comp = await _compute_component_change_pct(
                            symbol, interval, indicators_name, sig_type, _raw_comp
                        )
                        for _name, (_, _pct_key) in _raw_components_map.items():
                            _pct = _pct_comp.get(_name)
                            enriched_signal[_pct_key] = round(_pct, 2) if _pct is not None else None
                except Exception as _vpce:
                    logger.warning("[%s] VPMV ham-bileşen hesaplama hatası: %s", symbol, _vpce)

                # 13 Ağu 2026: Sinyal Mumu Ratioları'nın kalan metrikleri —
                # RSI/MFI/MACD'nin sinyal barındaki değişimi + OBV seviyesi.
                enriched_signal["signal_rsi_change"] = None
                enriched_signal["signal_mfi_change"] = None
                enriched_signal["signal_macd_change"] = None
                enriched_signal["signal_obv"] = None
                try:
                    _extras = _compute_signal_extras(df)
                    if _extras is not None:
                        enriched_signal["signal_rsi_change"] = (
                            round(_extras["rsi_change"], 4) if _extras["rsi_change"] is not None else None
                        )
                        enriched_signal["signal_mfi_change"] = (
                            round(_extras["mfi_change"], 4) if _extras["mfi_change"] is not None else None
                        )
                        enriched_signal["signal_macd_change"] = (
                            round(_extras["macd_change"], 4) if _extras["macd_change"] is not None else None
                        )
                        enriched_signal["signal_obv"] = (
                            round(_extras["obv"], 2) if _extras["obv"] is not None else None
                        )
                except Exception as _sxe:
                    logger.warning("[%s] Sinyal Mumu ekstra metrik hatası: %s", symbol, _sxe)

                # A/B/C deneyi: aynı skor vekil ve yönsüz hacimle (analiz için, panelde yok)
                _vpmv_proxy = _vpmv_total = None
                try:
                    _p, _, _ = compute_pre(df, sig_type, volume_mode="proxy")
                    _t, _, _ = compute_pre(df, sig_type, volume_mode="total")
                    _vpmv_proxy = round(_p, 2) if _p is not None else None
                    _vpmv_total = round(_t, 2) if _t is not None else None
                except Exception as _vve:
                    logger.debug("[%s] VPMV varyant hesabı atlandı: %s", symbol, _vve)
                enriched_signal["vpmv_pre_proxy"] = _vpmv_proxy
                enriched_signal["vpmv_pre_total"] = _vpmv_total

                min_ratio = float(Config.VPM.get("MIN_RATIO", 1.3))
                if _vpmv_ratio is not None and _vpmv_ratio < min_ratio:
                    logger.info(
                        "[%s] VPMV ratio=%.3f < %.2f — sinyal atlandı (%s)",
                        symbol,
                        _vpmv_ratio,
                        min_ratio,
                        signal_name,
                    )
                    continue
                enriched_signal["cvd_slope"] = _cvd_slope

                _deviso = _compute_devisso_score(df)
                enriched_signal["devisso_score"] = _deviso
                logger.info(
                    "DEVISSO | %s | %s | %s | score=%s",
                    symbol,
                    sig_type,
                    interval,
                    _deviso,
                )

                _vp_buy, _vp_sell = compute_vp_score(df)
                _vp_score = round(_vp_buy - _vp_sell, 2)
                enriched_signal["vp_buy_avg"] = _vp_buy
                enriched_signal["vp_sell_avg"] = _vp_sell
                enriched_signal["vp_score"] = _vp_score

                _vpr_buy, _vpr_sell = compute_vp_score(df, use_real_volume=True)
                _vp_score_real = None if np.isnan(_vpr_buy) else round(_vpr_buy - _vpr_sell, 2)
                enriched_signal["vp_score_real"] = _vp_score_real
                logger.info(
                    "VP | %s | %s | %s | buy=%.1f sell=%.1f score=%.1f",
                    symbol,
                    sig_type,
                    interval,
                    _vp_buy,
                    _vp_sell,
                    _vp_score,
                )

                # Premium/Discount zone KASITLI OLARAK kaldırıldı (19 Tem 2026,
                # kullanıcı kararı — işe yaramıyordu). Sadece market_structure kaldı.
                _mkt_structure = compute_smc_market_structure(df, sig_type)
                enriched_signal["market_structure"] = _mkt_structure
                logger.info(
                    "SMC | %s | %s | %s | struct=%s", symbol, sig_type, interval, _mkt_structure
                )

                _fvg_tfs = await _compute_fvg(
                    symbol, sig_type, float(enriched_signal.get("open_price", current_price or 0))
                )
                enriched_signal["fvg_tfs"] = _fvg_tfs
                logger.info("FVG | %s | %s | tfs=%s", symbol, sig_type, _fvg_tfs)

                _cdl = _compute_candle_pattern(df)
                enriched_signal["candle_pattern"] = _cdl
                logger.info("CDL | %s | %s | pattern=%s", symbol, sig_type, _cdl)

                # 2 Ağu 2026 (migration 029): bu alanlar zaten yukarıda (6.3/6.5)
                # hesaplanmıştı ama sadece _pt_kwargs'a (paper_trade'e) gidiyordu,
                # signals tablosuna hiç yazılmıyordu — artık enriched_signal'e de
                # ekleniyor ki signal_lifecycle_manager.process() Signal satırına
                # yazsın (simetri, "metrikler dağınık" toparlaması).
                enriched_signal["regime_trend"] = regime_trend
                enriched_signal["volatility_regime"] = volatility_regime
                enriched_signal["btc_z_score"] = btc_z
                enriched_signal["btc_trend"] = btc_trend_str
                _opened_at = enriched_signal.get("opened_at")
                if _opened_at is not None:
                    enriched_signal["hour_utc"] = _opened_at.hour
                    enriched_signal["day_of_week"] = _opened_at.weekday()

                _funding: Optional[float] = None
                try:
                    import json as _json

                    _ticker_raw = await asyncio.wait_for(
                        RedisClient.get_client().get(f"ticker:{symbol}"),
                        timeout=SAFE_EXTERNAL_TIMEOUT,
                    )
                    if _ticker_raw:
                        _funding = _json.loads(_ticker_raw).get("funding_rate")
                except Exception as exc:
                    logger.debug("funding_rate okunamadı [%s]: %s", symbol, exc)
                enriched_signal["funding_rate"] = _funding

                # 2 Ağu 2026 (Fable 5 performans denetimi): ranking:snapshot
                # ÖNCEDEN signal_lifecycle_manager.py VE (her strateji için)
                # paper_trade_manager.py'de bağımsız bağımsız okunuyordu — bir
                # sinyalde 5'e kadar ayrı Redis GET+json.loads (dakikada 250+
                # gereksiz round-trip, ölçüldü). _get_ranking_snapshot() zaten
                # 30sn cache'li — burada TEK sefer okunup enriched_signal'e
                # gömülüyor, aşağıdaki tüketiciler artık kendi okumasını yapmaz.
                _rank_entry = (await _get_ranking_snapshot() or {}).get(symbol) or {}
                enriched_signal["rank_at_entry"] = _rank_entry.get("rank")
                enriched_signal["rank_score"] = _rank_entry.get("rank_score")
                enriched_signal["vs_btc"] = _rank_entry.get("vs_btc")
                enriched_signal["rank_combined"] = _rank_entry.get("combined")
                enriched_signal["rank_rsi_cross"] = _rank_entry.get("rsi_cross_combined")
                enriched_signal["rank_z_confluence"] = _rank_entry.get("z_confluence")
                enriched_signal["rank_r_score"] = _rank_entry.get("r_score")
                enriched_signal["rank_aligned"] = _rank_entry.get("aligned")
                enriched_signal["rank_alignment_count"] = _rank_entry.get("alignment_count")

                logger.info(f"[{symbol}] Sinyal işleniyor: {signal_name} - {sig_type}")
                if dry_run:
                    logger.info(
                        "[DRY-RUN] [%s] Sinyal işlenecekti (%s - %s) — DB'ye yazılmadı, "
                        "paper trade tetiklenmedi",
                        symbol,
                        signal_name,
                        sig_type,
                    )
                    try:
                        await asyncio.wait_for(
                            RedisClient.get_client().incr("metrics:sigsvc:dry_run_signal"),
                            timeout=SAFE_EXTERNAL_TIMEOUT,
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    continue

                signal_id = await signal_lifecycle_manager.process(
                    enriched_signal, current_price=current_price
                )
                logger.info(f"[{symbol}] Sinyal işlendi: ID {signal_id}")
                if signal_id:
                    risk_manager.register(symbol)

                if signal_id and current_price:
                    _pt_flag = await _get_pt_flag()
                    if _pt_flag != "0":
                        # funding artık yukarıda (enriched_signal doldurulurken) bir
                        # kez okunuyor — signals VE paper_trades aynı değeri paylaşsın
                        # diye burada ikinci kez sorgulanmıyor (migration 029)
                        funding = enriched_signal.get("funding_rate")

                        _pt_kwargs = dict(
                            signal_data=enriched_signal,
                            signal_id=signal_id,
                            current_price=current_price,
                            btc_z_score=btc_z,
                            btc_trend=btc_trend_str,
                            funding_rate=funding,
                            regime_trend=regime_trend,
                            volatility_regime=volatility_regime,
                        )
                        if is_confluence:
                            await paper_trade_manager.on_new_signal(**_pt_kwargs)
                        await ha_cross_manager.on_new_signal(**_pt_kwargs)
                        await rsi_15m_manager.on_new_signal(**_pt_kwargs)
                        if enriched_signal.get("indicators") != "RSI_Cross(9,24)" or (
                            await _rsi_cross_gate_passed(symbol, enriched_signal["signal_type"])
                        ):
                            await rsi_cross_live_manager.on_new_signal(**_pt_kwargs)
                        else:
                            logger.info(
                                "[%s] RSI Cross gate reddetti (combined skor üst tercile altında)",
                                symbol,
                            )
                        await tf_alignment_gate.register_candidate(enriched_signal, signal_id)
                        await ta_kovalama_gate.evaluate_and_open(
                            enriched_signal,
                            signal_id,
                            current_price,
                            btc_z_score=btc_z,
                            btc_trend=btc_trend_str,
                            funding_rate=funding,
                            regime_trend=regime_trend,
                            volatility_regime=volatility_regime,
                        )

            except Exception as e:
                logger.error(f"[{symbol}] Sinyal kayıt hatası: {e}", exc_info=True)
