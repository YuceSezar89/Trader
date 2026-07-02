import time
import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger
from signals.signal_engine import signal_engine
from indicators.financial_metrics import calculate_metrics
from signals.signal_lifecycle_manager import signal_lifecycle_manager
from config import Config
from indicators.core import calculate_rsi, calculate_atr, calculate_adx
from signals.vpm_calculator import VPMCalculator
from utils.preprocessing import (
    normalize_volume_0_100,
    normalize_momentum_0_100,
    normalize_volatility_0_100,
    normalize_price_0_100,
)
from utils.redis_client import RedisClient
from utils.vpmv import compute_pre
from signals.paper_trade_manager import paper_trade_manager, ha_cross_manager, rsi_15m_manager
from signals.risk_manager import risk_manager

logger = get_logger(__name__)

_PT_FLAG_CACHE: dict = {"value": "1", "ts": 0.0}
_PT_FLAG_TTL = 30.0


async def _get_pt_flag() -> str:
    now = time.monotonic()
    if now - _PT_FLAG_CACHE["ts"] < _PT_FLAG_TTL:
        return _PT_FLAG_CACHE["value"]
    try:
        val = await RedisClient.get_client().get("settings:paper_trade_enabled")
        _PT_FLAG_CACHE["value"] = str(val) if val is not None else "1"
        _PT_FLAG_CACHE["ts"] = now
    except Exception as exc:
        logger.warning("paper_trade_enabled bayrağı okunamadı, önbellek kullanılıyor: %s", exc)
    return _PT_FLAG_CACHE["value"]

_MIN_BARS: dict[str, int] = {
    "1m":  300,
    "5m":  100,
    "15m": 50,
    "30m": 50,
    "1h":  50,
    "4h":  50,
    "6h":  50,
    "8h":  50,
    "12h": 50,
    "1d":  50,
}

_MTF_HIGHER: dict[str, list[str]] = {
    "1m":  ["5m",  "15m"],
    "5m":  ["15m", "1h"],
    "15m": ["1h",  "4h"],
    "1h":  ["4h",  "1d"],
    "4h":  ["1d"],
    "1d":  [],
}

_SIGNAL_GENERATION_TFS = {"5m", "15m"}

_HTF_CONFIRM_TFS = ["4h", "6h", "8h", "12h", "1d"]


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
            if (signal_type == "Long" and st_bullish) or \
               (signal_type == "Short" and not st_bullish):
                confirmed += 1
        except Exception as exc:
            logger.debug("MTF ST konfirmasyonu [%s %s] atlandı: %s", symbol, tf, exc)

    if checked == 0:
        return 100.0

    return round(confirmed / checked * 100)


def _compute_vpmv_scores(df: pd.DataFrame, signal_type: str) -> tuple[float, float, float, float]:
    """
    df üzerinden rolling normalize bileşen skorlarını hesaplar.

    Returns:
        (vol_score, momentum_score, vlt_score, price_score) — hepsi 0-100
    """
    side = 1.0 if signal_type == "Long" else -1.0

    # Volume: log + rolling min-max (yönsüz)
    vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])

    # Momentum: yönlü RSI delta + z-score sigmoid
    rsi_series = calculate_rsi(df, period=14)
    rsi_delta_series = rsi_series.diff().fillna(0.0) * side
    momentum_score = float(normalize_momentum_0_100(rsi_delta_series).iloc[-1])

    # Volatility: ATR percentile rank
    atr_series = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt_score = float(normalize_volatility_0_100(atr_series).iloc[-1])

    # Price: yönlü % değişim + rolling IQR
    price_pct = df["close"].pct_change().fillna(0.0) * 100.0 * side
    price_score = float(normalize_price_0_100(price_pct).iloc[-1])

    return vol_score, momentum_score, vlt_score, price_score


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


def _compute_smc(df: pd.DataFrame, sig_type: str, lookback: int = 50) -> tuple[Optional[float], str]:
    """
    Premium/Discount zone + Market Structure (BOS/CHoCH).

    pd_zone: (close - low_N) / (high_N - low_N) * 100
      0-25  → Deep Discount | 25-50 → Discount | 50-75 → Premium | 75-100 → Deep Premium

    market_structure: pivot tabanlı tespit (smartmoneyconcepts kütüphanesi).
      BOS↑  / BOS↓  — sinyal mevcut yapıyla aynı yönde (trend devamı)
      CHoCH↑ / CHoCH↓ — sinyal yapıya karşı (dönüş sinyali)
      -             — yapı belirlenemedi
    """
    try:
        if len(df) < lookback + 5:
            return None, "-"

        from smartmoneyconcepts import smc as _smc_lib  # pylint: disable=import-outside-toplevel

        df_use = df.tail(lookback).copy().reset_index(drop=True)
        high  = df_use["high"].astype(float)
        low   = df_use["low"].astype(float)
        close = df_use["close"].astype(float)

        rng_high = float(high.max())
        rng_low  = float(low.min())
        pd_zone: Optional[float] = None
        if rng_high > rng_low:
            pd_zone = round(((float(close.iloc[-1]) - rng_low) / (rng_high - rng_low)) * 100, 1)

        df_smc = df_use[["open", "high", "low", "close", "volume"]].copy()
        for col in df_smc.columns:
            df_smc[col] = df_smc[col].astype(float)

        swing_df = _smc_lib.swing_highs_lows(df_smc, swing_length=5)
        bos_df   = _smc_lib.bos_choch(df_smc, swing_df, close_break=True)

        structure_dir = 0
        for i in range(len(bos_df) - 1, -1, -1):
            bos_val   = bos_df["BOS"].iloc[i]
            choch_val = bos_df["CHOCH"].iloc[i]
            if not np.isnan(bos_val):
                structure_dir = int(bos_val)
                break
            if not np.isnan(choch_val):
                structure_dir = int(choch_val)
                break

        if structure_dir == 0:
            return pd_zone, "-"

        structure = "-"
        if structure_dir == 1:
            structure = "BOS↑" if sig_type == "Long" else "CHoCH↓"
        elif structure_dir == -1:
            structure = "BOS↓" if sig_type == "Short" else "CHoCH↑"

        return pd_zone, structure
    except Exception:  # pylint: disable=broad-exception-caught
        return None, "-"


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


_FVG_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_FVG_LOOKBACK   = 30


def _detect_fvg_in_df(df: pd.DataFrame, sig_type: str, entry_price: float) -> bool:
    """Entry fiyatının içinde kaldığı aktif (dolmamış) bir FVG var mı?"""
    if len(df) < 3:
        return False
    try:
        from smartmoneyconcepts import smc as _smc_lib  # pylint: disable=import-outside-toplevel
        df_smc = df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
        for col in df_smc.columns:
            df_smc[col] = df_smc[col].astype(float)
        fvg_df    = _smc_lib.fvg(df_smc)
        direction = 1 if sig_type == "Long" else -1
        for i in range(len(fvg_df)):
            fvg_val = fvg_df["FVG"].iloc[i]
            if np.isnan(fvg_val) or fvg_val != direction:
                continue
            mit = fvg_df["MitigatedIndex"].iloc[i]
            if not np.isnan(mit) and mit > 0:
                continue  # dolduruldu
            top = float(fvg_df["Top"].iloc[i])
            bot = float(fvg_df["Bottom"].iloc[i])
            if bot <= entry_price <= top:
                return True
        return False
    except Exception:  # pylint: disable=broad-exception-caught
        return False


async def _compute_fvg(symbol: str, sig_type: str, entry_price: float) -> str:
    """Tüm TF'lerde aktif FVG var mı? Sonuç: '1h,4h' veya '-'."""
    import io as _io  # pylint: disable=import-outside-toplevel
    matched: list[str] = []
    try:
        rc = RedisClient.get_client()
        for tf in _FVG_TIMEFRAMES:
            try:
                raw = await rc.get(f"live_kline_data:{symbol}:{tf}")
                if not raw:
                    continue
                if isinstance(raw, bytes) and raw[:4] == b"ARDF":
                    import pyarrow as _pa  # pylint: disable=import-outside-toplevel
                    reader = _pa.ipc.open_stream(raw[4:])
                    df_tf = reader.read_pandas()
                elif isinstance(raw, (bytes, str)):
                    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    df_tf = pd.read_json(_io.StringIO(raw_str), orient="split")
                else:
                    continue
                if "open_time" in df_tf.columns and "timestamp" not in df_tf.columns:
                    df_tf = df_tf.rename(columns={"open_time": "timestamp"})
                needed = {"high", "low", "close"}
                if not needed.issubset(df_tf.columns):
                    continue
                if _detect_fvg_in_df(df_tf, sig_type, entry_price):
                    matched.append(tf)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("FVG hesabı [%s] atlandı: %s", symbol, exc)
    return ",".join(matched) if matched else "-"


def _compute_vp_score(df: pd.DataFrame, lookback: int = 500) -> tuple[float, float]:
    """
    %VP Normalized Lines — PineScript birebir çeviri.

    PineScript ta.cum() → Python cumsum() (DataFrame başından itibaren)
    Normalize: rolling(lookback).min/max — PineScript ta.lowest/highest(lookback)

    Returns: (buy_positive_avg, sell_negative_avg) — her ikisi 0-100 arası
    vp_score = buy_positive_avg - sell_negative_avg  →  pozitif: alıcı baskısı
    """
    try:
        price_change = df["close"].diff().fillna(0.0)
        cum_positive = price_change.clip(lower=0.0).cumsum()
        cum_negative = (-price_change).clip(lower=0.0).cumsum()
        total_move   = (cum_positive + cum_negative).replace(0, np.nan)
        positive_pct = (cum_positive / total_move * 100).fillna(50.0)
        negative_pct = (cum_negative / total_move * 100).fillna(50.0)

        hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["volume"] * (df["close"] - df["low"]) / hl_range
        sv = df["volume"] * (df["high"] - df["close"]) / hl_range
        cum_buy  = bv.cumsum()
        cum_sell = sv.cumsum()
        total_vol = (cum_buy + cum_sell).replace(0, np.nan)
        buy_pct  = (cum_buy  / total_vol * 100).fillna(50.0)
        sell_pct = (cum_sell / total_vol * 100).fillna(50.0)

        def _norm(s: pd.Series) -> pd.Series:
            lo = s.rolling(lookback, min_periods=1).min()
            hi = s.rolling(lookback, min_periods=1).max()
            return ((s - lo) / (hi - lo + 1e-10) * 100).fillna(50.0)

        buy_pos_avg  = (_norm(buy_pct)  + _norm(positive_pct)) / 2
        sell_neg_avg = (_norm(sell_pct) + _norm(negative_pct)) / 2

        return round(float(buy_pos_avg.iloc[-1]), 2), round(float(sell_neg_avg.iloc[-1]), 2)
    except Exception:
        return 50.0, 50.0


async def process_and_enrich_signals(
    symbol: str,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    interval: str,
    oi_data: Optional[str] = None,
    symbol_buffers: Optional[dict] = None,
) -> None:
    """
    Bir sembol için teknik sinyalleri hesaplar, finansal metriklerle zenginleştirir
    ve veritabanına kaydeder.
    """
    logger.info(f"[{symbol}] Sinyal işleme başlatıldı - DataFrame boyutu: {len(df)}, Ref boyutu: {len(ref_df)}")

    if df.empty or ref_df.empty:
        logger.warning(f"[{symbol}] Veri çerçevelerinden biri boş, atlanıyor.")
        return

    min_bars = _MIN_BARS.get(interval, 50)
    if len(df) < min_bars:
        logger.warning(
            "[%s] %s: yetersiz bar (%d/%d), sinyal üretilmiyor",
            symbol, interval, len(df), min_bars,
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
            ref_df_prepared.index = pd.Index(pd.to_datetime(ref_df_prepared["open_time"], unit="ms"))

            df_with_metrics = calculate_metrics(df_prepared, ref_df_prepared, interval=interval)
            latest_metrics = df_with_metrics.iloc[-1].to_dict()
            alpha = latest_metrics.get("alpha")
            beta = latest_metrics.get("beta")

            alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
            beta_str  = f"{beta:.4f}"  if beta  is not None else "N/A"
            logger.info(f"[{symbol}] Finansal metrikler -> Alpha: {alpha_str}, Beta: {beta_str}")
        else:
            logger.warning(f"[{symbol}] Metrik için yeterli veri yok (gerekli: 50), atlanıyor.")
    except Exception as e:
        logger.error(f"[{symbol}] Finansal metrik hatası: {e}", exc_info=False)

    # 2. Teknik Sinyalleri Hesapla
    try:
        logger.info(f"[{symbol}] Teknik sinyal hesaplama başlatılıyor...")
        technical_signals = await signal_engine.calculate_all_signals(df, symbol=symbol, interval=interval)
        logger.info(f"[{symbol}] Teknik sinyal hesaplama tamamlandı - {len(technical_signals) if technical_signals else 0} tür")
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
        if st_dir_val is not None and not (isinstance(st_dir_val, float) and st_dir_val != st_dir_val):
            st_direction = float(st_dir_val)  # -1=bullish, 1=bearish

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
                    st_confirmed = (sig_type == "Long" and st_bullish) or \
                                   (sig_type == "Short" and not st_bullish)
                    if not st_confirmed:
                        logger.info(
                            f"[{symbol}] {signal_name} {sig_type} ST onaysız "
                            f"(ST {'bullish' if st_bullish else 'bearish'}) — yine de kaydediliyor"
                        )

                # 3. VPMV Hesapla
                vpms_score: Optional[float] = None
                try:
                    vol_s, mom_s, vlt_s, prc_s = _compute_vpmv_scores(df, sig_type)
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
                    _hl  = (_pre["high"] - _pre["low"]).clip(lower=1e-8)
                    _bv  = (_pre["volume"] * (_pre["close"] - _pre["low"])  / _hl).sum()
                    _sv  = (_pre["volume"] * (_pre["high"]  - _pre["close"]) / _hl).sum()
                    _tot = _bv + _sv
                    _buy_pct = _bv / _tot * 100 if _tot > 0 else 50.0
                    logger.info(
                        "PREVOL | %s | %s | %s | buy_pct=%.1f",
                        symbol, sig_type, interval, _buy_pct,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

                # CVD slope (normalize, -1..+1)
                _cvd_slope: Optional[float] = None
                try:
                    if "buy_volume" in df.columns and df["buy_volume"].notna().any():
                        _bv = df["buy_volume"].fillna(
                            df["volume"] * (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(lower=1e-8)
                        )
                    else:
                        _cvd_hl = (df["high"] - df["low"]).clip(lower=1e-8)
                        _bv = df["volume"] * (df["close"] - df["low"]) / _cvd_hl
                    _cvd      = (2 * _bv - df["volume"]).cumsum()
                    _avg_vol  = df["volume"].rolling(10).mean().clip(lower=1e-8)
                    _cvd_slope = round(float((_cvd.diff().rolling(10).mean() / _avg_vol).iloc[-1]), 4)
                    logger.info(
                        "CVD | %s | %s | %s | slope=%.4f",
                        symbol, sig_type, interval, _cvd_slope,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

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

                # 6. Z-score hesapla (EMA200 ayrışması)
                z_score_entry = None
                try:
                    if len(df) >= 210 and "close" in df.columns:
                        closes = df["close"].astype(float)
                        ema200 = closes.ewm(span=200, adjust=False).mean()
                        std200 = closes.rolling(200).std()
                        z_score_entry = round(
                            float((closes.iloc[-1] - ema200.iloc[-1]) / (std200.iloc[-1] + 1e-12)), 3
                        )
                except Exception as exc:
                    logger.debug("z_score_entry hesaplanamadı [%s]: %s", symbol, exc)

                if z_score_entry is not None:
                    _is_long = sig_type == "Long"
                    _z_min = Config.VPM.get("LONG_Z_MIN" if _is_long else "SHORT_Z_MIN")
                    _z_max = Config.VPM.get("LONG_Z_MAX" if _is_long else "SHORT_Z_MAX")
                    if (_z_min is not None and z_score_entry < _z_min) or \
                       (_z_max is not None and z_score_entry > _z_max):
                        logger.info(
                            "[%s] Z-score=%.3f filtre dışı (%s, min=%s max=%s) — sinyal atlandı",
                            symbol, z_score_entry, sig_type, _z_min, _z_max,
                        )
                        continue

                # 6.3. Rejim tespiti
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

                # 6.5. BTC trend (confluence filtresi için önceden hesapla)
                btc_z: Optional[float] = None
                btc_trend_str: Optional[str] = None
                try:
                    btc_df = ref_df if not ref_df.empty else None
                    if btc_df is not None and len(btc_df) >= 210:
                        btc_closes = btc_df["close"].astype(float)
                        btc_ema = btc_closes.ewm(span=200, adjust=False).mean()
                        btc_std = btc_closes.rolling(200).std()
                        btc_z = round(float(
                            (btc_closes.iloc[-1] - btc_ema.iloc[-1]) / (btc_std.iloc[-1] + 1e-12)
                        ), 3)
                        btc_trend_str = "bullish" if btc_z > 0.5 else "bearish" if btc_z < -0.5 else "neutral"
                except Exception as exc:
                    logger.debug("BTC trend hesaplanamadı: %s", exc)

                indicators_name = signal_data.get("indicators", "")

                # HTF HA hizalanması (sadece HA_Cross sinyalleri için)
                htf_bull_count = 0
                if indicators_name == "HA_Cross":
                    htf_bull_count = await _count_htf_ha_bullish(symbol, sig_type, symbol_buffers)

                is_confluence = (
                    indicators_name == "HA_Cross" and
                    sig_type == "Long" and
                    interval == "15m" and
                    htf_bull_count >= Config.HA_HTF_MIN_COUNT
                )

                if is_confluence:
                    logger.info(
                        "[%s] ★ KONFLUANS SİNYALİ %s %s | HTF=%d/%d",
                        symbol, interval, sig_type, htf_bull_count, len(_HTF_CONFIRM_TFS),
                    )

                # 7. Sinyali işle
                current_price = float(df["close"].iloc[-1]) if len(df) >= 1 else None
                enriched_signal = {
                    "symbol":         symbol,
                    "interval":       interval,
                    "indicators":     signal_data.get("indicators"),
                    "signal_type":    sig_type,
                    "opened_at":      signal_data.get("timestamp"),
                    "open_price":     signal_data.get("price"),
                    "vpms_score":     float(vpms_score) if vpms_score is not None else None,
                    "mtf_score":      mtf_score,
                    "st_confirmed":   st_confirmed,
                    "rsi":            signal_data.get("rsi"),
                    "strength":       signal_data.get("strength"),
                    "atr":            signal_data.get("atr"),
                    "alpha":          alpha,
                    "beta":           beta,
                    "sharpe_ratio":       latest_metrics.get("sharpe_ratio"),
                    "sortino_ratio":      latest_metrics.get("sortino_ratio"),
                    "calmar_ratio":       latest_metrics.get("calmar_ratio"),
                    "information_ratio":  latest_metrics.get("information_ratio"),
                    "oi_data":        oi_data,
                    "z_score_entry":  z_score_entry,
                    "is_confluence":  is_confluence,
                    "htf_bull_count": htf_bull_count,
                }

                _vpmv_pre_avg = _vpmv_slope = _vpmv_ratio = None
                try:
                    _pre_avg, _slope, _vpmv_sig = compute_pre(df, sig_type)
                    logger.info(
                        "[%s] VPMV pre raw: df_len=%d pre_avg=%s vpmv_sig=%s",
                        symbol, len(df), _pre_avg, _vpmv_sig,
                    )
                    if _pre_avg is not None and _vpmv_sig is not None:
                        _vpmv_pre_avg = round(_pre_avg, 2)
                        _vpmv_slope   = round(_slope, 2)
                        _vpmv_ratio   = round(_vpmv_sig / _pre_avg, 4) if _pre_avg > 0 else None
                except Exception as _vpe:
                    logger.warning("[%s] VPMV pre hesaplama hatası: %s", symbol, _vpe)

                enriched_signal["vpmv_pre_avg"] = _vpmv_pre_avg
                enriched_signal["vpmv_slope"]   = _vpmv_slope
                enriched_signal["vpmv_ratio"]   = _vpmv_ratio

                min_ratio = float(Config.VPM.get("MIN_RATIO", 1.3))
                if _vpmv_ratio is not None and _vpmv_ratio < min_ratio:
                    logger.info(
                        "[%s] VPMV ratio=%.3f < %.2f — sinyal atlandı (%s)",
                        symbol, _vpmv_ratio, min_ratio, signal_name,
                    )
                    continue
                enriched_signal["cvd_slope"]    = _cvd_slope

                _deviso = _compute_devisso_score(df)
                enriched_signal["devisso_score"] = _deviso
                logger.info(
                    "DEVISSO | %s | %s | %s | score=%s",
                    symbol, sig_type, interval, _deviso,
                )

                _vp_buy, _vp_sell = _compute_vp_score(df)
                _vp_score = round(_vp_buy - _vp_sell, 2)
                enriched_signal["vp_buy_avg"]  = _vp_buy
                enriched_signal["vp_sell_avg"] = _vp_sell
                enriched_signal["vp_score"]    = _vp_score
                logger.info(
                    "VP | %s | %s | %s | buy=%.1f sell=%.1f score=%.1f",
                    symbol, sig_type, interval, _vp_buy, _vp_sell, _vp_score,
                )

                _pd_zone, _mkt_structure = _compute_smc(df, sig_type)
                enriched_signal["pd_zone"]          = _pd_zone
                enriched_signal["market_structure"] = _mkt_structure
                logger.info(
                    "SMC | %s | %s | %s | pd=%.1f struct=%s",
                    symbol, sig_type, interval, _pd_zone or -1, _mkt_structure,
                )

                _fvg_tfs = await _compute_fvg(symbol, sig_type, float(enriched_signal.get("open_price", current_price or 0)))
                enriched_signal["fvg_tfs"] = _fvg_tfs
                logger.info("FVG | %s | %s | tfs=%s", symbol, sig_type, _fvg_tfs)

                _cdl = _compute_candle_pattern(df)
                enriched_signal["candle_pattern"] = _cdl
                logger.info("CDL | %s | %s | pattern=%s", symbol, sig_type, _cdl)

                logger.info(f"[{symbol}] Sinyal işleniyor: {signal_name} - {sig_type}")
                signal_id = await signal_lifecycle_manager.process(enriched_signal, current_price=current_price)
                logger.info(f"[{symbol}] Sinyal işlendi: ID {signal_id}")
                if signal_id:
                    risk_manager.register(symbol)

                if signal_id and current_price:
                    _pt_flag = await _get_pt_flag()
                    if _pt_flag != "0":
                        funding: Optional[float] = None
                        try:
                            import json as _json
                            ticker_raw = await RedisClient.get_client().get(f"ticker:{symbol}")
                            if ticker_raw:
                                ticker_d = _json.loads(ticker_raw)
                                funding = ticker_d.get("funding_rate")
                        except Exception as exc:
                            logger.debug("funding_rate okunamadı [%s]: %s", symbol, exc)

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

            except Exception as e:
                logger.error(f"[{symbol}] Sinyal kayıt hatası: {e}", exc_info=True)
