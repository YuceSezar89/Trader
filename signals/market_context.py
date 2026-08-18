"""
market_context.py — CVD/VP/SMC piyasa bağlamı hesaplamaları.

Sinyal açılışında (signal_processor.py) VE periyodik pozisyon izlemede
(signal_service.py::trade_snapshot_loop, 5dk'da bir açık pozisyonlar için)
AYNI kaynaktan kullanılır — önceden bu hesaplamalar signal_processor.py'nin
içine satır-içi/özel (alt çizgili) fonksiyon olarak gömülüydü, ikinci bir
kullanım yeri (periyodik izleme) ortaya çıkınca kod tekrarı riski oluştu,
buraya taşındı (19 Tem 2026).

VPMV bileşenleri BURADA DEĞİL — zaten utils/vpmv.py::compute_components'te
modüler, oradan doğrudan import edilmeli.
"""

import asyncio
import functools
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from indicators.core import (
    calculate_macd,
    calculate_mfi,
    calculate_obv,
    calculate_rsi,
    truncate_after_gap,
)
from indicators.financial_metrics import calculate_metrics
from utils.logger import get_logger
from utils.redis_client import RedisClient

logger = get_logger("MarketContext")


def compute_cvd_slope(df: pd.DataFrame) -> Optional[float]:
    """CVD (kümülatif hacim deltası) eğimi, normalize edilmiş (~-1..+1)."""
    try:
        df_g = truncate_after_gap(df)
        if "buy_volume" in df_g.columns and df_g["buy_volume"].notna().any():
            bv = df_g["buy_volume"].fillna(
                df_g["volume"]
                * (df_g["close"] - df_g["low"])
                / (df_g["high"] - df_g["low"]).clip(lower=1e-8)
            )
        else:
            hl = (df_g["high"] - df_g["low"]).clip(lower=1e-8)
            bv = df_g["volume"] * (df_g["close"] - df_g["low"]) / hl
        cvd = (2 * bv - df_g["volume"]).cumsum()
        avg_vol = df_g["volume"].rolling(10).mean().clip(lower=1e-8)
        return round(float((cvd.diff().rolling(10).mean() / avg_vol).iloc[-1]), 4)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def compute_vp_score(
    df: pd.DataFrame, lookback: int = 500, use_real_volume: bool = False
) -> tuple[float, float]:
    """%VP Normalized Lines — PineScript birebir çeviri.

    Returns: (buy_positive_avg, sell_negative_avg) — her ikisi 0-100 arası.
    vp_score = buy_positive_avg - sell_negative_avg → pozitif: alıcı baskısı.
    """
    try:
        price_change = df["close"].diff().fillna(0.0)
        cum_positive = price_change.clip(lower=0.0).cumsum()
        cum_negative = (-price_change).clip(lower=0.0).cumsum()
        total_move = (cum_positive + cum_negative).replace(0, np.nan)
        positive_pct = (cum_positive / total_move * 100).fillna(50.0)
        negative_pct = (cum_negative / total_move * 100).fillna(50.0)

        if use_real_volume:
            if "buy_volume" not in df.columns or not df["buy_volume"].notna().any():
                return float("nan"), float("nan")
            bv = df["buy_volume"].fillna(0.0)
            sv = df["sell_volume"].fillna(0.0)
        else:
            hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
            bv = df["volume"] * (df["close"] - df["low"]) / hl_range
            sv = df["volume"] * (df["high"] - df["close"]) / hl_range
        cum_buy = bv.cumsum()
        cum_sell = sv.cumsum()
        total_vol = (cum_buy + cum_sell).replace(0, np.nan)
        buy_pct = (cum_buy / total_vol * 100).fillna(50.0)
        sell_pct = (cum_sell / total_vol * 100).fillna(50.0)

        def _norm(s: pd.Series) -> pd.Series:
            lo = s.rolling(lookback, min_periods=1).min()
            hi = s.rolling(lookback, min_periods=1).max()
            return ((s - lo) / (hi - lo + 1e-10) * 100).fillna(50.0)

        buy_pos_avg = (_norm(buy_pct) + _norm(positive_pct)) / 2
        sell_neg_avg = (_norm(sell_pct) + _norm(negative_pct)) / 2

        return round(float(buy_pos_avg.iloc[-1]), 2), round(float(sell_neg_avg.iloc[-1]), 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return 50.0, 50.0


def compute_smc_market_structure(df: pd.DataFrame, sig_type: str, lookback: int = 50) -> str:
    """SMC Market Structure (BOS/CHoCH) — sadece yön uyumu ('Uyum↑' vb.).

    Premium/Discount zone KASITLI OLARAK burada YOK (19 Tem 2026, kullanıcı
    kararı — işe yaramıyordu, bkz. sohbet). Sadece son yapısal olayın
    (BOS/CHoCH) YÖNÜ, sinyal yönüyle karşılaştırılıyor:
      Uyum↑/Uyum↓  — sinyal mevcut yapıyla aynı yönde (trend devamı)
      Karşı↑/Karşı↓ — sinyal yapıya karşı (dönüş denemesi)
      -             — yapı belirlenemedi
    """
    try:
        if len(df) < lookback + 5:
            return "-"

        from smartmoneyconcepts import smc as _smc_lib  # pylint: disable=import-outside-toplevel

        df_use = df.tail(lookback).copy().reset_index(drop=True)
        df_smc = df_use[["open", "high", "low", "close", "volume"]].copy()
        for col in df_smc.columns:
            df_smc[col] = df_smc[col].astype(float)

        swing_df = _smc_lib.swing_highs_lows(df_smc, swing_length=5)
        bos_df = _smc_lib.bos_choch(df_smc, swing_df, close_break=True)

        structure_dir = 0
        for i in range(len(bos_df) - 1, -1, -1):
            bos_val = bos_df["BOS"].iloc[i]
            choch_val = bos_df["CHOCH"].iloc[i]
            if not np.isnan(bos_val):
                structure_dir = int(bos_val)
                break
            if not np.isnan(choch_val):
                structure_dir = int(choch_val)
                break

        if structure_dir == 1:
            return "Uyum↑" if sig_type == "Long" else "Karşı↓"
        if structure_dir == -1:
            return "Uyum↓" if sig_type == "Short" else "Karşı↑"
        return "-"
    except Exception:  # pylint: disable=broad-exception-caught
        return "-"


# 2 Ağu 2026 (Fable 5 performans denetimi): get_mtf_klines'ın in-process cache'i
# sadece veriyi YAZAN process'te (live_data_manager) dolu — signal_service.py
# ayrı bir process olduğu için BTC referansı her _process_event çağrısında
# gerçek bir Redis GET + Arrow deserialize'a düşüyordu. Bir bar-kapanışı
# burst'ünde (ör. 5m'de 548 sembol) hepsi AYNI BTC verisini istiyor — kısa
# TTL'li (2sn) interval-bazlı bir cache, aynı burst içindeki tekrarları eler.
# 17 Ağu 2026: signal_service.py'den buraya taşındı — trade_snapshot.py da
# (alpha/beta/finansal oranlar için) aynı BTC referansına ihtiyaç duyunca
# signal_service.py'ye dairesel import olmadan erişebilsin diye.
_REF_DF_CACHE_TTL = 2.0
_ref_df_cache: dict[str, tuple[float, pd.DataFrame]] = {}

# calculate_metrics CPU-yoğun (rolling regresyon + çok sayıda normalize_series
# çağrısı) — ayrı bir process pool'da çalıştırılıyor ki event loop bloklanmasın.
# 17 Ağu 2026: signal_service.py'den buraya taşındı — trade_snapshot.py da
# (alpha/beta/finansal oranların periyodik izlemesi için) AYNI havuzu
# kullanmalı, iki ayrı pool açmak gereksiz process/bellek maliyeti olurdu.
_METRICS_POOL_WORKERS = 5
_metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)


def prepare_for_metrics(
    df: pd.DataFrame, ref_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """calculate_metrics'in beklediği DatetimeIndex'e çevirir (open_time ms ->
    index) — signal_processor.py VE trade_snapshot.py AYNI hazırlığı yapıyordu
    (17 Ağu 2026, pylint duplicate-code uyarısı), buraya toplandı."""
    df_prepared = df.copy()
    df_prepared.index = pd.Index(pd.to_datetime(df_prepared["open_time"], unit="ms"))
    ref_df_prepared = ref_df.copy()
    ref_df_prepared.index = pd.Index(pd.to_datetime(ref_df_prepared["open_time"], unit="ms"))
    return df_prepared, ref_df_prepared


async def calculate_metrics_via_pool(
    df_prepared: pd.DataFrame, ref_df_prepared: pd.DataFrame, interval: str
) -> pd.DataFrame:
    """calculate_metrics'i ProcessPoolExecutor'da çalıştırır. Pool çökerse (BrokenProcessPool)
    pool yeniden oluşturulur VE aynı event senkron olarak (yavaş ama veri kaybetmeden)
    hesaplanır — process_and_enrich_signals gerçek DB yazımı/paper trade tetiklemesi
    içerdiği için bu event'in sessizce kaybolması kabul edilebilir değil (dry_run/gölge
    dönemindeki eski davranıştan bilinçli fark)."""
    global _metrics_pool  # pylint: disable=global-statement
    loop = asyncio.get_running_loop()
    fn = functools.partial(calculate_metrics, df_prepared, ref_df_prepared, interval=interval)
    try:
        return await loop.run_in_executor(_metrics_pool, fn)
    except BrokenProcessPool as e:
        logger.warning(
            "ENDİŞE: Metrics process pool çöktü — pool yeniden oluşturuluyor, "
            "bu event senkron fallback ile hesaplanıyor (yavaş yol): %s",
            e,
        )
        _metrics_pool.shutdown(wait=False)
        _metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)
        return calculate_metrics(df_prepared, ref_df_prepared, interval=interval)


async def get_ref_df_cached(interval: str) -> "pd.DataFrame | None":
    cached = _ref_df_cache.get(interval)
    now = time.monotonic()
    if cached is not None and (now - cached[0]) < _REF_DF_CACHE_TTL:
        return cached[1]
    ref_df = await RedisClient.get_fresh_klines(Config.MARKET_REFERENCE_SYMBOL, interval)
    if ref_df is not None:
        _ref_df_cache[interval] = (now, ref_df)
    return ref_df


def classify_candle(last: pd.Series) -> tuple[str, "float | None", "float | None"]:
    """Son mumun gövde/üst fitil/alt fitil oranına göre baskın kısmını VE
    sayısal gövde/fitil yüzdelerini döner: (kategori, body_pct, wick_pct).
    wick_pct = üst+alt fitil toplamı (=100-body_pct) — Hoca'nın Sinyal Mumu
    Ratioları'ndaki Body Ratio/Wick Length metriklerinin sayısal karşılığı
    (13 Ağu 2026 — önceden sadece kategori tutulup sayı atılıyordu; 17 Ağu
    2026: signal_processor.py'den buraya taşındı, trade_snapshot.py da
    kullanıyor)."""
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


def compute_signal_extras(df: pd.DataFrame) -> Optional[dict]:
    """Sinyal Mumu Ratioları'nın (Hoca Telegram külliyatı, S29 — orijinal
    Pine dosyası) kalan metrikleri: RSI(14)/MFI(14)/RSI-of-MACD(12,26,9,14)
    bar-to-bar değişimi + OBV seviyesi, df'nin SON barında. VPMV'nin 4
    bileşeninden (vol/mom/vlt/prc) BAĞIMSIZ, kendi ham hesabı — isim
    çakışmasın diye "signal_" önekiyle (13 Ağu 2026; 17 Ağu 2026:
    signal_processor.py'den buraya taşındı — bar-to-bar değişim tanımı
    "sinyal anı" kavramına bağımlı değil, periyodik izlemede de aynı
    anlamı taşıyor, trade_snapshot.py da kullanıyor)."""
    if len(df) < 35:  # MACD(26) + üstüne RSI(14) ısınma payı
        return None

    out: dict = {"rsi_change": None, "mfi_change": None, "macd_change": None, "obv": None}
    try:
        rsi = calculate_rsi(df, period=14)
        out["rsi_change"] = float(rsi.diff().iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
        mfi = calculate_mfi(df, period=14)
        out["mfi_change"] = float(mfi.diff().iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
        macd_line, _, _ = calculate_macd(df, fast=12, slow=26, signal=9)
        macd_rsi = calculate_rsi(pd.DataFrame({"macd": macd_line}), period=14, price_col="macd")
        out["macd_change"] = float(macd_rsi.diff().iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
        obv = calculate_obv(df)
        out["obv"] = float(obv.iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return out
