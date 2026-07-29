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

from typing import Optional

import numpy as np
import pandas as pd

from indicators.core import truncate_after_gap


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
