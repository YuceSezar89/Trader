"""
VPMV (Volume-Price-Momentum-Volatility) hesap yardımcıları.

Ağırlıklar: V=%35, M=%35, Vlt=%20, P=%10
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from config import Config
from indicators.core import calculate_atr, calculate_rsi
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)

PRE_BARS  = 5
POST_BARS = 4


def directional_volume(df: pd.DataFrame, side: float) -> pd.Series:
    """Long için buy_volume, Short için sell_volume; yoksa toplam hacim."""
    has_dir = (
        "buy_volume" in df.columns
        and "sell_volume" in df.columns
        and df["buy_volume"].notna().any()
    )
    if has_dir:
        return df["buy_volume"] if side > 0 else df["sell_volume"]
    return df["volume"]


def proxy_volume(df: pd.DataFrame, side: float) -> pd.Series:
    """Pine vekili: mumun kapanış konumundan alıcı/satıcı hacmi tahmini."""
    hl = (df["high"] - df["low"]).clip(lower=1e-8)
    if side > 0:
        return df["volume"] * (df["close"] - df["low"]) / hl
    return df["volume"] * (df["high"] - df["close"]) / hl


def _volume_by_mode(df: pd.DataFrame, side: float, volume_mode: str) -> pd.Series:
    if volume_mode == "proxy":
        return proxy_volume(df, side)
    if volume_mode == "total":
        return df["volume"]
    return directional_volume(df, side)


def compute_components(
    df: pd.DataFrame, signal_type: str, volume_mode: str = "real"
) -> Tuple[float, float, float, float]:
    """df'nin SON barı için 4 bileşeni ayrı ayrı döner (hepsi 0-100):
    (vol_score, momentum_score, vlt_score, price_score) — compute_series()'in
    ağırlıklı toplamının kırılımı. Anlık/canlı VPMV bileşen gösterimi için
    (ör. Aktif Sinyaller panelinde bir sinyale tıklandığında) kullanılır."""
    side = 1.0 if signal_type == "Long" else -1.0
    vol = normalize_volume_0_100(_volume_by_mode(df, side, volume_mode))
    rsi = calculate_rsi(df, period=14)
    mom = normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)
    atr = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt = normalize_volatility_0_100(atr)
    prc = normalize_price_0_100(df["close"].pct_change().fillna(0.0) * 100.0 * side)
    return float(vol.iloc[-1]), float(mom.iloc[-1]), float(vlt.iloc[-1]), float(prc.iloc[-1])


def compute_series(df: pd.DataFrame, signal_type: str, volume_mode: str = "real") -> pd.Series:
    """Tüm df için bar bazlı VPMV serisi döner (0-100).
    volume_mode: 'real' (taker) | 'proxy' (Pine vekili) | 'total' (yönsüz)."""
    side = 1.0 if signal_type == "Long" else -1.0

    vol = normalize_volume_0_100(_volume_by_mode(df, side, volume_mode))
    rsi = calculate_rsi(df, period=14)
    mom = normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)
    atr = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt = normalize_volatility_0_100(atr)
    prc = normalize_price_0_100(df["close"].pct_change().fillna(0.0) * 100.0 * side)

    return (0.35 * vol + 0.35 * mom + 0.20 * vlt + 0.10 * prc).clip(0, 100)


def compute_pre(
    df: pd.DataFrame,
    signal_type: str,
    pre_bars: int = PRE_BARS,
    volume_mode: str = "real",
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Sinyal barı df'nin son satırı kabul edilir.
    Döner: (pre_avg, slope, vpmv_signal)
    Yeterli bar yoksa (None, None, None).
    """
    if len(df) < pre_bars + 1:
        return None, None, None

    scores = compute_series(df, signal_type, volume_mode)
    vpmv_signal = float(scores.iloc[-1])
    pre_slice   = scores.iloc[-(pre_bars + 1):-1]
    pre_avg     = float(pre_slice.mean())
    slope       = float(pre_slice.iloc[-1] - pre_slice.iloc[0])
    return pre_avg, slope, vpmv_signal


def compute_post(
    df: pd.DataFrame,
    signal_type: str,
    signal_bar_idx: int,
    post_bars: int = POST_BARS,
) -> Tuple[Optional[float], Optional[float]]:
    """
    signal_bar_idx: df içinde sinyal barının pozisyonu (iloc index).
    Döner: (post_avg, post_delta)
    Yeterli bar yoksa (None, None).
    """
    available = len(df) - signal_bar_idx - 1
    if available < post_bars:
        return None, None

    scores      = compute_series(df, signal_type)
    vpmv_signal = float(scores.iloc[signal_bar_idx])
    post_slice  = scores.iloc[signal_bar_idx + 1: signal_bar_idx + 1 + post_bars]
    post_avg    = float(post_slice.mean())
    post_delta  = post_avg - vpmv_signal
    return post_avg, post_delta
