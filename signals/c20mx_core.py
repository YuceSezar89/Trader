"""
Core logic to compute C (RSI-change), L (low-RSI change), and M (MACD-derived) signals
without I/O, network calls, or side effects.

This module isolates and adapts the stable logic found in `inceleme/c20m5.py.txt`.

Expected input DataFrame columns:
- 'open', 'high', 'low', 'close' (float-like)
- Optional: 'timestamp' (ms) for reference

Provided functions:
- compute_features(df): add/return necessary feature series used by signal detection
- detect_signals(df, i=-1, interval=None): return list of signal codes at bar index i

Notes:
- All calculations are pandas/numpy based and self-contained
- No prints/logging; pure functions to integrate into scanners/panel
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any, TypedDict
import numpy as np
import pandas as pd


# -----------------------------
# Feature Calculations
# -----------------------------

def _calculate_rsi_wilders(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on a single series.
    Safe for zeros/NaNs; returns float Series in [0, 100] (with NaNs at head).
    """
    s = series.astype(float)
    delta = s.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _calculate_diff_percent(df: pd.DataFrame, length: int = 200) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Replicates the pullback-level logic from the reference script and returns
    three series:
    - diff_percent: percent difference from the active long pullback level
    - long_pullback_level: last known long-side pullback level (low at long signals)
    - short_pullback_level: last known short-side pullback level (high at short signals)
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    ema_value = _calculate_ema(close, length)

    # Crossover / Crossunder conditions on close vs EMA
    long_signal = (close > ema_value) & (close.shift(1) <= ema_value.shift(1))
    short_signal = (close < ema_value) & (close.shift(1) >= ema_value.shift(1))

    long_pull = np.full_like(close, np.nan, dtype=float)
    short_pull = np.full_like(close, np.nan, dtype=float)
    is_after_short = np.zeros_like(close, dtype=bool)

    for i in range(1, len(close)):
        long_pull[i] = long_pull[i - 1]
        short_pull[i] = short_pull[i - 1]
        is_after_short[i] = is_after_short[i - 1]

        if short_signal.iloc[i]:
            short_pull[i] = high.iloc[i]
            long_pull[i] = short_pull[i]
            is_after_short[i] = True

        if long_signal.iloc[i] and np.isnan(long_pull[i]):
            long_pull[i] = low.iloc[i]

        if long_signal.iloc[i] and is_after_short[i]:
            long_pull[i] = low.iloc[i]
            is_after_short[i] = False

    diff_percent = pd.Series(np.nan, index=close.index)
    for i in range(len(close)):
        if not np.isnan(long_pull[i]):
            diff_percent.iloc[i] = ((close.iloc[i] - long_pull[i]) / long_pull[i]) * 100.0

    long_pull_series = pd.Series(long_pull, index=df.index, dtype=float)
    short_pull_series = pd.Series(short_pull, index=df.index, dtype=float)
    return diff_percent, long_pull_series, short_pull_series


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all required features into a copy of df.

    Adds columns:
    - 'ema200_close'
    - 'logHigh', 'logLow', 'logClose'
    - 'rsihigh', 'rsilow', 'rsiclose'
    - 'rsiChangeclose', 'rsiChangelow'
    - 'fast_ma', 'slow_ma', 'rsi_temp', 'macd', 'ema_macd', 'rsi_signal', 'signal', 'top'
    - 'diff_percent'
    """
    if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")

    out = df.copy()
    # Ensure numeric first
    for c in ['open', 'high', 'low', 'close']:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    # EMA200 on close
    out['ema200_close'] = out['close'].ewm(span=200, adjust=False).mean().astype(float)

    # Log transforms (protect zeros)
    out['logHigh'] = np.log(out['high'].replace(0, np.nan))
    out['logLow'] = np.log(out['low'].replace(0, np.nan))
    out['logClose'] = np.log(out['close'].replace(0, np.nan))

    # Wilder RSI on logs and close
    out['rsihigh'] = _calculate_rsi_wilders(out['logHigh'])
    out['rsilow'] = _calculate_rsi_wilders(out['logLow'])
    out['rsiclose'] = _calculate_rsi_wilders(out['logClose'])

    # Changes
    out['rsiChangeclose'] = out['rsiclose'].diff()
    out['rsiChangelow'] = out['rsilow'].diff()

    # MACD-like path from reference
    out['fast_ma'] = out['close'].ewm(span=12, adjust=False).mean()
    out['slow_ma'] = out['close'].ewm(span=26, adjust=False).mean()
    temp = out['fast_ma'] - out['slow_ma']
    out['rsi_temp'] = _calculate_rsi_wilders(temp, period=100)
    out['macd'] = out['rsi_temp'].diff()
    out['ema_macd'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['rsi_signal'] = _calculate_rsi_wilders(out['ema_macd'], period=100)
    out['signal'] = out['rsi_signal'].diff()
    out['top'] = (out['macd'] + out['signal']) / 2.0

    # Diff percent and pullback levels per reference pullback logic
    diff_percent, long_pull_s, short_pull_s = _calculate_diff_percent(out, length=200)
    out['diff_percent'] = diff_percent
    out['long_pullback_level'] = long_pull_s
    out['short_pullback_level'] = short_pull_s

    return out


# -----------------------------
# Signal Detection
# -----------------------------

def _is_crossover(series: pd.Series, threshold: float, i: int) -> bool:
    return bool(series.iloc[i - 1] < threshold and series.iloc[i] >= threshold)


def _is_crossunder(series: pd.Series, threshold: float, i: int) -> bool:
    return bool(series.iloc[i - 1] > threshold and series.iloc[i] <= threshold)


def detect_signals(df: pd.DataFrame, i: int = -1, interval: Optional[str] = None) -> List[str]:
    """Detect C/L/M style signals at bar index i, returning a list of codes.

    Uses thresholds consistent with the reference:
    - C: rsiChangeclose crossovers (±20, ±10)
    - L: rsiChangelow crossovers (±20)
    - M: macd/signal/top crossing ±2..±5
    """
    required = {'rsiChangeclose', 'rsiChangelow', 'macd', 'signal', 'top'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame is missing required features: {sorted(required - set(df.columns))}")

    n = len(df)
    if n < 3 or not (-n <= i < n):
        return []

    sigs: List[str] = []

    # C signals (close RSI-change)
    if _is_crossover(df['rsiChangeclose'], 20, i):
        sigs.append("C20L")
    if _is_crossover(df['rsiChangeclose'], 10, i):
        sigs.append("C10L")
    if _is_crossunder(df['rsiChangeclose'], -20, i):
        sigs.append("C20S")
    if _is_crossunder(df['rsiChangeclose'], -10, i):
        sigs.append("C10S")

    # L signals (low RSI-change based in ref code)
    if _is_crossover(df['rsiChangelow'], -20, i):
        sigs.append("L20S")
    if _is_crossunder(df['rsiChangelow'], 20, i):
        sigs.append("L20L")

    # M signals (MACD-like thresholds)
    if any((_is_crossover(df['macd'], 2, i), _is_crossover(df['signal'], 2, i), _is_crossover(df['top'], 2, i))):
        sigs.append("M2L")
    if any((_is_crossover(df['macd'], 3, i), _is_crossover(df['signal'], 3, i), _is_crossover(df['top'], 3, i))):
        sigs.append("M3L")
    if any((_is_crossover(df['macd'], 4, i), _is_crossover(df['signal'], 4, i), _is_crossover(df['top'], 4, i))):
        sigs.append("M4L")
    if any((_is_crossover(df['macd'], 5, i), _is_crossover(df['signal'], 5, i), _is_crossover(df['top'], 5, i))):
        sigs.append("M5L")

    if any((_is_crossunder(df['macd'], -2, i), _is_crossunder(df['signal'], -2, i), _is_crossunder(df['top'], -2, i))):
        sigs.append("M2S")
    if any((_is_crossunder(df['macd'], -3, i), _is_crossunder(df['signal'], -3, i), _is_crossunder(df['top'], -3, i))):
        sigs.append("M3S")
    if any((_is_crossunder(df['macd'], -4, i), _is_crossunder(df['signal'], -4, i), _is_crossunder(df['top'], -4, i))):
        sigs.append("M4S")
    if any((_is_crossunder(df['macd'], -5, i), _is_crossunder(df['signal'], -5, i), _is_crossunder(df['top'], -5, i))):
        sigs.append("M5S")

    # Append interval tag if provided (e.g., "[15m]")
    if interval:
        sigs.append(f"[{interval}]")

    return sigs


class LastBarSummary(TypedDict):
    price: Optional[float]
    ema200: Optional[float]
    diff_percent: Optional[float]
    signals: List[str]


def last_bar_summary(df: pd.DataFrame, interval: Optional[str] = None) -> LastBarSummary:
    """Convenience helper to extract last-bar summary values used often in UI/logs.
    Returns a dict with price, ema200, diff_percent, and codes list.
    """
    if len(df) == 0:
        empty_codes: List[str] = []
        return {"price": None, "ema200": None, "diff_percent": None, "signals": empty_codes}

    i = -1
    price = float(df['close'].iloc[i]) if 'close' in df else None
    ema200 = float(df['ema200_close'].iloc[i]) if 'ema200_close' in df else None
    diff_percent = float(df['diff_percent'].iloc[i]) if 'diff_percent' in df and pd.notna(df['diff_percent'].iloc[i]) else None
    codes = detect_signals(df, i=i, interval=interval)
    return {"price": price, "ema200": ema200, "diff_percent": diff_percent, "signals": codes}
