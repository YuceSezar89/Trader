import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to sys.path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from signals.c20mx_core import compute_features, detect_signals, last_bar_summary


def make_synthetic_ohlc(n=300, start_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0008, scale=0.01, size=n)  # slight drift
    prices = start_price * np.exp(np.cumsum(rets))

    # build OHLC with small ranges around close
    close = prices
    high = close * (1 + np.abs(rng.normal(0.002, 0.002, size=n)))
    low = close * (1 - np.abs(rng.normal(0.002, 0.002, size=n)))
    open_ = (high + low) / 2
    volume = rng.integers(1_000, 10_000, size=n)

    # timestamps (ms)
    base = datetime.utcnow() - timedelta(minutes=n)
    ts = [(base + timedelta(minutes=i)).timestamp() * 1000 for i in range(n)]

    df = pd.DataFrame({
        'timestamp': ts,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df


def main():
    df = make_synthetic_ohlc()
    feat = compute_features(df)
    codes = detect_signals(feat, i=-1, interval="15m")
    summary = last_bar_summary(feat, interval="15m")

    print("Signals:", codes)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
