"""
Sinyal rejim analizi.
Her kapanmış sinyal için açılış anındaki ATR ratio + ADX hesaplar,
4 rejim hücresinde kazanma oranı ve PnL raporlar.

Kullanım:
    python scripts/analyze_regime.py [--interval 5m]
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

DB_DSN = (f"dbname={Config.DB_NAME} user={Config.DB_USER} "
          f"host={Config.DB_HOST} port={Config.DB_PORT}")

ATR_PERIOD   = 14
ATR_MA_PERIOD = 50
ADX_PERIOD   = 14
MIN_BARS     = ATR_PERIOD + ATR_MA_PERIOD + 10
ADX_THRESHOLD = 25.0
VOL_THRESHOLD = 1.0   # ATR ratio: >1.0 yüksek, ≤1.0 düşük


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)

    up   = h - h.shift(1)
    down = l.shift(1) - l

    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    s = pd.Series

    def wilder(arr):
        out = pd.Series(arr, dtype=float).ewm(alpha=1 / period, adjust=False).mean()
        return out

    atr14    = wilder(tr)
    plus_di  = 100 * wilder(s(plus_dm,  index=df.index)) / atr14
    minus_di = 100 * wilder(s(minus_dm, index=df.index)) / atr14

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _regime(atr_ratio: float, adx_val: float) -> str:
    trend  = "Trend+" if adx_val >= ADX_THRESHOLD else "Trend-"
    vol    = "Vol+"   if atr_ratio > VOL_THRESHOLD else "Vol-"
    return f"{trend} / {vol}"


def fetch_signals(conn, interval_filter=None):
    q = """
        SELECT id, symbol, interval, signal_type, opened_at, realized_pnl
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL
    """
    params = []
    if interval_filter:
        q += " AND interval = %s"
        params.append(interval_filter)
    q += " ORDER BY symbol, interval, opened_at"
    cur = conn.cursor()
    cur.execute(q, params)
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "realized_pnl"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


_CAGG_MAP = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h"}

def fetch_price(conn, symbol, interval):
    cur = conn.cursor()
    cagg = _CAGG_MAP.get(interval)
    if cagg:
        cur.execute(f"""
            SELECT bucket, open, high, low, close, volume
            FROM {cagg}
            WHERE symbol = %s
            ORDER BY bucket ASC
        """, (symbol,))
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
    else:
        cur.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = %s AND interval = %s
            ORDER BY timestamp ASC
        """, (symbol, interval))
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


def analyze(interval_filter=None):
    conn = psycopg2.connect(DB_DSN)
    signals = fetch_signals(conn, interval_filter)
    print(f"Toplam sinyal: {len(signals)}")

    regime_buckets = defaultdict(list)
    skipped = 0

    groups = signals.groupby(["symbol", "interval"])
    total_groups = len(groups)

    for g_idx, ((sym, ivl), grp) in enumerate(groups):
        if g_idx % 20 == 0:
            print(f"  İşleniyor {g_idx}/{total_groups}...", end="\r")

        price_df = fetch_price(conn, sym, ivl)
        if price_df is None or len(price_df) < MIN_BARS:
            skipped += len(grp)
            continue

        atr_series   = _atr(price_df, ATR_PERIOD)
        atr_ma       = atr_series.rolling(ATR_MA_PERIOD).mean()
        atr_ratio_s  = atr_series / atr_ma
        adx_series   = _adx(price_df, ADX_PERIOD)

        for _, sig in grp.iterrows():
            ts = pd.Timestamp(sig["opened_at"])
            idx = price_df.index.searchsorted(ts, side="right") - 1
            if idx < MIN_BARS:
                skipped += 1
                continue

            atr_ratio = float(atr_ratio_s.iloc[idx])
            adx_val   = float(adx_series.iloc[idx])

            if np.isnan(atr_ratio) or np.isnan(adx_val):
                skipped += 1
                continue

            regime = _regime(atr_ratio, adx_val)
            regime_buckets[regime].append({
                "pnl":         sig["realized_pnl"],
                "signal_type": sig["signal_type"],
                "interval":    ivl,
            })

    conn.close()
    print(f"\nAtlanan: {skipped} (yetersiz veri)")
    _report(regime_buckets)


def _report(buckets):
    ORDER = [
        "Trend+ / Vol+",
        "Trend+ / Vol-",
        "Trend- / Vol+",
        "Trend- / Vol-",
    ]

    print(f"\n{'─'*70}")
    print(f"{'Rejim':<22} {'Sinyal':>7} {'Kazanma%':>9} {'Ort PnL':>9} {'Med PnL':>9} {'Long%':>7}")
    print(f"{'─'*70}")

    for regime in ORDER:
        rows = buckets.get(regime, [])
        if not rows:
            print(f"{regime:<22} {'—':>7}")
            continue
        pnls   = [r["pnl"] for r in rows]
        wins   = sum(1 for p in pnls if p > 0)
        longs  = sum(1 for r in rows if r["signal_type"] == "Long")
        print(
            f"{regime:<22} {len(rows):>7} {wins/len(rows)*100:>8.1f}%"
            f" {np.mean(pnls):>+8.2f}% {np.median(pnls):>+8.2f}%"
            f" {longs/len(rows)*100:>6.0f}%"
        )

    print(f"{'─'*70}")

    # Interval dağılımı
    print("\nInterval dağılımı:")
    iv_count = defaultdict(int)
    for rows in buckets.values():
        for r in rows:
            iv_count[r["interval"]] += 1
    for iv, cnt in sorted(iv_count.items(), key=lambda x: -x[1]):
        print(f"  {iv:>4}: {cnt}")

    # En iyi / en kötü rejim
    scored = [(r, np.mean([x["pnl"] for x in rows])) for r, rows in buckets.items() if rows]
    if scored:
        best  = max(scored, key=lambda x: x[1])
        worst = min(scored, key=lambda x: x[1])
        print(f"\nEn iyi rejim : {best[0]}  ({best[1]:+.2f}% ort PnL)")
        print(f"En kötü rejim: {worst[0]}  ({worst[1]:+.2f}% ort PnL)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default=None, help="5m, 15m, 1h vb.")
    args = parser.parse_args()
    analyze(args.interval)
