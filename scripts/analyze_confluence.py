"""
Konfluans analizi: Z-score (EMA200 ayrışması) + VPMV yüksek olan sinyallerin
PnL'ini tüm sinyallerle karşılaştırır.

Hipotez: Her iki metrikte de üst %X'de olan sinyaller daha iyi performans gösterir.

Kullanım:
    python scripts/analyze_confluence.py [--interval 5m] [--threshold 20]
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
EMA_PERIOD = 200
MIN_BARS   = EMA_PERIOD + 10

_CAGG_MAP = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h"}


def fetch_signals(conn, interval_filter=None):
    q = """
        SELECT id, symbol, interval, signal_type, opened_at,
               realized_pnl, vpms_score
        FROM signals
        WHERE status = 'closed'
          AND realized_pnl IS NOT NULL
          AND vpms_score   IS NOT NULL
    """
    params = []
    if interval_filter:
        q += " AND interval = %s"
        params.append(interval_filter)
    q += " ORDER BY symbol, interval, opened_at"
    cur = conn.cursor()
    cur.execute(q, params)
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "pnl", "vpmv"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_price(conn, symbol, interval):
    cur = conn.cursor()
    cagg = _CAGG_MAP.get(interval)
    if cagg:
        cur.execute(
            f"SELECT bucket, close FROM {cagg} WHERE symbol = %s ORDER BY bucket ASC",
            (symbol,)
        )
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "close"])
    else:
        cur.execute(
            "SELECT timestamp, close FROM price_data WHERE symbol = %s AND interval = %s ORDER BY timestamp ASC",
            (symbol, interval)
        )
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


def _zscore_series(close: pd.Series) -> pd.Series:
    ema  = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    std  = close.rolling(EMA_PERIOD).std()
    return (close - ema) / (std + 1e-12)


def compute_zscores(conn, signals: pd.DataFrame) -> pd.DataFrame:
    results = []
    skipped = 0
    groups  = signals.groupby(["symbol", "interval"])
    total   = len(groups)

    for g_idx, ((sym, ivl), grp) in enumerate(groups):
        if g_idx % 30 == 0:
            print(f"  Z-score hesaplanıyor {g_idx}/{total}...", end="\r")

        price_df = fetch_price(conn, sym, ivl)
        if price_df is None or len(price_df) < MIN_BARS:
            skipped += len(grp)
            continue

        z_series = _zscore_series(price_df["close"])

        for _, sig in grp.iterrows():
            ts  = pd.Timestamp(sig["opened_at"])
            idx = price_df.index.searchsorted(ts, side="right") - 1
            if idx < MIN_BARS:
                skipped += 1
                continue
            z = float(z_series.iloc[idx])
            if np.isnan(z):
                skipped += 1
                continue
            results.append({
                "id":          sig["id"],
                "symbol":      sym,
                "interval":    ivl,
                "signal_type": sig["signal_type"],
                "pnl":         sig["pnl"],
                "vpmv":        sig["vpmv"],
                "zscore":      z,
            })

    print(f"\n  Atlanan: {skipped} (yetersiz veri)")
    return pd.DataFrame(results)


def report(df: pd.DataFrame, threshold: float) -> None:
    n = len(df)
    vpmv_cut  = np.percentile(df["vpmv"],          100 - threshold)
    z_cut_pos = np.percentile(df["zscore"].abs(),  100 - threshold)

    df["vpmv_top"]  = df["vpmv"] >= vpmv_cut
    df["z_top"]     = df["zscore"].abs() >= z_cut_pos
    df["confluence"] = df["vpmv_top"] & df["z_top"]

    def _stats(subset, label):
        if len(subset) == 0:
            print(f"{label:<35} — veri yok")
            return
        wins = (subset["pnl"] > 0).sum()
        print(
            f"{label:<35} {len(subset):>6}  "
            f"{wins/len(subset)*100:>7.1f}%  "
            f"{subset['pnl'].mean():>+8.3f}%  "
            f"{subset['pnl'].median():>+8.3f}%"
        )

    print(f"\n{'─'*75}")
    print(f"{'Grup':<35} {'Sinyal':>6}  {'Kazan%':>7}  {'OrtPnL':>8}  {'MedPnL':>8}")
    print(f"{'─'*75}")

    _stats(df, "Tüm sinyaller")
    _stats(df[df["vpmv_top"]],   f"VPMV üst %{threshold:.0f}")
    _stats(df[df["z_top"]],      f"|Z-score| üst %{threshold:.0f}")
    _stats(df[df["confluence"]], f"Konfluans (VPMV + Z, üst %{threshold:.0f})")

    print(f"{'─'*75}")

    # Yön bazlı konfluans
    cf = df[df["confluence"]]
    if len(cf) > 0:
        print(f"\nKonfluans — sinyal tipine göre:")
        for st in ["Long", "Short"]:
            sub = cf[cf["signal_type"] == st]
            if len(sub) == 0:
                continue
            wins = (sub["pnl"] > 0).sum()
            print(f"  {st:<8} {len(sub):>4} sinyal  "
                  f"kazanma={wins/len(sub)*100:.1f}%  "
                  f"ort={sub['pnl'].mean():+.3f}%")

    # Interval bazlı
    print(f"\nKonfluans — interval bazlı:")
    for ivl, sub in cf.groupby("interval"):
        wins = (sub["pnl"] > 0).sum()
        print(f"  {ivl:<5} {len(sub):>4} sinyal  "
              f"kazanma={wins/len(sub)*100:.1f}%  "
              f"ort={sub['pnl'].mean():+.3f}%")

    # Z-score yönü: pozitif mi negatif mi daha iyi?
    print(f"\nKonfluans — Z-score yönüne göre:")
    cf_pos = cf[cf["zscore"] > 0]
    cf_neg = cf[cf["zscore"] < 0]
    for label, sub in [("Z > 0 (pozitif ayrışma)", cf_pos), ("Z < 0 (negatif ayrışma)", cf_neg)]:
        if len(sub) == 0:
            continue
        wins = (sub["pnl"] > 0).sum()
        print(f"  {label:<30} {len(sub):>4}  "
              f"kazanma={wins/len(sub)*100:.1f}%  "
              f"ort={sub['pnl'].mean():+.3f}%")

    # Eşik hassasiyet analizi
    print(f"\nEşik hassasiyeti (konfluans):")
    print(f"  {'Eşik':<8} {'Sinyal':>6}  {'Kazan%':>7}  {'OrtPnL':>8}")
    for t in [10, 20, 30, 40]:
        vc = np.percentile(df["vpmv"], 100 - t)
        zc = np.percentile(df["zscore"].abs(), 100 - t)
        sub = df[(df["vpmv"] >= vc) & (df["zscore"].abs() >= zc)]
        if len(sub) == 0:
            continue
        wins = (sub["pnl"] > 0).sum()
        print(f"  üst %{t:<3}  {len(sub):>6}  "
              f"{wins/len(sub)*100:>7.1f}%  "
              f"{sub['pnl'].mean():>+8.3f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",  default=None)
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Üst yüzdelik eşik (varsayılan: 20)")
    args = parser.parse_args()

    conn    = psycopg2.connect(DB_DSN)
    signals = fetch_signals(conn, args.interval)
    print(f"Toplam sinyal: {len(signals)}")

    df = compute_zscores(conn, signals)
    conn.close()

    print(f"Z-score hesaplanan: {len(df)}")
    if len(df) < 20:
        print("Yeterli veri yok.")
        return

    report(df, args.threshold)


if __name__ == "__main__":
    main()
