import os
import sys
import sqlite3
import argparse
import pandas as pd
from typing import List

# Ensure project root in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from signals.c20mx_core import compute_features, detect_signals


def load_ethusdt_from_db(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        q = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = 'ETHUSDT'
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    # Normalize types
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # If timestamp is numeric ms string, keep as is; if iso, try to convert to ms
    # We will also keep an ISO column for easier TradingView comparison
    try:
        # If already numeric-like ms
        ts_ms = pd.to_numeric(df['timestamp'], errors='coerce')
        if ts_ms.notna().all():
            df['ts_ms'] = ts_ms.astype('int64')
            df['ts_iso'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # treat as ISO
            ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df['ts_iso'] = ts.dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['ts_ms'] = (ts.view('int64') // 1_000_000).astype('int64')
    except Exception:
        # Fallback: keep original as string
        df['ts_iso'] = df['timestamp'].astype(str)
        df['ts_ms'] = pd.NA

    return df


def build_signals_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    feat = compute_features(df[['open', 'high', 'low', 'close']].copy())

    codes_col: List[str] = []
    for i in range(len(feat)):
        if i < 3:
            codes_col.append('')
            continue
        codes = detect_signals(feat, i=i, interval=interval)
        # join codes into pipe-separated string for CSV
        codes_col.append('|'.join(codes))

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'ts_iso': df.get('ts_iso', df['timestamp']),
        'ts_ms': df.get('ts_ms', pd.NA),
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'signals': codes_col,
    })
    return out


def main():
    parser = argparse.ArgumentParser(description='Export ETHUSDT C20/MX signals from signals.db')
    parser.add_argument('--db', default=os.path.join(PROJECT_ROOT, 'signals.db'))
    parser.add_argument('--interval', default='15m', help='Label for interval (for TradingView compare)')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'exports', 'ethusdt_c20mx_signals.csv'))
    args = parser.parse_args()

    df = load_ethusdt_from_db(args.db)
    if df.empty:
        print('No ETHUSDT rows found in DB')
        sys.exit(1)

    sig_df = build_signals_df(df, args.interval)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sig_df.to_csv(args.out, index=False)
    print(f'Exported {len(sig_df)} rows to {args.out}')


if __name__ == '__main__':
    main()
