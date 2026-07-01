"""Analyze signal performance by querying the project's database using Config.

This is adapted from utils/sorgu.txt to use Config and be runnable from the repo root.
"""
import sys
import os
import traceback
import psycopg2
import pandas as pd
import numpy as np

# Ensure project root is on sys.path so `from config import Config` works when
# running this script from the repository root or the scripts/ folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config


def get_connection():
    pwd = Config.DB_PASSWORD if Config.DB_PASSWORD != '' else None
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=pwd,
    )


QUERY = """
    SELECT 
        s.id,
        s.symbol,
        s.timestamp,
        s.signal_type,
        s.interval,
        s.price,
        s.strength,
        s.rsi,
        s.macd,
        s.momentum,
        s.atr,
        s.adx,
        s.sharpe_ratio,
        s.sortino_ratio,
        s.vpms_score,
        s.mtf_score,
        p.return_t3_pct,
        p.return_t5_pct,
        p.return_t10_pct,
        p.return_t3_atr,
        p.return_t5_atr,
        p.return_t10_atr,
        p.mfe_atr,
        p.mae_atr,
        p.risk_reward,
        p.is_calculated
    FROM signals s
    JOIN signal_performance p ON p.signal_id = s.id
    WHERE p.is_calculated = TRUE
    ORDER BY s.timestamp DESC
"""


def main():
    try:
        conn = get_connection()
    except Exception as e:
        print("ERROR: Could not connect to database using Config settings.")
        print("Config used:")
        print(f"  host={Config.DB_HOST} port={Config.DB_PORT} db={Config.DB_NAME} user={Config.DB_USER}")
        print("Make sure the DB is running and credentials are correct (check .env or environment variables).")
        print()
        traceback.print_exc()
        sys.exit(1)

    try:
        df = pd.read_sql(QUERY, conn)
    except Exception as e:
        print("ERROR: Query failed")
        traceback.print_exc()
        conn.close()
        sys.exit(1)

    conn.close()

    print(f"Toplam sinyal: {len(df)}")
    if len(df) == 0:
        print("No calculated signal_performance rows found (p.is_calculated = TRUE). Exiting.")
        return

    print(f"Semboller (unique count): {df['symbol'].nunique()}")
    print(f"Sinyal tipleri: {list(df['signal_type'].unique())}")
    print(f"Interval'ler: {list(df['interval'].unique())}")
    try:
        print(f"\nTarih aralığı: {df['timestamp'].min()} → {df['timestamp'].max()}")
    except Exception:
        pass

    # --- TEMEL İSTATİSTİK ---
    print("\n=== GENEL PERFORMANS ===")
    for t in [3, 5, 10]:
        col = f'return_t{t}_pct'
        if col in df.columns:
            returns = df[col].dropna()
            if len(returns) == 0:
                continue
            win_rate = (returns > 0).mean() * 100
            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()
            ev = returns.mean()
            print(f"\nT+{t}:")
            print(f"  Win Rate      : {win_rate:.1f}%")
            print(f"  Avg Win       : {avg_win:.4f}%")
            print(f"  Avg Loss      : {avg_loss:.4f}%")
            print(f"  Expected Value: {ev:.4f}%")

    # --- SİNYAL TİPİNE GÖRE ---
    print("\n=== SİNYAL TİPİNE GÖRE (T+5) ===")
    if 'return_t5_pct' in df.columns:
        grouped = df.groupby('signal_type')['return_t5_pct'].agg([
            ('count', 'count'),
            ('win_rate', lambda x: (x > 0).mean() * 100),
            ('avg_return', 'mean'),
            ('std', 'std')
        ])
        print(grouped.round(4))

    # --- INTERVAL'E GÖRE ---
    print("\n=== INTERVAL'E GÖRE (T+5) ===")
    if 'return_t5_pct' in df.columns:
        grouped = df.groupby('interval')['return_t5_pct'].agg([
            ('count', 'count'),
            ('win_rate', lambda x: (x > 0).mean() * 100),
            ('avg_return', 'mean')
        ])
        print(grouped.round(4))

    # --- STRENGTH VARSA ---
    if 'strength' in df.columns and df['strength'].notna().any():
        print("\n=== STRENGTH'E GÖRE (T+5) ===")
        grouped = df.groupby('strength')['return_t5_pct'].agg([
            ('count', 'count'),
            ('win_rate', lambda x: (x > 0).mean() * 100),
            ('avg_return', 'mean')
        ])
        print(grouped.round(4))

    # --- USER FILTER: Long signals, intervals 15m/1h/4h, strength >= 2 ---
    try:
        filtreli = df[
            (df['signal_type'] == 'Long') &
            (df['interval'].isin(['15m', '1h', '4h'])) &
            (df['strength'] >= 2)
        ]
    except KeyError:
        print("One or more required columns for the filter are missing: 'signal_type', 'interval', 'strength'")
        return

    print("\n=== KULLANICI FİLTRESİ SONUÇLARI ===")
    print(f"Kalan sinyal: {len(filtreli)}")
    if 'return_t5_pct' in filtreli.columns and len(filtreli) > 0:
        win_rate = (filtreli['return_t5_pct'] > 0).mean() * 100
        avg_return = filtreli['return_t5_pct'].mean()
        ev = avg_return
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Return: {avg_return:.2f}%")
        print(f"Expected Value: {ev:.3f}%")
    else:
        print("No return_t5_pct data available for the filtered set.")

    # --- Sembole göre dağılım: hangi semboller sonucu şişiriyor? (top 20) ---
    if len(filtreli) > 0 and 'symbol' in filtreli.columns and 'return_t5_pct' in filtreli.columns:
        symbol_stats = (
            filtreli.groupby('symbol')['return_t5_pct']
            .agg(count='count', win_rate=lambda x: (x > 0).mean() * 100, mean='mean')
            .round(2)
            .sort_values('mean', ascending=False)
        )

        print("\n=== SEMBOL DAĞILIMI (Top 20 by mean return) ===")
        print(symbol_stats.head(20).to_string())

        # Filter out low-count symbols to avoid noisy / inflated results
        min_count = 10
        robust = symbol_stats[symbol_stats['count'] >= min_count].sort_values('mean', ascending=False)
        print(f"\n=== SEMBOL DAĞILIMI (count >= {min_count}) Top 20 ===")
        if len(robust) == 0:
            print(f"No symbols with count >= {min_count} in the filtered set.")
        else:
            print(robust.head(20).to_string())


if __name__ == '__main__':
    main()
