#!/usr/bin/env python3
"""Top50 volume coins'in sinyal/performance özetini gösterir.

Kullanım: proje dizininde çalıştırın. Bağlantı bilgileri `config.Config` üzerinden alınır.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from config import Config


def main():
    conn = None
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            connect_timeout=Config.DB_TIMEOUT
        )

        cur = conn.cursor()

        q = '''
        WITH recent AS (
            SELECT symbol, SUM(quote_asset_volume) AS vol24
            FROM price_data
            WHERE timeframe = '1m' AND timestamp >= now() - interval '24 hours'
            GROUP BY symbol
        ), top AS (
            SELECT symbol, vol24 FROM recent ORDER BY vol24 DESC LIMIT 50
        )
        SELECT
            t.symbol,
            t.vol24,
            COALESCE(s.total_signals, 0) AS total_signals,
            COALESCE(sp.perf_count, 0) AS perf_count,
            COALESCE(s.total_signals,0) - COALESCE(sp.perf_count,0) AS missing
        FROM top t
        LEFT JOIN (
            SELECT symbol, COUNT(*) AS total_signals FROM signals GROUP BY symbol
        ) s ON s.symbol = t.symbol
        LEFT JOIN (
            SELECT s.symbol, COUNT(sp.*) AS perf_count FROM signals s JOIN signal_performance sp ON s.id = sp.signal_id GROUP BY s.symbol
        ) sp ON sp.symbol = t.symbol
        ORDER BY t.vol24 DESC;
        '''

        cur.execute(q)
        rows = cur.fetchall()

        if not rows:
            print("Hiç satır dönmedi. price_data veya sinyal tabloları boş olabilir veya DB bağlantısı sorunlu.")
            return

        print(f"{'SYMBOL':<12} {'VOL24':>15} {'TOTAL':>8} {'PERF':>8} {'MISSING':>8}")
        print('-'*60)
        for r in rows:
            symbol, vol24, total_signals, perf_count, missing = r
            vol_fmt = f"{vol24:,.2f}" if vol24 is not None else '0'
            print(f"{symbol:<12} {vol_fmt:>15} {total_signals:8d} {perf_count:8d} {missing:8d}")

        cur.close()

    except Exception as e:
        print(f"Hata: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    main()
