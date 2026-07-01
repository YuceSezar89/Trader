"""Quick signal summary that prints counts and date range using Config DB settings.

Usage:
    .venv/bin/python scripts/signal_summary.py
"""
import os
import sys
import traceback

# Ensure repo root on path so `from config import Config` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config
import psycopg2


QUERY = """
    SELECT 
        MIN(s.timestamp) as ilk_sinyal,
        MAX(s.timestamp) as son_sinyal,
        COUNT(*) as toplam_sinyal,
        COUNT(DISTINCT s.symbol) as sembol_sayisi,
        COUNT(DISTINCT s.interval) as interval_sayisi,
        COUNT(p.id) as performans_kaydi,
        SUM(CASE WHEN p.is_calculated = TRUE THEN 1 ELSE 0 END) as hesaplanmis
    FROM signals s
    LEFT JOIN signal_performance p ON p.signal_id = s.id
"""


def get_connection():
    pwd = Config.DB_PASSWORD if getattr(Config, 'DB_PASSWORD', '') != '' else None
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=pwd,
    )


def main():
    try:
        conn = get_connection()
    except Exception:
        print("ERROR: could not connect to DB using Config — check environment/.env and Config values")
        traceback.print_exc()
        sys.exit(1)

    try:
        cur = conn.cursor()
        cur.execute(QUERY)
        row = cur.fetchone()
        if row is None:
            print("No data returned.")
        else:
            print(f"İlk sinyal        : {row[0]}")
            print(f"Son sinyal        : {row[1]}")
            print(f"Toplam sinyal     : {row[2]}")
            print(f"Sembol sayısı     : {row[3]}")
            print(f"Interval sayısı   : {row[4]}")
            print(f"Performans kaydı  : {row[5]}")
            print(f"Hesaplanmış       : {row[6]}")
        cur.close()
    except Exception:
        print("ERROR: query failed")
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
