#!/usr/bin/env python3
"""price_data tablosunda hangi interval'lerde veri var kontrol et"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
from config import Config

def check_price_data():
    """price_data tablosundaki interval dağılımını göster"""
    
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        # Interval bazında veri sayısı
        cursor.execute("""
            SELECT 
                interval,
                COUNT(*) as total_rows,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest
            FROM price_data
            GROUP BY interval
            ORDER BY interval;
        """)
        
        results = cursor.fetchall()
        
        print("=" * 100)
        print("PRICE_DATA TABLOSU - INTERVAL DAĞILIMI")
        print("=" * 100)
        print(f"{'Interval':<10} {'Total Rows':<15} {'Symbols':<10} {'Oldest':<20} {'Newest':<20}")
        print("-" * 100)
        
        if results:
            for row in results:
                interval, total, symbols, oldest, newest = row
                print(f"{interval:<10} {total:<15} {symbols:<10} {str(oldest):<20} {str(newest):<20}")
        else:
            print("❌ price_data tablosu BOŞ!")
        
        print("=" * 100)
        
        # Toplam
        cursor.execute("SELECT COUNT(*) FROM price_data")
        total = cursor.fetchone()[0]
        print(f"\nToplam kayıt: {total:,}")
        
        cursor.close()
        
    finally:
        conn.close()


if __name__ == "__main__":
    check_price_data()
