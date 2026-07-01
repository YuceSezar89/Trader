#!/usr/bin/env python3
"""Hesaplanmış sinyalleri interval bazında kontrol et"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
from config import Config

def check_calculated_signals():
    """Interval bazında hesaplanmış sinyal sayısını göster"""
    
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        # Interval bazında hesaplanmış sinyaller
        query = """
            SELECT 
                s.interval,
                COUNT(*) as total_signals,
                COUNT(CASE WHEN sp.is_calculated = true THEN 1 END) as calculated,
                COUNT(CASE WHEN sp.is_calculated = false THEN 1 END) as pending
            FROM signals s
            JOIN signal_performance sp ON s.id = sp.signal_id
            GROUP BY s.interval
            ORDER BY s.interval;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print("=" * 80)
        print("INTERVAL BAZINDA HESAPLANMIŞ SİNYALLER")
        print("=" * 80)
        print(f"{'Interval':<10} {'Total':<10} {'Calculated':<12} {'Pending':<10} {'%':<10}")
        print("-" * 80)
        
        total_all = 0
        calculated_all = 0
        
        for row in results:
            interval, total, calculated, pending = row
            pct = (calculated / total * 100) if total > 0 else 0
            print(f"{interval:<10} {total:<10} {calculated:<12} {pending:<10} {pct:.1f}%")
            total_all += total
            calculated_all += calculated
        
        print("-" * 80)
        pct_all = (calculated_all / total_all * 100) if total_all > 0 else 0
        print(f"{'TOPLAM':<10} {total_all:<10} {calculated_all:<12} {total_all - calculated_all:<10} {pct_all:.1f}%")
        print("=" * 80)
        
        # Top 10 calculated signals (her interval'den)
        print("\n" + "=" * 80)
        print("HER INTERVAL'DEN TOP 5 HESAPLANMIŞ SİNYAL")
        print("=" * 80)
        
        for interval in ['1m', '5m', '15m', '1h', '4h']:
            cursor.execute("""
                SELECT 
                    s.id,
                    s.symbol,
                    s.signal_type,
                    sp.return_t5_atr,
                    sp.return_t5_pct
                FROM signals s
                JOIN signal_performance sp ON s.id = sp.signal_id
                WHERE s.interval = %s
                  AND sp.is_calculated = true
                ORDER BY sp.return_t5_atr DESC
                LIMIT 5
            """, (interval,))
            
            interval_results = cursor.fetchall()
            
            if interval_results:
                print(f"\n{interval} Interval:")
                print(f"  {'ID':<8} {'Symbol':<12} {'Type':<15} {'T+5 ATR':<10} {'T+5 %':<10}")
                print(f"  {'-'*70}")
                for r in interval_results:
                    sig_id, symbol, sig_type, t5_atr, t5_pct = r
                    print(f"  {sig_id:<8} {symbol:<12} {sig_type:<15} {t5_atr:>9.2f} {t5_pct:>9.2f}%")
            else:
                print(f"\n{interval} Interval: Hesaplanmış sinyal yok!")
        
        print("=" * 80)
        
        cursor.close()
        
    finally:
        conn.close()


if __name__ == "__main__":
    check_calculated_signals()
