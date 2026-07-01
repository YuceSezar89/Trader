#!/usr/bin/env python3
"""Interval bazında signal performance durumunu kontrol et"""

import psycopg2
from config import Config

def check_interval_performance():
    """Interval bazında sinyal dağılımını göster"""
    
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        query = """
            SELECT 
                s.interval,
                COUNT(*) as total_signals,
                COUNT(CASE WHEN sp.is_calculated = true THEN 1 END) as calculated,
                COUNT(CASE WHEN sp.is_calculated = false THEN 1 END) as pending,
                COUNT(CASE WHEN sp.signal_id IS NULL THEN 1 END) as no_performance_record,
                MIN(s.timestamp) as oldest_signal,
                MAX(s.timestamp) as newest_signal
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            GROUP BY s.interval
            ORDER BY s.interval;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print("=" * 100)
        print("INTERVAL BAZINDA SİNYAL PERFORMANS DURUMU")
        print("=" * 100)
        print(f"{'Interval':<10} {'Total':<10} {'Calculated':<12} {'Pending':<10} {'No Record':<12} {'Oldest':<20} {'Newest':<20}")
        print("-" * 100)
        
        for row in results:
            interval, total, calculated, pending, no_record, oldest, newest = row
            print(f"{interval:<10} {total:<10} {calculated:<12} {pending:<10} {no_record:<12} {str(oldest):<20} {str(newest):<20}")
        
        print("=" * 100)
        
        # Özet
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT s.id) as total_signals,
                COUNT(DISTINCT sp.signal_id) as has_performance_record,
                COUNT(DISTINCT CASE WHEN sp.is_calculated = true THEN sp.signal_id END) as calculated
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
        """)
        
        total_signals, has_record, calculated = cursor.fetchone()
        
        print("\nÖZET:")
        print(f"  Toplam Sinyal: {total_signals}")
        print(f"  Performance Kaydı Olan: {has_record} ({has_record/total_signals*100:.1f}%)")
        print(f"  Hesaplanmış: {calculated} ({calculated/total_signals*100:.1f}%)")
        print(f"  Performance Kaydı Yok: {total_signals - has_record}")
        
        cursor.close()
        
    finally:
        conn.close()


if __name__ == "__main__":
    check_interval_performance()
