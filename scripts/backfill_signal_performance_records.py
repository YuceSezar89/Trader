#!/usr/bin/env python3
"""
Mevcut tüm sinyaller için signal_performance kayıtları oluştur.
Trigger öncesi oluşturulmuş sinyaller için retroaktif kayıt.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_performance_records():
    """Performance kaydı olmayan tüm sinyaller için kayıt oluştur"""
    
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        # Performance kaydı olmayan sinyalleri bul
        logger.info("Performance kaydı olmayan sinyaller bulunuyor...")
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            WHERE sp.signal_id IS NULL
        """)
        
        missing_count = cursor.fetchone()[0]
        logger.info(f"{missing_count} sinyal için kayıt oluşturulacak")
        
        if missing_count == 0:
            logger.info("Tüm sinyallerin performance kaydı mevcut!")
            return
        
        # Toplu insert
        logger.info("Kayıtlar oluşturuluyor...")
        
        cursor.execute("""
            INSERT INTO signal_performance (
                signal_id,
                entry_price,
                entry_timestamp,
                atr_at_entry,
                interval,
                is_calculated
            )
            SELECT 
                s.id,
                s.price,
                s.timestamp,
                CASE 
                    WHEN s.atr IS NULL OR s.atr = 0 THEN 0.01
                    ELSE s.atr
                END,
                s.interval,
                FALSE
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            WHERE sp.signal_id IS NULL
            ON CONFLICT (signal_id) DO NOTHING;
        """)
        
        inserted = cursor.rowcount
        conn.commit()
        
        logger.info(f"✅ {inserted} kayıt başarıyla oluşturuldu!")
        
        # Özet
        cursor.execute("""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(sp.signal_id) as has_record,
                COUNT(CASE WHEN sp.is_calculated = TRUE THEN 1 END) as calculated
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
        """)
        
        total, has_record, calculated = cursor.fetchone()
        
        logger.info("=" * 60)
        logger.info("GÜNCEL DURUM:")
        logger.info(f"  Toplam Sinyal: {total}")
        logger.info(f"  Performance Kaydı Olan: {has_record} ({has_record/total*100:.1f}%)")
        logger.info(f"  Hesaplanmış: {calculated} ({calculated/total*100:.1f}%)")
        logger.info(f"  Hesaplanacak: {has_record - calculated}")
        logger.info("=" * 60)
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Hata: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Signal Performance Kayıt Backfill")
    logger.info("=" * 60)
    backfill_performance_records()
    logger.info("\n✅ İşlem tamamlandı!")
    logger.info("\nŞimdi performance hesaplama çalıştırabilirsin:")
    logger.info("  python scripts/update_signal_performance.py --hours-back 8760 --max-signals 10000")
