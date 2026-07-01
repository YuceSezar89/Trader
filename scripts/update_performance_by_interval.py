#!/usr/bin/env python3
"""Her interval için ayrı ayrı performance hesapla"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from config import Config
from signals.signal_performance_analyzer import SignalPerformanceAnalyzer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_by_interval(interval: str, max_signals: int = 1000):
    """Belirli bir interval için performance hesapla"""
    
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        # Bu interval'de hesaplanmamış sinyalleri bul
        cursor.execute("""
            SELECT sp.signal_id
            FROM signal_performance sp
            JOIN signals s ON s.id = sp.signal_id
            WHERE s.interval = %s
              AND sp.is_calculated = FALSE
            ORDER BY s.timestamp DESC
            LIMIT %s
        """, (interval, max_signals))
        
        signal_ids = [row[0] for row in cursor.fetchall()]
        
        if not signal_ids:
            logger.info(f"{interval}: Hesaplanacak sinyal yok")
            return 0
        
        logger.info(f"{interval}: {len(signal_ids)} sinyal hesaplanacak")
        
        # Performance analyzer
        analyzer = SignalPerformanceAnalyzer()
        
        success = 0
        failed = 0
        
        for signal_id in signal_ids:
            try:
                if analyzer.update_signal_performance(signal_id):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Signal {signal_id} hatası: {e}")
                failed += 1
        
        logger.info(f"{interval}: ✅ {success} başarılı, ❌ {failed} başarısız")
        
        cursor.close()
        return success
        
    finally:
        conn.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Interval Bazında Performance Hesaplama")
    logger.info("=" * 60)
    
    intervals = ['5m', '15m', '1h', '4h']
    
    for interval in intervals:
        logger.info(f"\n{'='*60}")
        logger.info(f"İşleniyor: {interval}")
        logger.info(f"{'='*60}")
        update_by_interval(interval, max_signals=1000)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Tüm interval'ler tamamlandı!")
    logger.info("=" * 60)
