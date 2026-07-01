#!/usr/bin/env python3
"""
Signal Performance Batch Update Script
=======================================
Günlük otomatik çalıştırma için optimize edilmiş batch update scripti.
"""

import sys
import argparse
from pathlib import Path

# Project root'u path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.signal_performance_analyzer import SignalPerformanceAnalyzer
from config import Config
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description='Signal Performance Batch Update')
    parser.add_argument('--hours-back', type=int, default=720, help='Kaç saat geriye bak (default: 720 = 30 gün)')
    parser.add_argument('--max-signals', type=int, default=None, help='Maksimum sinyal sayısı (default: None = hepsi)')
    parser.add_argument('--test', action='store_true', help='Test mode (dry run)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Signal Performance Batch Update")
    logger.info("=" * 60)
    logger.info(f"Hours back: {args.hours_back} ({args.hours_back/24:.1f} days)")
    logger.info(f"Max signals: {args.max_signals or 'All'}")
    logger.info(f"Test mode: {args.test}")
    logger.info("=" * 60)
    
    try:
        # Analyzer oluştur (kendi bağlantısını oluşturur)
        analyzer = SignalPerformanceAnalyzer()
        
        # Batch update çalıştır
        logger.info("Starting batch update...")
        
        result = analyzer.batch_update_performance(
            hours_back=args.hours_back,
            max_signals=args.max_signals
        )
        
        # Sonuçları logla
        logger.info("=" * 60)
        logger.info("RESULTS:")
        logger.info(f"  Total processed: {result.get('total', 0)}")
        logger.info(f"  Successful: {result.get('success', 0)}")
        logger.info(f"  Failed: {result.get('failed', 0)}")
        logger.info(f"  Skipped: {result.get('skipped', 0)}")
        
        if result.get('errors'):
            logger.warning(f"  Errors: {len(result['errors'])}")
            for error in result['errors'][:5]:  # İlk 5 hatayı göster
                logger.warning(f"    - {error}")
        
        logger.info("=" * 60)
        
        # Exit code
        if result.get('failed', 0) > 0:
            logger.warning("Some updates failed!")
            sys.exit(1)
        else:
            logger.info("Batch update completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
