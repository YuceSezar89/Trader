#!/usr/bin/env python3
"""
Signal Performance Batch Update
================================
Son N saatteki sinyallerin performansını toplu günceller.

Kullanım:
    python run_performance_update.py                # Son 24 saat
    python run_performance_update.py --hours 48     # Son 48 saat
    python run_performance_update.py --test         # Test modu (10 sinyal)
    python run_performance_update.py --signal-id 123  # Tek sinyal
"""

import argparse
import logging
import sys
from signals.signal_performance_analyzer import SignalPerformanceAnalyzer

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description='Signal Performance Batch Update')
    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Kaç saat geriye bak (default: 24)'
    )
    parser.add_argument(
        '--max-signals',
        type=int,
        default=None,
        help='Maksimum sinyal sayısı (default: sınırsız)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test modu (sadece 10 sinyal)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Hesaplamaları yap ama DB yazma (sadece logla)'
    )
    parser.add_argument(
        '--signal-id',
        type=int,
        default=None,
        help='Tek bir sinyal ID güncelle'
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("  SIGNAL PERFORMANCE BATCH UPDATE")
    print("=" * 60)
    print()
    
    try:
        with SignalPerformanceAnalyzer() as analyzer:
            
            if args.signal_id:
                # Tek sinyal güncelle
                print(f"🔄 Signal ID {args.signal_id} güncelleniyor...")
                print()
                success = analyzer.update_signal_performance(args.signal_id, dry_run=args.dry_run)
                
                if success:
                    print()
                    print("=" * 60)
                    print("✅ Güncelleme başarılı!")
                    print("=" * 60)
                    print()
                else:
                    print()
                    print("=" * 60)
                    print("❌ Güncelleme başarısız!")
                    print("=" * 60)
                    print()
                    sys.exit(1)
            
            else:
                # Batch güncelleme
                hours = args.hours
                max_signals = 10 if args.test else args.max_signals
                
                print(f"⏰ Zaman aralığı: Son {hours} saat")
                if max_signals:
                    print(f"📊 Maksimum sinyal: {max_signals}")
                print()
                
                print("🔄 Performans hesaplamaları başlıyor...")
                print()
                stats = analyzer.batch_update_performance(
                    hours_back=hours,
                    max_signals=max_signals,
                    dry_run=args.dry_run
                )
                
                print()
                print("=" * 60)
                print("✅ Batch güncelleme tamamlandı!")
                print("=" * 60)
                print()
                print(f"📊 İstatistikler:")
                print(f"  • Toplam sinyal: {stats['total']}")
                print(f"  • Başarılı: {stats['success']}")
                print(f"  • Başarısız: {stats['failed']}")
                print(f"  • Atlanan: {stats['skipped']}")
                print()
                
                if stats['failed'] > 0:
                    print("⚠️  Bazı sinyaller güncellenemedi. Log'ları kontrol edin.")
                    print()
    
    except KeyboardInterrupt:
        print()
        print("⚠️  İşlem kullanıcı tarafından iptal edildi")
        print()
        sys.exit(1)
    
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Hata: {e}")
        print("=" * 60)
        print()
        logger.exception("Beklenmeyen hata")
        sys.exit(1)


if __name__ == "__main__":
    main()
