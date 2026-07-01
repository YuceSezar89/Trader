#!/usr/bin/env python3
"""
MTF Backfill Test ve Çalıştırma Scripti
"""

import asyncio
import sys
import argparse
from backtest.mtf_backfill import run_mtf_backfill

def main():
    parser = argparse.ArgumentParser(description='MTF Backfill Sistemi')
    parser.add_argument('--symbols', nargs='+', help='İşlenecek semboller (örn: BTCUSDT ETHUSDT)')
    parser.add_argument('--days', type=int, default=30, help='Kaç gün geriye gidilecek (varsayılan: 30)')
    parser.add_argument('--test', action='store_true', help='Test modu (sadece 2 sembol, 7 gün)')
    
    args = parser.parse_args()
    
    print("🚀 MTF BACKFILL SİSTEMİ")
    print("=" * 40)
    
    if args.test:
        print("🧪 TEST MODU AKTIF")
        symbols = ['BTCUSDT', 'ETHUSDT']
        days_back = 7
    else:
        symbols = args.symbols
        days_back = args.days
    
    print(f"📊 Semboller: {symbols if symbols else 'TÜM SEMBOLLER'}")
    print(f"📅 Geri gidiş: {days_back} gün")
    
    # Onay al
    if not args.test:
        confirm = input("\n⚠️ Bu işlem database'e yeni sinyaller ekleyecek. Devam etmek istiyor musun? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ İşlem iptal edildi.")
            sys.exit(0)
    
    # Backfill çalıştır
    try:
        stats = asyncio.run(run_mtf_backfill(symbols=symbols, days_back=days_back))
        
        print("\n✅ BAŞARILI!")
        print(f"Panel'de artık {sum(stats.timeframe_signals.values())} yeni MTF sinyali göreceksin!")
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
