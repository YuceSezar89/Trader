"""
Backtest Runner - Basit backtest çalıştırma script'i
"""

import sys
import logging
from datetime import datetime
from utils.simple_backtest import run_simple_backtest

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_results(results):
    """Sonuçları güzel formatta yazdır"""
    
    print("\n" + "="*80)
    print("📊 BACKTEST SONUÇLARI")
    print("="*80)
    
    print(f"\n🎯 Strateji: {results['strategy_name']}")
    print(f"⏰ Timeframe: {results['timeframe']}")
    print(f"📅 Test Süresi: {results['days_back']} gün")
    
    print(f"\n📈 TEMEL METRİKLER")
    print(f"   Toplam İşlem: {results['total_trades']}")
    print(f"   Kazanan: {results['winning_trades']} ✅")
    print(f"   Kaybeden: {results['losing_trades']} ❌")
    print(f"   Win Rate: {results['win_rate']:.2f}%")
    
    print(f"\n💰 PERFORMANS")
    print(f"   Toplam PnL: {results['total_pnl_pct']:.2f}%")
    print(f"   Ortalama PnL: {results['avg_pnl_pct']:.2f}%")
    print(f"   Ortalama Kazanç: {results['avg_win']:.2f}%")
    print(f"   Ortalama Kayıp: {results['avg_loss']:.2f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    
    print(f"\n⚠️  RİSK METRİKLERİ")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Max Ardışık Kazanç: {results['max_consecutive_wins']}")
    print(f"   Max Ardışık Kayıp: {results['max_consecutive_losses']}")
    
    if results.get('exit_reasons'):
        print(f"\n🚪 ÇIKIŞ NEDENLERİ")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    
    # Değerlendirme
    print("\n💡 DEĞERLENDİRME:")
    
    if results['win_rate'] >= 60:
        print("   ✅ Win rate mükemmel (>60%)")
    elif results['win_rate'] >= 50:
        print("   ⚠️  Win rate orta (50-60%)")
    else:
        print("   ❌ Win rate düşük (<50%)")
    
    if results['profit_factor'] >= 2.0:
        print("   ✅ Profit factor mükemmel (>2.0)")
    elif results['profit_factor'] >= 1.5:
        print("   ⚠️  Profit factor orta (1.5-2.0)")
    else:
        print("   ❌ Profit factor düşük (<1.5)")
    
    if results['max_drawdown'] <= 10:
        print("   ✅ Max drawdown iyi (<10%)")
    elif results['max_drawdown'] <= 20:
        print("   ⚠️  Max drawdown orta (10-20%)")
    else:
        print("   ❌ Max drawdown yüksek (>20%)")
    
    print("\n" + "="*80 + "\n")


def main():
    """Ana fonksiyon"""
    
    print("\n🚀 Backtest Başlatılıyor...\n")
    
    days_back = 7
    intervals = ['15m']
    min_vpm = 10.0
    signal_filter = None
    print(f"📊 Parametreler (OPTIMIZE):")
    print(f"   Gün: {days_back}")
    print(f"   Interval'ler: {intervals}")
    print(f"   Min VPM Skoru: {min_vpm}")
    print()
    
    try:
        # Backtest çalıştır
        results = run_simple_backtest(
            days_back=days_back,
            intervals=intervals,
            min_vpm_score=min_vpm
        )
        
        # Sonuçları yazdır
        print_results(results)
        
        # Database ID
        if results.get('backtest_id'):
            print(f"✅ Sonuçlar database'e kaydedildi (ID: {results['backtest_id']})")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest hatası: {e}", exc_info=True)
        print(f"\n❌ HATA: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
