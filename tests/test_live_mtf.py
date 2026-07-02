#!/usr/bin/env python3
"""
Live MTF System Test

Bu script canlı MTF sistemini test eder.
"""

import pytest

pytest.skip(
    "Manuel entegrasyon scripti - pytest testi degil; calistirmak icin: python %s" % __file__,
    allow_module_level=True,
)


import asyncio
import sys
import os
import signal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_data_manager import LiveDataManager
from database.crud import initialize_database
from binance_client import BinanceClientManager
from config import Config
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveMTFTester:
    def __init__(self):
        self.manager = None
        self.running = False
    
    async def test_database_connection(self):
        """Test database connection"""
        print("🔍 Database bağlantısı test ediliyor...")
        try:
            await initialize_database()
            print("✅ Database bağlantısı başarılı")
            return True
        except Exception as e:
            print(f"❌ Database bağlantı hatası: {e}")
            return False
    
    async def test_binance_connection(self):
        """Test Binance API connection"""
        print("🔍 Binance API bağlantısı test ediliyor...")
        try:
            symbols = await BinanceClientManager.get_top_volume_symbols_async(limit=5)
            print(f"✅ Binance API bağlantısı başarılı. {len(symbols)} sembol alındı")
            return True, symbols
        except Exception as e:
            print(f"❌ Binance API bağlantı hatası: {e}")
            return False, []
    
    async def test_mtf_initialization(self, symbols):
        """Test MTF system initialization"""
        print("🔍 MTF sistemi başlatılıyor...")
        try:
            # Küçük sembol listesi ile test
            test_symbols = symbols[:3] if len(symbols) >= 3 else symbols
            self.manager = LiveDataManager(symbols=test_symbols, interval='1m')
            
            print(f"✅ MTF sistemi başlatıldı")
            print(f"  MTF Enabled: {self.manager.mtf_enabled}")
            print(f"  Timeframes: {self.manager.supported_timeframes}")
            print(f"  Symbols: {self.manager.symbols}")
            print(f"  MTF Buffers: {len(self.manager.mtf_buffers) if hasattr(self.manager, 'mtf_buffers') else 0}")
            
            return True
        except Exception as e:
            print(f"❌ MTF sistem başlatma hatası: {e}")
            return False
    
    async def test_historical_sync(self):
        """Test historical data sync"""
        print("🔍 Tarihsel veri senkronizasyonu test ediliyor...")
        try:
            await self.manager.sync_historical_data()
            print("✅ Tarihsel veri senkronizasyonu başarılı")
            return True
        except Exception as e:
            print(f"❌ Tarihsel veri senkronizasyon hatası: {e}")
            return False
    
    async def test_mtf_initialization_with_data(self):
        """Test MTF buffer initialization with real data"""
        print("🔍 MTF buffer'ları gerçek veri ile başlatılıyor...")
        try:
            await self.manager._initialize_dataframes()
            if self.manager.mtf_enabled:
                await self.manager._initialize_mtf_dataframes()
            
            # MTF stats kontrolü
            if hasattr(self.manager, 'mtf_buffers'):
                stats = self.manager.get_mtf_stats()
                print("✅ MTF buffer'ları başarıyla başlatıldı")
                print("📊 MTF Buffer Stats:")
                for symbol, tf_stats in stats.items():
                    print(f"  {symbol}: {tf_stats}")
                return True
            else:
                print("⚠️ MTF buffers mevcut değil")
                return False
                
        except Exception as e:
            print(f"❌ MTF buffer başlatma hatası: {e}")
            return False
    
    async def test_redis_mtf_cache(self):
        """Test Redis MTF cache"""
        print("🔍 Redis MTF cache test ediliyor...")
        try:
            from utils.redis_client import RedisClient
            
            # Her sembol için cache kontrolü
            for symbol in self.manager.symbols:
                for tf in self.manager.supported_timeframes:
                    cached_df = await RedisClient.get_mtf_klines(symbol, tf)
                    if cached_df is not None and not cached_df.empty:
                        print(f"  ✅ {symbol} {tf}: {len(cached_df)} bars cached")
                    else:
                        print(f"  ⚠️ {symbol} {tf}: No cache data")
            
            print("✅ Redis MTF cache testi tamamlandı")
            return True
            
        except Exception as e:
            print(f"❌ Redis MTF cache test hatası: {e}")
            return False
    
    async def run_short_live_test(self, duration=30):
        """Run short live test"""
        print(f"🔍 {duration} saniye canlı test başlatılıyor...")
        print("  WebSocket bağlantısı kuruluyor...")
        
        try:
            # WebSocket'i başlat
            await self.manager.start_streams()
            
            if self.manager.is_ws_connected:
                print("✅ WebSocket bağlantısı başarılı")
                print(f"⏰ {duration} saniye bekleniyor...")
                
                # Belirtilen süre kadar bekle
                await asyncio.sleep(duration)
                
                # Son durumu kontrol et
                if hasattr(self.manager, 'mtf_buffers'):
                    stats = self.manager.get_mtf_stats()
                    print("📊 Test sonrası MTF Stats:")
                    for symbol, tf_stats in stats.items():
                        print(f"  {symbol}: {tf_stats}")
                
                print("✅ Canlı test başarıyla tamamlandı")
                return True
            else:
                print("❌ WebSocket bağlantısı kurulamadı")
                return False
                
        except Exception as e:
            print(f"❌ Canlı test hatası: {e}")
            return False
        finally:
            # Temizlik
            if self.manager:
                await self.manager.shutdown()
    
    async def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Canlı MTF Sistem Test Başlatılıyor")
        print("=" * 60)
        
        # Test adımları
        tests = [
            ("Database Connection", self.test_database_connection),
            ("Binance API Connection", self.test_binance_connection),
        ]
        
        results = {}
        symbols = []
        
        # İlk testler
        for test_name, test_func in tests:
            try:
                if test_name == "Binance API Connection":
                    result, symbols = await test_func()
                else:
                    result = await test_func()
                results[test_name] = result
                
                if not result:
                    print(f"❌ {test_name} başarısız, test durduruluyor")
                    return False
                    
            except Exception as e:
                print(f"❌ {test_name} test hatası: {e}")
                results[test_name] = False
                return False
        
        # MTF testleri
        mtf_tests = [
            ("MTF Initialization", lambda: self.test_mtf_initialization(symbols)),
            ("Historical Sync", self.test_historical_sync),
            ("MTF Buffer Init", self.test_mtf_initialization_with_data),
            ("Redis MTF Cache", self.test_redis_mtf_cache),
        ]
        
        for test_name, test_func in mtf_tests:
            try:
                result = await test_func()
                results[test_name] = result
                
                if not result:
                    print(f"⚠️ {test_name} başarısız, devam ediliyor...")
                    
            except Exception as e:
                print(f"❌ {test_name} test hatası: {e}")
                results[test_name] = False
        
        # Kısa canlı test
        if results.get("MTF Initialization", False):
            try:
                live_result = await self.run_short_live_test(duration=15)  # 15 saniye
                results["Live Test"] = live_result
            except Exception as e:
                print(f"❌ Live test hatası: {e}")
                results["Live Test"] = False
        
        # Sonuçları özetle
        print("\n" + "=" * 60)
        print("📋 TEST SONUÇLARI:")
        print("=" * 60)
        
        success_count = 0
        total_count = len(results)
        
        for test_name, result in results.items():
            status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
            print(f"  {test_name}: {status}")
            if result:
                success_count += 1
        
        print(f"\n📊 Genel Başarı Oranı: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count >= total_count * 0.8:  # %80+ başarı
            print("🎉 MTF sistemi canlı olarak çalışmaya hazır!")
            return True
        else:
            print("⚠️ MTF sistemi bazı sorunlar içeriyor")
            return False

async def main():
    """Main test function"""
    tester = LiveMTFTester()
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Test durduruldu")
        tester.running = False
        if tester.manager:
            asyncio.create_task(tester.manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = await tester.run_full_test()
        return success
    except KeyboardInterrupt:
        print("\n🛑 Test kullanıcı tarafından durduruldu")
        return False
    except Exception as e:
        print(f"\n❌ Test genel hatası: {e}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test sonlandırıldı")
        sys.exit(0)
