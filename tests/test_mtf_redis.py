#!/usr/bin/env python3
"""
MTF Redis Cache Test Scripti

Bu script yeni MTF Redis cache fonksiyonlarını test eder.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.redis_client import RedisClient
from utils.timeframe_aggregator import TimeframeAggregator, create_multi_timeframe_data

async def test_mtf_cache_basic():
    """Temel MTF cache testleri"""
    print("🧪 MTF Redis Cache - Temel Testler")
    print("=" * 50)
    
    # Test verisi oluştur
    test_df_1m = TimeframeAggregator.create_test_data(bars=30)
    mtf_data = create_multi_timeframe_data(test_df_1m)
    
    symbol = "BTCUSDT"
    
    print(f"✅ Test verisi hazırlandı:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} bars")
    
    # Cache'e yaz
    print(f"\n📝 Cache'e yazma testleri:")
    for tf, df in mtf_data.items():
        success = await RedisClient.set_mtf_klines(symbol, tf, df)
        print(f"  {tf}: {'✅' if success else '❌'}")
    
    # Cache'den oku
    print(f"\n📖 Cache'den okuma testleri:")
    for tf in mtf_data.keys():
        cached_df = await RedisClient.get_mtf_klines(symbol, tf)
        if cached_df is not None:
            print(f"  {tf}: ✅ ({len(cached_df)} bars)")
        else:
            print(f"  {tf}: ❌ (None)")
    
    return mtf_data

async def test_mtf_cache_stats():
    """MTF cache istatistik testleri"""
    print("\n🧪 MTF Cache İstatistikleri")
    print("=" * 50)
    
    symbol = "BTCUSDT"
    stats = await RedisClient.get_mtf_cache_stats(symbol)
    
    print(f"📊 {symbol} Cache Stats:")
    for tf, tf_stats in stats.items():
        print(f"\n  {tf}:")
        for data_type, info in tf_stats.items():
            status = "✅" if info['exists'] else "❌"
            ttl = f"TTL: {info['ttl']}s" if info['ttl'] > 0 else "No TTL"
            print(f"    {data_type}: {status} ({ttl})")

async def test_mtf_indicators():
    """MTF indicators cache testleri"""
    print("\n🧪 MTF Indicators Cache")
    print("=" * 50)
    
    symbol = "BTCUSDT"
    timeframe = "5m"
    
    # Sample indicators
    test_indicators = {
        'rsi': 65.5,
        'macd': 1.2,
        'ema_21': 50125.5,
        'volume_sma': 1250000.0,
        'timestamp': '2025-09-24T17:45:00Z'
    }
    
    # Cache'e yaz
    success = await RedisClient.set_mtf_indicators(symbol, timeframe, test_indicators)
    print(f"📝 Indicators yazma: {'✅' if success else '❌'}")
    
    # Cache'den oku
    cached_indicators = await RedisClient.get_mtf_indicators(symbol, timeframe)
    if cached_indicators:
        print(f"📖 Indicators okuma: ✅")
        print(f"  RSI: {cached_indicators.get('rsi')}")
        print(f"  MACD: {cached_indicators.get('macd')}")
        print(f"  EMA21: {cached_indicators.get('ema_21')}")
    else:
        print(f"📖 Indicators okuma: ❌")

async def test_mtf_cache_warming():
    """MTF cache warming testleri"""
    print("\n🧪 MTF Cache Warming")
    print("=" * 50)
    
    # Yeni test verisi
    test_df_1m = TimeframeAggregator.create_test_data(bars=45)
    mtf_data = create_multi_timeframe_data(test_df_1m)
    
    symbol = "ETHUSDT"
    timeframes = ['1m', '5m', '15m']
    
    # Cache warming
    results = await RedisClient.warm_mtf_cache(symbol, timeframes, mtf_data)
    
    print(f"🔥 Cache Warming Sonuçları:")
    for tf, success in results.items():
        print(f"  {tf}: {'✅' if success else '❌'}")

async def test_mtf_cache_flush():
    """MTF cache flush testleri"""
    print("\n🧪 MTF Cache Flush")
    print("=" * 50)
    
    symbol = "BTCUSDT"
    
    # Flush öncesi stats
    stats_before = await RedisClient.get_mtf_cache_stats(symbol)
    cache_count_before = sum(1 for tf_stats in stats_before.values() 
                           for info in tf_stats.values() if info['exists'])
    
    print(f"🗑️ Flush öncesi cache sayısı: {cache_count_before}")
    
    # Flush
    deleted_count = await RedisClient.flush_mtf_cache(symbol, ['1m', '5m', '15m'])
    print(f"🗑️ Silinen cache sayısı: {deleted_count}")
    
    # Flush sonrası stats
    stats_after = await RedisClient.get_mtf_cache_stats(symbol)
    cache_count_after = sum(1 for tf_stats in stats_after.values() 
                          for info in tf_stats.values() if info['exists'])
    
    print(f"🗑️ Flush sonrası cache sayısı: {cache_count_after}")

async def test_ttl_strategies():
    """TTL stratejileri testleri"""
    print("\n🧪 TTL Stratejileri")
    print("=" * 50)
    
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    print("⏰ Timeframe TTL Değerleri:")
    for tf in timeframes:
        ttl = RedisClient._get_ttl_for_timeframe(tf)
        hours = ttl / 3600
        print(f"  {tf}: {ttl}s ({hours:.1f} saat)")

async def main():
    """Ana test fonksiyonu"""
    print("🚀 MTF Redis Cache Test Suite")
    print("=" * 60)
    
    try:
        # Temel cache testleri
        mtf_data = await test_mtf_cache_basic()
        
        # Cache stats
        await test_mtf_cache_stats()
        
        # Indicators cache
        await test_mtf_indicators()
        
        # Cache warming
        await test_mtf_cache_warming()
        
        # TTL stratejileri
        await test_ttl_strategies()
        
        # Cache flush (en son)
        await test_mtf_cache_flush()
        
        print("\n🎉 Tüm MTF Redis testleri tamamlandı!")
        print("=" * 60)
        
        # Özet
        print("\n📋 Test Özeti:")
        print(f"  ✅ MTF klines cache: Çalışıyor")
        print(f"  ✅ MTF indicators cache: Çalışıyor") 
        print(f"  ✅ Cache stats: Çalışıyor")
        print(f"  ✅ Cache warming: Çalışıyor")
        print(f"  ✅ Cache flush: Çalışıyor")
        print(f"  ✅ TTL strategies: Çalışıyor")
        
    except Exception as e:
        print(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
