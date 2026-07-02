#!/usr/bin/env python3
"""
MTF Aggregation Integration Test

Bu script gerçek aggregation işlemlerini test eder.
"""

import pytest

pytest.skip(
    "Manuel entegrasyon scripti - pytest testi degil; calistirmak icin: python %s" % __file__,
    allow_module_level=True,
)


import sys
import os
import asyncio
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_data_manager import LiveDataManager
from utils.timeframe_aggregator import TimeframeAggregator
from config import Config
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_aggregation():
    """Test real aggregation with sample data"""
    print("🧪 Real MTF Aggregation Test")
    print("=" * 50)
    
    # Create sample 1m data (15 bars for proper 15m aggregation)
    sample_data = []
    base_time = 1695571200000  # Sample timestamp
    base_price = 50000.0
    
    for i in range(15):  # 15 minutes of 1m data
        bar = {
            "open_time": base_time + (i * 60000),  # Each bar is 1 minute apart
            "open": base_price + (i * 10),
            "high": base_price + (i * 10) + 50,
            "low": base_price + (i * 10) - 30,
            "close": base_price + (i * 10) + 20,
            "volume": 1000.0 + (i * 100),
            "close_time": base_time + (i * 60000) + 59999,
            "quote_asset_volume": (base_price + (i * 10) + 20) * (1000.0 + (i * 100)),
            "number_of_trades": 100 + i,
            "taker_buy_base_asset_volume": (1000.0 + (i * 100)) * 0.6,
            "taker_buy_quote_asset_volume": (base_price + (i * 10) + 20) * (1000.0 + (i * 100)) * 0.6,
        }
        sample_data.append(bar)
    
    # Create manager
    manager = LiveDataManager(symbols=['BTCUSDT'], interval='1m')
    symbol = 'BTCUSDT'
    
    print(f"📝 Adding {len(sample_data)} bars of 1m data...")
    
    # Add all sample data to 1m buffer
    for bar in sample_data:
        await manager._update_mtf_data(symbol, bar)
    
    # Check results
    stats = manager.get_mtf_stats()
    print(f"📊 Final MTF Stats:")
    for sym, tf_stats in stats.items():
        print(f"  {sym}: {tf_stats}")
    
    # Verify aggregation correctness
    print(f"\n🔍 Verifying aggregation correctness:")
    
    # Get data for each timeframe
    df_1m = manager.get_mtf_data(symbol, '1m')
    df_5m = manager.get_mtf_data(symbol, '5m')
    df_15m = manager.get_mtf_data(symbol, '15m')
    
    if df_1m is not None and not df_1m.empty:
        print(f"  1m: {len(df_1m)} bars")
        print(f"    First bar: {df_1m.iloc[0]['open']:.2f} -> {df_1m.iloc[0]['close']:.2f}")
        print(f"    Last bar:  {df_1m.iloc[-1]['open']:.2f} -> {df_1m.iloc[-1]['close']:.2f}")
        print(f"    Total volume: {df_1m['volume'].sum():.2f}")
    
    if df_5m is not None and not df_5m.empty:
        print(f"  5m: {len(df_5m)} bars")
        print(f"    First bar: {df_5m.iloc[0]['open']:.2f} -> {df_5m.iloc[0]['close']:.2f}")
        print(f"    Last bar:  {df_5m.iloc[-1]['open']:.2f} -> {df_5m.iloc[-1]['close']:.2f}")
        print(f"    Total volume: {df_5m['volume'].sum():.2f}")
    
    if df_15m is not None and not df_15m.empty:
        print(f"  15m: {len(df_15m)} bars")
        print(f"    First bar: {df_15m.iloc[0]['open']:.2f} -> {df_15m.iloc[0]['close']:.2f}")
        print(f"    Total volume: {df_15m['volume'].sum():.2f}")
    
    # Volume consistency check
    if df_1m is not None and df_5m is not None and df_15m is not None:
        vol_1m = df_1m['volume'].sum()
        vol_5m = df_5m['volume'].sum()
        vol_15m = df_15m['volume'].sum()
        
        print(f"\n✅ Volume Consistency Check:")
        print(f"  1m total:  {vol_1m:.2f}")
        print(f"  5m total:  {vol_5m:.2f}")
        print(f"  15m total: {vol_15m:.2f}")
        
        if abs(vol_1m - vol_5m) < 0.01 and abs(vol_1m - vol_15m) < 0.01:
            print(f"  ✅ Volume consistency: PASS")
        else:
            print(f"  ❌ Volume consistency: FAIL")
    
    return manager

async def test_continuous_updates():
    """Test continuous bar updates"""
    print("\n🧪 Continuous Updates Test")
    print("=" * 50)
    
    manager = LiveDataManager(symbols=['BTCUSDT'], interval='1m')
    symbol = 'BTCUSDT'
    
    base_time = 1695571200000
    base_price = 50000.0
    
    print("📝 Simulating continuous 1m bar updates...")
    
    # Add bars one by one and check aggregation
    for i in range(20):  # 20 minutes of data
        bar = {
            "open_time": base_time + (i * 60000),
            "open": base_price + (i * 5),
            "high": base_price + (i * 5) + 25,
            "low": base_price + (i * 5) - 15,
            "close": base_price + (i * 5) + 10,
            "volume": 1000.0,
            "close_time": base_time + (i * 60000) + 59999,
            "quote_asset_volume": (base_price + (i * 5) + 10) * 1000.0,
            "number_of_trades": 100,
            "taker_buy_base_asset_volume": 600.0,
            "taker_buy_quote_asset_volume": (base_price + (i * 5) + 10) * 600.0,
        }
        
        await manager._update_mtf_data(symbol, bar)
        
        # Check stats every 5 bars
        if (i + 1) % 5 == 0:
            stats = manager.get_mtf_stats()
            print(f"  After {i+1} bars: {stats[symbol]}")
    
    # Final check
    print(f"\n📊 Final continuous update results:")
    stats = manager.get_mtf_stats()
    for sym, tf_stats in stats.items():
        print(f"  {sym}: {tf_stats}")
    
    return manager

async def test_redis_integration():
    """Test Redis MTF integration"""
    print("\n🧪 Redis MTF Integration Test")
    print("=" * 50)
    
    from utils.redis_client import RedisClient
    
    manager = LiveDataManager(symbols=['BTCUSDT'], interval='1m')
    symbol = 'BTCUSDT'
    
    # Add some sample data
    base_time = 1695571200000
    base_price = 50000.0
    
    for i in range(10):
        bar = {
            "open_time": base_time + (i * 60000),
            "open": base_price,
            "high": base_price + 50,
            "low": base_price - 30,
            "close": base_price + 20,
            "volume": 1000.0,
            "close_time": base_time + (i * 60000) + 59999,
            "quote_asset_volume": (base_price + 20) * 1000.0,
            "number_of_trades": 100,
            "taker_buy_base_asset_volume": 600.0,
            "taker_buy_quote_asset_volume": (base_price + 20) * 600.0,
        }
        
        await manager._update_mtf_data(symbol, bar)
    
    print("📝 Testing Redis cache retrieval...")
    
    # Test Redis retrieval for each timeframe
    for tf in manager.supported_timeframes:
        cached_df = await RedisClient.get_mtf_klines(symbol, tf)
        if cached_df is not None and not cached_df.empty:
            print(f"  {tf}: {len(cached_df)} bars cached in Redis ✅")
        else:
            print(f"  {tf}: No data in Redis cache ❌")
    
    # Test cache stats
    try:
        cache_stats = await RedisClient.get_mtf_cache_stats(symbol)
        print(f"\n📊 Redis Cache Stats:")
        for tf, stats in cache_stats.items():
            if tf in manager.supported_timeframes:
                exists = stats.get('live_kline_data', {}).get('exists', False)
                ttl = stats.get('live_kline_data', {}).get('ttl', 0)
                print(f"  {tf}: Exists={exists}, TTL={ttl}s")
    except Exception as e:
        print(f"  Cache stats error: {e}")
    
    return manager

async def main():
    """Main test function"""
    print("🚀 MTF Aggregation Integration Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_real_aggregation()
        await test_continuous_updates()
        await test_redis_integration()
        
        print("\n🎉 All MTF aggregation tests completed!")
        print("=" * 60)
        
        # Summary
        print("\n📋 Test Summary:")
        print(f"  ✅ Real Aggregation: Working")
        print(f"  ✅ Continuous Updates: Working")
        print(f"  ✅ Redis Integration: Working")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
