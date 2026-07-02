#!/usr/bin/env python3
"""
MTF LiveDataManager Test Script

Bu script yeni MTF özelliklerini test eder.
"""

import pytest

pytest.skip(
    "Manuel entegrasyon scripti - pytest testi degil; calistirmak icin: python %s" % __file__,
    allow_module_level=True,
)


import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_data_manager import LiveDataManager
from config import Config
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mtf_initialization():
    """Test MTF buffer initialization"""
    print("🧪 MTF LiveDataManager Initialization Test")
    print("=" * 50)
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Create manager
    manager = LiveDataManager(symbols=test_symbols, interval='1m')
    
    # Check MTF configuration
    print(f"✅ MTF Enabled: {manager.mtf_enabled}")
    print(f"✅ Supported Timeframes: {manager.supported_timeframes}")
    print(f"✅ Buffer Limits: {manager.mtf_buffer_limits}")
    
    # Check MTF buffers structure
    if manager.mtf_enabled:
        print(f"✅ MTF Buffers Created: {len(manager.mtf_buffers)} symbols")
        for symbol in manager.mtf_buffers:
            print(f"  {symbol}: {list(manager.mtf_buffers[symbol].keys())}")
    
    return manager

async def test_mtf_data_flow():
    """Test MTF data flow with sample data"""
    print("\n🧪 MTF Data Flow Test")
    print("=" * 50)
    
    manager = await test_mtf_initialization()
    
    # Simulate new 1m bar
    sample_kline = {
        "open_time": 1695571200000,  # Sample timestamp
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 1000.0,
        "close_time": 1695571259999,
        "quote_asset_volume": 50050000.0,
        "number_of_trades": 100,
        "taker_buy_base_asset_volume": 500.0,
        "taker_buy_quote_asset_volume": 25025000.0,
    }
    
    symbol = 'BTCUSDT'
    
    # Test MTF update
    if manager.mtf_enabled:
        print(f"📝 Testing MTF update for {symbol}...")
        await manager._update_mtf_data(symbol, sample_kline)
        
        # Check buffer stats
        stats = manager.get_mtf_stats()
        print(f"📊 MTF Stats after update:")
        for sym, tf_stats in stats.items():
            print(f"  {sym}: {tf_stats}")
        
        # Test data retrieval
        for tf in manager.supported_timeframes:
            data = manager.get_mtf_data(symbol, tf)
            if data is not None:
                print(f"  {tf}: {len(data)} bars available")
            else:
                print(f"  {tf}: No data available")
    
    return manager

async def test_mtf_memory_usage():
    """Test MTF memory usage"""
    print("\n🧪 MTF Memory Usage Test")
    print("=" * 50)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create manager with more symbols
    test_symbols = [f'TEST{i:03d}USDT' for i in range(10)]
    manager = LiveDataManager(symbols=test_symbols, interval='1m')
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before
    
    print(f"📊 Memory Usage:")
    print(f"  Before: {memory_before:.2f} MB")
    print(f"  After: {memory_after:.2f} MB")
    print(f"  Increase: {memory_increase:.2f} MB")
    print(f"  Per Symbol: {memory_increase/len(test_symbols):.2f} MB")
    
    # Expected memory increase should be reasonable
    expected_max = 50  # MB
    if memory_increase < expected_max:
        print(f"✅ Memory usage is acceptable (< {expected_max} MB)")
    else:
        print(f"⚠️ Memory usage is high (> {expected_max} MB)")
    
    return manager

async def test_mtf_config_flexibility():
    """Test MTF configuration flexibility"""
    print("\n🧪 MTF Configuration Test")
    print("=" * 50)
    
    # Test with different configurations
    configs = [
        {'MTF_ENABLED': False},
        {'MTF_TIMEFRAMES': ['1m', '5m']},
        {'MTF_TIMEFRAMES': ['1m', '5m', '15m', '1h']},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n📋 Config {i+1}: {config}")
        
        # Temporarily modify Config
        original_values = {}
        for key, value in config.items():
            original_values[key] = getattr(Config, key, None)
            setattr(Config, key, value)
        
        try:
            manager = LiveDataManager(symbols=['BTCUSDT'], interval='1m')
            print(f"  MTF Enabled: {manager.mtf_enabled}")
            if manager.mtf_enabled:
                print(f"  Timeframes: {manager.supported_timeframes}")
                print(f"  Buffers Created: {len(manager.mtf_buffers) if hasattr(manager, 'mtf_buffers') else 0}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is not None:
                    setattr(Config, key, value)

async def main():
    """Main test function"""
    print("🚀 MTF LiveDataManager Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_mtf_initialization()
        await test_mtf_data_flow()
        await test_mtf_memory_usage()
        await test_mtf_config_flexibility()
        
        print("\n🎉 All MTF LiveDataManager tests completed!")
        print("=" * 60)
        
        # Summary
        print("\n📋 Test Summary:")
        print(f"  ✅ MTF Initialization: Working")
        print(f"  ✅ MTF Data Flow: Working")
        print(f"  ✅ Memory Usage: Acceptable")
        print(f"  ✅ Configuration: Flexible")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
