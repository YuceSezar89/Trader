"""
Integration tests for Multi-Timeframe system.
Tests the complete pipeline from aggregation to cache to retrieval.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

from utils.timeframe_aggregator import TimeframeAggregator, create_multi_timeframe_data
from utils.redis_client import RedisClient
from tests.conftest import assert_dataframe_valid


class TestMTFIntegration:
    """Integration tests for complete MTF pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.mtf
    @pytest.mark.asyncio
    async def test_complete_mtf_pipeline(self, test_symbol, clean_redis_state):
        """Test complete MTF pipeline: create -> aggregate -> cache -> retrieve."""
        # Step 1: Create 1m data
        bars_1m = 60  # 1 hour of 1m data
        df_1m = TimeframeAggregator.create_test_data(bars=bars_1m)
        assert_dataframe_valid(df_1m)
        
        # Step 2: Create multi-timeframe data
        mtf_data = create_multi_timeframe_data(df_1m)
        
        expected_timeframes = ['1m', '5m']  # 15m might not have enough data
        if bars_1m >= 15:
            expected_timeframes.append('15m')
        
        for tf in expected_timeframes:
            assert tf in mtf_data, f"Should have {tf} data"
            assert_dataframe_valid(mtf_data[tf])
        
        # Step 3: Cache all timeframes
        cache_results = await RedisClient.warm_mtf_cache(test_symbol, expected_timeframes, mtf_data)
        
        for tf in expected_timeframes:
            assert cache_results[tf] == True, f"Should successfully cache {tf}"
        
        # Step 4: Retrieve and verify
        for tf in expected_timeframes:
            cached_df = await RedisClient.get_mtf_klines(test_symbol, tf)
            assert cached_df is not None, f"Should retrieve {tf} data"
            assert_dataframe_valid(cached_df)
            assert len(cached_df) == len(mtf_data[tf]), f"Length should match for {tf}"
        
        # Step 5: Verify cache stats
        stats = await RedisClient.get_mtf_cache_stats(test_symbol)
        for tf in expected_timeframes:
            assert stats[tf]['live_kline_data']['exists'], f"Cache should exist for {tf}"
            assert stats[tf]['live_kline_data']['ttl'] > 0, f"TTL should be positive for {tf}"
    
    @pytest.mark.integration
    @pytest.mark.mtf
    @pytest.mark.asyncio
    async def test_mtf_data_consistency(self, test_symbol, clean_redis_state):
        """Test data consistency across different timeframes."""
        # Create 1 hour of 1m data (60 bars)
        df_1m = TimeframeAggregator.create_test_data(bars=60)
        
        # Aggregate to different timeframes
        df_5m = TimeframeAggregator.aggregate_ohlcv(df_1m, '1m', '5m')  # Should be 12 bars
        df_15m = TimeframeAggregator.aggregate_ohlcv(df_1m, '1m', '15m')  # Should be 4 bars
        
        # Cache all data
        await RedisClient.set_mtf_klines(test_symbol, '1m', df_1m)
        await RedisClient.set_mtf_klines(test_symbol, '5m', df_5m)
        await RedisClient.set_mtf_klines(test_symbol, '15m', df_15m)
        
        # Retrieve and verify consistency
        cached_1m = await RedisClient.get_mtf_klines(test_symbol, '1m')
        cached_5m = await RedisClient.get_mtf_klines(test_symbol, '5m')
        cached_15m = await RedisClient.get_mtf_klines(test_symbol, '15m')
        
        # Verify volume consistency (1m total should equal 5m total should equal 15m total)
        total_volume_1m = cached_1m['volume'].sum()
        total_volume_5m = cached_5m['volume'].sum()
        total_volume_15m = cached_15m['volume'].sum()
        
        # Allow small floating point differences
        assert abs(total_volume_1m - total_volume_5m) < 0.01, "1m and 5m volume totals should match"
        assert abs(total_volume_1m - total_volume_15m) < 0.01, "1m and 15m volume totals should match"
        
        # Verify time consistency
        first_time_1m = cached_1m['open_time'].iloc[0]
        first_time_5m = cached_5m['open_time'].iloc[0]
        first_time_15m = cached_15m['open_time'].iloc[0]
        
        assert first_time_1m == first_time_5m == first_time_15m, "First timestamps should match"
    
    @pytest.mark.integration
    @pytest.mark.mtf
    @pytest.mark.asyncio
    async def test_mtf_cache_expiration(self, test_symbol, clean_redis_state):
        """Test MTF cache expiration behavior."""
        # Create test data
        df_1m = TimeframeAggregator.create_test_data(bars=10)
        
        # Cache with very short TTL for testing
        await RedisClient.set_mtf_klines(test_symbol, '1m', df_1m)
        
        # Verify data exists
        cached_df = await RedisClient.get_mtf_klines(test_symbol, '1m')
        assert cached_df is not None, "Data should be cached initially"
        
        # Check TTL
        stats = await RedisClient.get_mtf_cache_stats(test_symbol)
        initial_ttl = stats['1m']['live_kline_data']['ttl']
        assert initial_ttl > 0, "TTL should be positive"
        
        # Wait a bit and check TTL decreased
        await asyncio.sleep(2)
        stats = await RedisClient.get_mtf_cache_stats(test_symbol)
        new_ttl = stats['1m']['live_kline_data']['ttl']
        assert new_ttl < initial_ttl, "TTL should decrease over time"
    
    @pytest.mark.integration
    @pytest.mark.mtf
    @pytest.mark.asyncio
    async def test_mtf_multiple_symbols(self, clean_redis_state):
        """Test MTF system with multiple symbols."""
        symbols = ['TESTBTCUSDT', 'TESTETHUSDT', 'TESTADAUSDT']
        timeframes = ['1m', '5m']
        
        # Create and cache data for multiple symbols
        for symbol in symbols:
            df_1m = TimeframeAggregator.create_test_data(bars=30)
            mtf_data = create_multi_timeframe_data(df_1m)
            
            await RedisClient.warm_mtf_cache(symbol, timeframes, mtf_data)
        
        # Verify all symbols have cached data
        for symbol in symbols:
            for tf in timeframes:
                cached_df = await RedisClient.get_mtf_klines(symbol, tf)
                assert cached_df is not None, f"Should have cached data for {symbol}:{tf}"
                assert_dataframe_valid(cached_df)
        
        # Verify cache stats for all symbols
        for symbol in symbols:
            stats = await RedisClient.get_mtf_cache_stats(symbol)
            for tf in timeframes:
                assert stats[tf]['live_kline_data']['exists'], f"Cache should exist for {symbol}:{tf}"
    
    @pytest.mark.integration
    @pytest.mark.mtf
    @pytest.mark.asyncio
    async def test_mtf_indicators_and_signals_integration(self, test_symbol, clean_redis_state):
        """Test integration of klines, indicators, and signals."""
        # Create base data
        df_1m = TimeframeAggregator.create_test_data(bars=30)
        df_5m = TimeframeAggregator.aggregate_ohlcv(df_1m, '1m', '5m')
        
        # Cache klines
        await RedisClient.set_mtf_klines(test_symbol, '5m', df_5m)
        
        # Create mock indicators based on the data
        indicators = {
            'rsi': 65.5,
            'macd': 1.2,
            'ema_21': float(df_5m['close'].mean()),
            'volume_avg': float(df_5m['volume'].mean()),
            'price_range': float(df_5m['high'].max() - df_5m['low'].min()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache indicators
        await RedisClient.set_mtf_indicators(test_symbol, '5m', indicators)
        
        # Create mock signals
        signals = [
            {
                'signal_type': 'Long',
                'signal_strength': 2,
                'price': float(df_5m['close'].iloc[-1]),
                'timestamp': datetime.now().isoformat(),
                'reason': 'RSI oversold + MACD bullish cross',
                'timeframe': '5m'
            }
        ]
        
        # Cache signals
        await RedisClient.set_mtf_signals(test_symbol, '5m', signals)
        
        # Retrieve all data types
        cached_klines = await RedisClient.get_mtf_klines(test_symbol, '5m')
        cached_indicators = await RedisClient.get_mtf_indicators(test_symbol, '5m')
        cached_signals = await RedisClient.get_mtf_signals(test_symbol, '5m')
        
        # Verify all data exists and is consistent
        assert cached_klines is not None, "Klines should be cached"
        assert cached_indicators is not None, "Indicators should be cached"
        assert cached_signals is not None, "Signals should be cached"
        
        # Verify data relationships
        assert abs(cached_indicators['ema_21'] - df_5m['close'].mean()) < 0.01, "Indicator should match calculated value"
        assert abs(cached_signals[0]['price'] - df_5m['close'].iloc[-1]) < 0.01, "Signal price should match last close"
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mtf_performance_integration(self, test_symbol, clean_redis_state):
        """Test performance of complete MTF pipeline."""
        # Create larger dataset for performance testing
        df_1m = TimeframeAggregator.create_test_data(bars=300)  # 5 hours of data
        
        start_time = time.time()
        
        # Step 1: Aggregation
        aggregation_start = time.time()
        mtf_data = create_multi_timeframe_data(df_1m)
        aggregation_time = time.time() - aggregation_start
        
        # Step 2: Caching
        cache_start = time.time()
        timeframes = list(mtf_data.keys())
        cache_results = await RedisClient.warm_mtf_cache(test_symbol, timeframes, mtf_data)
        cache_time = time.time() - cache_start
        
        # Step 3: Retrieval
        retrieval_start = time.time()
        for tf in timeframes:
            cached_df = await RedisClient.get_mtf_klines(test_symbol, tf)
            assert cached_df is not None
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert aggregation_time < 0.5, f"Aggregation took {aggregation_time:.3f}s, should be < 0.5s"
        assert cache_time < 0.2, f"Caching took {cache_time:.3f}s, should be < 0.2s"
        assert retrieval_time < 0.1, f"Retrieval took {retrieval_time:.3f}s, should be < 0.1s"
        assert total_time < 1.0, f"Total pipeline took {total_time:.3f}s, should be < 1.0s"
        
        # Verify all operations succeeded
        assert all(cache_results.values()), "All cache operations should succeed"


class TestMTFErrorRecovery:
    """Test error recovery in MTF integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_partial_cache_failure_recovery(self, test_symbol, clean_redis_state):
        """Test recovery from partial cache failures."""
        # Create MTF data
        df_1m = TimeframeAggregator.create_test_data(bars=30)
        mtf_data = create_multi_timeframe_data(df_1m)
        
        # Simulate partial failure by caching only some timeframes
        successful_tfs = []
        failed_tfs = []
        
        for tf, df in mtf_data.items():
            if tf == '5m':
                # Simulate failure for 5m
                failed_tfs.append(tf)
                continue
            
            success = await RedisClient.set_mtf_klines(test_symbol, tf, df)
            if success:
                successful_tfs.append(tf)
            else:
                failed_tfs.append(tf)
        
        # Verify partial success
        for tf in successful_tfs:
            cached_df = await RedisClient.get_mtf_klines(test_symbol, tf)
            assert cached_df is not None, f"Should have cached data for {tf}"
        
        for tf in failed_tfs:
            cached_df = await RedisClient.get_mtf_klines(test_symbol, tf)
            assert cached_df is None, f"Should not have cached data for {tf}"
        
        # Retry failed timeframes
        for tf in failed_tfs:
            if tf in mtf_data:
                success = await RedisClient.set_mtf_klines(test_symbol, tf, mtf_data[tf])
                if success:
                    cached_df = await RedisClient.get_mtf_klines(test_symbol, tf)
                    assert cached_df is not None, f"Should have cached data for {tf} after retry"
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_data_corruption_recovery(self, test_symbol, clean_redis_state):
        """Test recovery from data corruption scenarios."""
        # Create and cache valid data
        df_1m = TimeframeAggregator.create_test_data(bars=15)
        await RedisClient.set_mtf_klines(test_symbol, '1m', df_1m)
        
        # Verify data is cached
        cached_df = await RedisClient.get_mtf_klines(test_symbol, '1m')
        assert cached_df is not None, "Data should be cached initially"
        
        # Simulate data corruption by manually inserting bad data
        r = RedisClient.get_client()
        try:
            key = RedisClient._get_mtf_key(test_symbol, '1m', 'live_kline_data')
            await r.set(key, "corrupted_data", ex=3600)
            
            # Try to retrieve corrupted data
            cached_df = await RedisClient.get_mtf_klines(test_symbol, '1m')
            assert cached_df is None, "Should return None for corrupted data"
            
            # Recovery: re-cache valid data
            await RedisClient.set_mtf_klines(test_symbol, '1m', df_1m)
            
            # Verify recovery
            cached_df = await RedisClient.get_mtf_klines(test_symbol, '1m')
            assert cached_df is not None, "Should recover with valid data"
            assert_dataframe_valid(cached_df)
            
        finally:
            await r.close()


class TestMTFScalability:
    """Test MTF system scalability."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_volume_mtf_operations(self, clean_redis_state):
        """Test MTF system with high volume of operations."""
        symbols = [f"TEST{i:03d}USDT" for i in range(10)]  # 10 test symbols
        timeframes = ['1m', '5m', '15m']
        
        # Create data for all symbols
        all_data = {}
        for symbol in symbols:
            df_1m = TimeframeAggregator.create_test_data(bars=60)
            all_data[symbol] = create_multi_timeframe_data(df_1m)
        
        # Concurrent caching operations
        cache_tasks = []
        for symbol, mtf_data in all_data.items():
            for tf, df in mtf_data.items():
                if tf in timeframes:
                    task = RedisClient.set_mtf_klines(symbol, tf, df)
                    cache_tasks.append(task)
        
        start_time = time.time()
        cache_results = await asyncio.gather(*cache_tasks)
        cache_time = time.time() - start_time
        
        # Verify all operations succeeded
        success_count = sum(1 for result in cache_results if result)
        total_operations = len(symbols) * len(timeframes)
        
        assert success_count >= total_operations * 0.9, f"At least 90% of operations should succeed"
        assert cache_time < 5.0, f"High volume caching took {cache_time:.2f}s, should be < 5s"
        
        # Concurrent retrieval operations
        retrieval_tasks = []
        for symbol in symbols:
            for tf in timeframes:
                task = RedisClient.get_mtf_klines(symbol, tf)
                retrieval_tasks.append(task)
        
        start_time = time.time()
        retrieval_results = await asyncio.gather(*retrieval_tasks)
        retrieval_time = time.time() - start_time
        
        # Verify retrievals
        successful_retrievals = sum(1 for result in retrieval_results if result is not None)
        assert successful_retrievals >= total_operations * 0.9, "At least 90% of retrievals should succeed"
        assert retrieval_time < 2.0, f"High volume retrieval took {retrieval_time:.2f}s, should be < 2s"
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_efficiency_mtf(self, test_symbol, clean_redis_state):
        """Test memory efficiency of MTF operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large amounts of data
        large_datasets = []
        for i in range(5):  # 5 large datasets
            df_1m = TimeframeAggregator.create_test_data(bars=1000)  # Large dataset
            mtf_data = create_multi_timeframe_data(df_1m)
            large_datasets.append(mtf_data)
            
            # Cache each dataset
            symbol = f"{test_symbol}_{i}"
            await RedisClient.warm_mtf_cache(symbol, list(mtf_data.keys()), mtf_data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up datasets from memory
        large_datasets.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory assertions
        assert memory_increase < 200, f"Memory increased by {memory_increase:.2f}MB, should be < 200MB"
        assert final_memory - initial_memory < 50, "Memory should be mostly freed after cleanup"
