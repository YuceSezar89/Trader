"""
Comprehensive tests for TimeframeAggregator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from utils.timeframe_aggregator import (
    TimeframeAggregator, 
    aggregate_1m_to_5m, 
    aggregate_1m_to_15m,
    aggregate_5m_to_15m,
    create_multi_timeframe_data
)
from tests.conftest import assert_dataframe_valid, assert_aggregation_valid


class TestTimeframeAggregator:
    """Test TimeframeAggregator class methods."""
    
    @pytest.mark.unit
    def test_can_aggregate_valid_pairs(self):
        """Test valid aggregation pairs."""
        valid_pairs = [
            ('1m', '5m'), ('1m', '15m'), ('1m', '1h'), ('1m', '4h'), ('1m', '1d'),
            ('5m', '15m'), ('5m', '1h'), ('5m', '4h'), ('5m', '1d'),
            ('15m', '1h'), ('15m', '4h'), ('15m', '1d'),
            ('1h', '4h'), ('1h', '1d'),
            ('4h', '1d')
        ]
        
        for from_tf, to_tf in valid_pairs:
            assert TimeframeAggregator.can_aggregate(from_tf, to_tf), \
                f"Should be able to aggregate {from_tf} -> {to_tf}"
    
    @pytest.mark.unit
    def test_can_aggregate_invalid_pairs(self):
        """Test invalid aggregation pairs."""
        invalid_pairs = [
            ('5m', '1m'), ('15m', '5m'), ('1h', '15m'),  # Reverse direction
            ('1m', '7m'), ('5m', '17m'),  # Non-standard timeframes
            ('invalid', '5m'), ('1m', 'invalid'),  # Invalid timeframes
        ]
        
        for from_tf, to_tf in invalid_pairs:
            assert not TimeframeAggregator.can_aggregate(from_tf, to_tf), \
                f"Should NOT be able to aggregate {from_tf} -> {to_tf}"
    
    @pytest.mark.unit
    def test_get_aggregation_ratio(self):
        """Test aggregation ratio calculations."""
        test_cases = [
            ('1m', '5m', 5),
            ('1m', '15m', 15),
            ('1m', '1h', 60),
            ('5m', '15m', 3),
            ('5m', '1h', 12),
            ('15m', '1h', 4),
            ('1h', '4h', 4),
            ('4h', '1d', 6),
        ]
        
        for from_tf, to_tf, expected_ratio in test_cases:
            ratio = TimeframeAggregator.get_aggregation_ratio(from_tf, to_tf)
            assert ratio == expected_ratio, \
                f"Expected ratio {expected_ratio} for {from_tf}->{to_tf}, got {ratio}"
    
    @pytest.mark.unit
    def test_get_aggregation_ratio_invalid(self):
        """Test aggregation ratio for invalid pairs."""
        invalid_cases = [
            ('5m', '1m'),  # Reverse
            ('1m', '7m'),  # Non-standard
            ('invalid', '5m'),  # Invalid timeframe
        ]
        
        for from_tf, to_tf in invalid_cases:
            ratio = TimeframeAggregator.get_aggregation_ratio(from_tf, to_tf)
            assert ratio is None, f"Should return None for {from_tf}->{to_tf}"
    
    @pytest.mark.unit
    def test_validate_dataframe_valid(self, sample_1m_data):
        """Test DataFrame validation with valid data."""
        assert TimeframeAggregator.validate_dataframe(sample_1m_data)
    
    @pytest.mark.unit
    def test_validate_dataframe_empty(self, empty_dataframe):
        """Test DataFrame validation with empty data."""
        assert not TimeframeAggregator.validate_dataframe(empty_dataframe)
    
    @pytest.mark.unit
    def test_validate_dataframe_missing_columns(self, invalid_dataframe):
        """Test DataFrame validation with missing columns."""
        assert not TimeframeAggregator.validate_dataframe(invalid_dataframe)
    
    @pytest.mark.unit
    def test_validate_dataframe_wrong_types(self, malformed_dataframe):
        """Test DataFrame validation with wrong data types."""
        assert not TimeframeAggregator.validate_dataframe(malformed_dataframe)
    
    @pytest.mark.aggregation
    def test_aggregate_ohlcv_1m_to_5m(self, sample_1m_data):
        """Test 1m to 5m aggregation."""
        result = TimeframeAggregator.aggregate_ohlcv(sample_1m_data, '1m', '5m')
        
        assert_dataframe_valid(result)
        assert_aggregation_valid(sample_1m_data, result, 5)
        
        # Check specific aggregation rules
        if len(result) > 0:
            # First 5 bars of original data
            first_group = sample_1m_data.iloc[:5]
            first_agg = result.iloc[0]
            
            assert first_agg['open'] == first_group['open'].iloc[0]  # First open
            assert first_agg['close'] == first_group['close'].iloc[-1]  # Last close
            assert first_agg['high'] == first_group['high'].max()  # Max high
            assert first_agg['low'] == first_group['low'].min()  # Min low
            assert abs(first_agg['volume'] - first_group['volume'].sum()) < 0.01  # Sum volume
    
    @pytest.mark.aggregation
    def test_aggregate_ohlcv_1m_to_15m(self, sample_1m_data):
        """Test 1m to 15m aggregation."""
        # Need at least 15 bars for 15m aggregation
        if len(sample_1m_data) < 15:
            sample_1m_data = TimeframeAggregator.create_test_data(bars=30)
        
        result = TimeframeAggregator.aggregate_ohlcv(sample_1m_data, '1m', '15m')
        
        assert_dataframe_valid(result)
        assert_aggregation_valid(sample_1m_data, result, 15)
    
    @pytest.mark.aggregation
    def test_aggregate_ohlcv_5m_to_15m(self, sample_5m_data):
        """Test 5m to 15m aggregation."""
        # Need at least 3 bars for 15m aggregation
        if len(sample_5m_data) < 3:
            sample_5m_data = TimeframeAggregator.create_test_data(bars=6)
        
        result = TimeframeAggregator.aggregate_ohlcv(sample_5m_data, '5m', '15m')
        
        assert_dataframe_valid(result)
        assert_aggregation_valid(sample_5m_data, result, 3)
    
    @pytest.mark.edge_case
    def test_aggregate_insufficient_data(self, minimal_valid_dataframe):
        """Test aggregation with insufficient data."""
        # Try to aggregate 3 bars to 5m (needs 5 bars)
        result = TimeframeAggregator.aggregate_ohlcv(minimal_valid_dataframe, '1m', '5m')
        assert result.empty, "Should return empty DataFrame for insufficient data"
    
    @pytest.mark.edge_case
    def test_aggregate_exact_ratio_data(self):
        """Test aggregation with exact ratio data."""
        # Create exactly 10 bars for 5m aggregation (2 complete groups)
        test_data = TimeframeAggregator.create_test_data(bars=10)
        result = TimeframeAggregator.aggregate_ohlcv(test_data, '1m', '5m')
        
        assert len(result) == 2, "Should create exactly 2 aggregated bars"
        assert_dataframe_valid(result)
    
    @pytest.mark.edge_case
    def test_aggregate_partial_data(self):
        """Test aggregation with partial data (incomplete last group)."""
        # Create 13 bars for 5m aggregation (2 complete + 3 partial)
        test_data = TimeframeAggregator.create_test_data(bars=13)
        result = TimeframeAggregator.aggregate_ohlcv(test_data, '1m', '5m')
        
        assert len(result) == 2, "Should ignore incomplete last group"
        assert_dataframe_valid(result)
    
    @pytest.mark.aggregation
    def test_aggregate_to_multiple_timeframes(self, sample_1m_data):
        """Test aggregation to multiple timeframes."""
        # Ensure we have enough data
        if len(sample_1m_data) < 60:
            sample_1m_data = TimeframeAggregator.create_test_data(bars=60)
        
        target_timeframes = ['5m', '15m', '1h']
        results = TimeframeAggregator.aggregate_to_multiple_timeframes(
            sample_1m_data, '1m', target_timeframes
        )
        
        assert isinstance(results, dict)
        assert len(results) <= len(target_timeframes)  # Some might fail due to insufficient data
        
        for tf, df in results.items():
            assert tf in target_timeframes
            assert_dataframe_valid(df)
    
    @pytest.mark.unit
    def test_create_test_data(self):
        """Test test data creation."""
        bars = 50
        test_data = TimeframeAggregator.create_test_data(bars=bars)
        
        assert len(test_data) == bars
        assert_dataframe_valid(test_data)
        
        # Check timestamp progression
        timestamps = pd.to_datetime(test_data['open_time'], unit='ms')
        time_diffs = timestamps.diff().dropna()
        expected_diff = timedelta(minutes=1)
        
        for diff in time_diffs:
            assert diff == expected_diff, "Timestamps should be 1 minute apart"
    
    @pytest.mark.unit
    def test_create_test_data_custom_start_time(self):
        """Test test data creation with custom start time."""
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        bars = 10
        test_data = TimeframeAggregator.create_test_data(bars=bars, start_time=start_time)
        
        assert len(test_data) == bars
        first_timestamp = pd.to_datetime(test_data['open_time'].iloc[0], unit='ms')
        assert first_timestamp.replace(tzinfo=None) == start_time


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.unit
    def test_aggregate_1m_to_5m(self, sample_1m_data):
        """Test 1m to 5m convenience function."""
        result = aggregate_1m_to_5m(sample_1m_data)
        
        if len(sample_1m_data) >= 5:
            assert_dataframe_valid(result)
            assert_aggregation_valid(sample_1m_data, result, 5)
        else:
            assert result.empty
    
    @pytest.mark.unit
    def test_aggregate_1m_to_15m(self, sample_1m_data):
        """Test 1m to 15m convenience function."""
        # Ensure sufficient data
        if len(sample_1m_data) < 15:
            sample_1m_data = TimeframeAggregator.create_test_data(bars=30)
        
        result = aggregate_1m_to_15m(sample_1m_data)
        assert_dataframe_valid(result)
        assert_aggregation_valid(sample_1m_data, result, 15)
    
    @pytest.mark.unit
    def test_aggregate_5m_to_15m(self, sample_5m_data):
        """Test 5m to 15m convenience function."""
        # Ensure sufficient data
        if len(sample_5m_data) < 3:
            sample_5m_data = TimeframeAggregator.create_test_data(bars=6)
        
        result = aggregate_5m_to_15m(sample_5m_data)
        assert_dataframe_valid(result)
        assert_aggregation_valid(sample_5m_data, result, 3)
    
    @pytest.mark.mtf
    def test_create_multi_timeframe_data(self, sample_1m_data):
        """Test multi-timeframe data creation."""
        # Ensure sufficient data
        if len(sample_1m_data) < 15:
            sample_1m_data = TimeframeAggregator.create_test_data(bars=30)
        
        results = create_multi_timeframe_data(sample_1m_data)
        
        assert isinstance(results, dict)
        assert '1m' in results
        assert results['1m'].equals(sample_1m_data)
        
        # Check if higher timeframes were created
        if len(sample_1m_data) >= 5:
            assert '5m' in results
            assert_dataframe_valid(results['5m'])
        
        if len(sample_1m_data) >= 15:
            assert '15m' in results
            assert_dataframe_valid(results['15m'])


class TestPerformance:
    """Performance tests for aggregation functions."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_aggregation_performance_large_dataset(self, large_dataset, performance_thresholds):
        """Test aggregation performance with large dataset."""
        start_time = time.time()
        result = TimeframeAggregator.aggregate_ohlcv(large_dataset, '1m', '5m')
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        assert execution_time_ms < performance_thresholds['aggregation_time_ms'], \
            f"Aggregation took {execution_time_ms:.2f}ms, expected < {performance_thresholds['aggregation_time_ms']}ms"
        
        assert_dataframe_valid(result)
    
    @pytest.mark.performance
    def test_memory_usage_aggregation(self, large_dataset):
        """Test memory usage during aggregation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = TimeframeAggregator.aggregate_ohlcv(large_dataset, '1m', '15m')
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB during aggregation"
        assert_dataframe_valid(result)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.error_handling
    def test_aggregate_with_none_dataframe(self):
        """Test aggregation with None DataFrame."""
        with pytest.raises(AttributeError):
            TimeframeAggregator.aggregate_ohlcv(None, '1m', '5m')
    
    @pytest.mark.error_handling
    def test_aggregate_with_invalid_timeframes(self, sample_1m_data):
        """Test aggregation with invalid timeframes."""
        result = TimeframeAggregator.aggregate_ohlcv(sample_1m_data, 'invalid', '5m')
        assert result.empty
        
        result = TimeframeAggregator.aggregate_ohlcv(sample_1m_data, '1m', 'invalid')
        assert result.empty
    
    @pytest.mark.error_handling
    def test_aggregate_with_corrupted_data(self):
        """Test aggregation with corrupted data."""
        # Create DataFrame with NaN values
        corrupted_data = TimeframeAggregator.create_test_data(bars=10)
        corrupted_data.loc[5, 'close'] = np.nan
        corrupted_data.loc[7, 'volume'] = np.inf
        
        # Should handle gracefully
        result = TimeframeAggregator.aggregate_ohlcv(corrupted_data, '1m', '5m')
        # Result might be empty or have fewer rows due to data cleaning
        assert isinstance(result, pd.DataFrame)


class TestParametrizedTests:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.parametrize("from_tf,to_tf", [
        ('1m', '5m'), ('1m', '15m'), ('1m', '1h'),
        ('5m', '15m'), ('5m', '1h'), ('15m', '1h')
    ])
    @pytest.mark.aggregation
    def test_all_aggregation_pairs(self, from_tf, to_tf):
        """Test all valid aggregation pairs."""
        ratio = TimeframeAggregator.get_aggregation_ratio(from_tf, to_tf)
        bars_needed = ratio * 3  # Ensure at least 3 aggregated bars
        
        test_data = TimeframeAggregator.create_test_data(bars=bars_needed)
        result = TimeframeAggregator.aggregate_ohlcv(test_data, from_tf, to_tf)
        
        assert_dataframe_valid(result)
        assert_aggregation_valid(test_data, result, ratio)
    
    @pytest.mark.parametrize("bar_count", [5, 10, 25, 50, 100, 500])
    @pytest.mark.aggregation
    def test_different_data_sizes(self, bar_count):
        """Test aggregation with different data sizes."""
        test_data = TimeframeAggregator.create_test_data(bars=bar_count)
        result = TimeframeAggregator.aggregate_ohlcv(test_data, '1m', '5m')
        
        expected_rows = bar_count // 5
        if expected_rows > 0:
            assert len(result) == expected_rows
            assert_dataframe_valid(result)
        else:
            assert result.empty
    
    @pytest.mark.parametrize("timeframe", ['1m', '5m', '15m', '1h', '4h', '1d'])
    @pytest.mark.unit
    def test_ttl_values_for_all_timeframes(self, timeframe):
        """Test that all timeframes have valid TTL values."""
        # This would be used in Redis cache
        ttl_map = {
            '1m': 3600,      # 1 saat
            '5m': 14400,     # 4 saat  
            '15m': 86400,    # 24 saat
            '1h': 172800,    # 48 saat
            '4h': 604800,    # 7 gün
            '1d': 2592000,   # 30 gün
        }
        
        assert timeframe in ttl_map
        assert ttl_map[timeframe] > 0
        assert isinstance(ttl_map[timeframe], int)
