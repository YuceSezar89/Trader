"""
Pytest configuration and fixtures for TRader Panel tests.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os
import psycopg2
import psycopg2.extras

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.timeframe_aggregator import TimeframeAggregator
from utils.redis_client import RedisClient
from config import Config


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def db_connection():
    """Create database connection for tests."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )
    conn.autocommit = False  # Manual transaction control
    
    yield conn
    
    # Cleanup: rollback any uncommitted changes
    conn.rollback()
    conn.close()


@pytest.fixture(autouse=True)
def _db_rollback_guard(request):
    """Başarısız statement paylaşılan bağlantıyı zehirlemesin — her testten sonra rollback."""
    yield
    if "db_connection" in request.fixturenames:
        try:
            request.getfixturevalue("db_connection").rollback()
        except Exception:
            pass


@pytest.fixture(scope="function")
def db_cursor(db_connection):
    """Create database cursor for each test."""
    cursor = db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    yield cursor
    
    # Cleanup: rollback test changes
    db_connection.rollback()
    cursor.close()


@pytest.fixture
def sample_1m_data():
    """Generate sample 1m OHLCV data for testing."""
    return TimeframeAggregator.create_test_data(bars=60)  # 1 hour of 1m data


@pytest.fixture
def sample_5m_data():
    """Generate sample 5m OHLCV data for testing."""
    return TimeframeAggregator.create_test_data(bars=12)  # 1 hour of 5m data


@pytest.fixture
def sample_15m_data():
    """Generate sample 15m OHLCV data for testing."""
    return TimeframeAggregator.create_test_data(bars=4)  # 1 hour of 15m data


@pytest.fixture
def empty_dataframe():
    """Return an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def invalid_dataframe():
    """Return a DataFrame with missing required columns."""
    return pd.DataFrame({
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        # Missing: low, close, volume, open_time
    })


@pytest.fixture
def malformed_dataframe():
    """Return a DataFrame with incorrect data types."""
    return pd.DataFrame({
        'open_time': ['invalid', 'timestamps', 'here'],
        'open': ['not', 'numeric', 'data'],
        'high': [1, 2, 3],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [100, 200, 300]
    })


@pytest.fixture
def minimal_valid_dataframe():
    """Return minimal valid DataFrame for edge case testing."""
    base_time = datetime.now()
    return pd.DataFrame({
        'open_time': [int((base_time + timedelta(minutes=i)).timestamp() * 1000) for i in range(3)],
        'open': [100.0, 101.0, 102.0],
        'high': [101.0, 102.0, 103.0],
        'low': [99.0, 100.0, 101.0],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000.0, 1100.0, 1200.0]
    })


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    return TimeframeAggregator.create_test_data(bars=10000)  # Large dataset


@pytest.fixture
def test_symbol():
    """Return test symbol (TEST öneki: canlı backend'in yazdığı key'lerle çakışmaz)."""
    return "TESTBTCUSDT"


@pytest.fixture
def test_timeframes():
    """Return list of test timeframes."""
    return ['1m', '5m', '15m', '1h', '4h', '1d']


@pytest.fixture
def redis_test_keys():
    """Return list of Redis keys used in tests for cleanup."""
    return []


@pytest_asyncio.fixture
async def cleanup_redis(redis_test_keys):
    """Cleanup Redis keys after each test."""
    yield
    # Cleanup after test
    if redis_test_keys:
        r = RedisClient.get_client()
        try:
            for key in redis_test_keys:
                await r.delete(key)
        except Exception:
            pass  # Ignore cleanup errors
        finally:
            await r.close()


@pytest.fixture
def mock_indicators():
    """Return mock indicator data."""
    return {
        'rsi': 65.5,
        'macd': 1.2,
        'macd_signal': 0.8,
        'macd_histogram': 0.4,
        'ema_21': 50125.5,
        'sma_50': 49800.2,
        'bb_upper': 51000.0,
        'bb_lower': 49000.0,
        'volume_sma': 1250000.0,
        'atr': 150.5,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def mock_signals():
    """Return mock signal data."""
    return [
        {
            'signal_type': 'Long',
            'signal_strength': 2,
            'price': 50000.0,
            'timestamp': datetime.now().isoformat(),
            'reason': 'RSI oversold + MACD bullish cross'
        },
        {
            'signal_type': 'Short',
            'signal_strength': 1,
            'price': 51000.0,
            'timestamp': (datetime.now() + timedelta(minutes=5)).isoformat(),
            'reason': 'RSI overbought'
        }
    ]


@pytest.fixture
def performance_thresholds():
    """Return performance test thresholds."""
    return {
        'aggregation_time_ms': 2000,  # Regresyon eşiği; yüklü makinede flake yapmasın
        'cache_write_time_ms': 100,  # Max 100ms for cache write
        'cache_read_time_ms': 50,    # Max 50ms for cache read
        'memory_usage_mb': 100,      # Max 100MB memory usage
    }


@pytest.fixture
def error_scenarios():
    """Return various error scenarios for testing."""
    return {
        'network_error': Exception("Network connection failed"),
        'redis_error': Exception("Redis connection timeout"),
        'data_corruption': Exception("Data integrity check failed"),
        'memory_error': MemoryError("Out of memory"),
        'value_error': ValueError("Invalid input parameters"),
        'type_error': TypeError("Incorrect data type"),
    }


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=['1m', '5m', '15m', '1h', '4h', '1d'])
def timeframe(request):
    """Parametrized timeframe fixture."""
    return request.param


@pytest.fixture(params=[5, 10, 25, 50, 100, 500, 1000])
def bar_counts(request):
    """Parametrized bar count fixture."""
    return request.param


@pytest.fixture(params=[
    ('1m', '5m'), ('1m', '15m'), ('1m', '1h'),
    ('5m', '15m'), ('5m', '1h'), ('15m', '1h')
])
def aggregation_pairs(request):
    """Parametrized aggregation pair fixture."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 4, 5, 10, 15, 20])
def aggregation_ratios(request):
    """Parametrized aggregation ratio fixture."""
    return request.param


# Async fixtures
@pytest_asyncio.fixture
async def redis_client():
    """Return Redis client instance."""
    return RedisClient()


@pytest_asyncio.fixture
async def clean_redis_state():
    """Ensure clean Redis state for testing."""
    # In-memory pending buffer önceki testin event loop'undan taşınmasın
    RedisClient._pending_klines = {}
    RedisClient._pending_publishes = set()
    RedisClient._flusher_task = None
    RedisClient._flush_immediately = True  # testlerde deterministik yazma
    # Clean up any existing test data
    r = RedisClient.get_client()
    try:
        # Delete all test keys
        test_patterns = [
            "live_kline_data:TEST*",
            "indicators:TEST*", 
            "signals:TEST*",
            "hot_klines:TEST*"
        ]
        
        for pattern in test_patterns:
            keys = await r.keys(pattern)
            if keys:
                await r.delete(*keys)
    except Exception:
        pass  # Ignore cleanup errors
    finally:
        await r.close()
    
    yield
    
    # Cleanup after test
    r = RedisClient.get_client()
    try:
        for pattern in test_patterns:
            keys = await r.keys(pattern)
            if keys:
                await r.delete(*keys)
    except Exception:
        pass
    finally:
        await r.close()


# Utility functions for tests
def assert_dataframe_valid(df: pd.DataFrame, min_rows: int = 1):
    """Assert that DataFrame is valid for OHLCV data."""
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) >= min_rows, f"DataFrame should have at least {min_rows} rows"
    
    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
    
    # Check OHLC logic
    for idx, row in df.iterrows():
        assert row['low'] <= row['open'], f"Low should be <= Open at index {idx}"
        assert row['low'] <= row['close'], f"Low should be <= Close at index {idx}"
        assert row['high'] >= row['open'], f"High should be >= Open at index {idx}"
        assert row['high'] >= row['close'], f"High should be >= Close at index {idx}"
        assert row['volume'] >= 0, f"Volume should be >= 0 at index {idx}"


def assert_aggregation_valid(original_df: pd.DataFrame, aggregated_df: pd.DataFrame, ratio: int):
    """Assert that aggregation is mathematically correct."""
    expected_rows = len(original_df) // ratio
    assert len(aggregated_df) == expected_rows, f"Expected {expected_rows} rows, got {len(aggregated_df)}"
    
    # Check first and last timestamps
    if not aggregated_df.empty and not original_df.empty:
        first_agg_time = aggregated_df['open_time'].iloc[0]
        first_orig_time = original_df['open_time'].iloc[0]
        assert first_agg_time == first_orig_time, "First timestamp should match"
        
        # Check volume aggregation (should be sum)
        first_group_volume = original_df['volume'].iloc[:ratio].sum()
        first_agg_volume = aggregated_df['volume'].iloc[0]
        assert abs(first_group_volume - first_agg_volume) < 0.01, "Volume aggregation incorrect"
