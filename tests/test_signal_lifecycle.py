"""
SignalLifecycleManager için pytest testleri
Supersede mekaniğinin doğru çalışıp çalışmadığını test eder
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from signals.signal_lifecycle_manager import SignalLifecycleManager
from database.engine import get_session
from database.models import Signal
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

@pytest_asyncio.fixture
async def lifecycle_manager():
    """Test için SignalLifecycleManager instance'ı"""
    return SignalLifecycleManager()


@pytest_asyncio.fixture(scope="function")
async def clean_test_signals():
    """Test öncesi ve sonrası test sinyallerini temizle"""
    # Test öncesi temizlik
    async with get_session() as session:
        try:
            await session.execute(
                delete(Signal).where(Signal.symbol.like('TEST%'))
            )
            await session.commit()
        except Exception:
            await session.rollback()
    
    yield
    
    # Test sonrası temizlik
    async with get_session() as session:
        try:
            await session.execute(
                delete(Signal).where(Signal.symbol.like('TEST%'))
            )
            await session.commit()
        except Exception:
            await session.rollback()


def create_test_signal_data(
    symbol: str = "TESTUSDT",
    signal_type: str = "Long",
    interval: str = "15m",
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Test sinyali verisi oluştur"""
    if timestamp is None:
        timestamp = datetime.now()
    
    return {
        'symbol': symbol,
        'timestamp': timestamp,
        'signal_type': signal_type,
        'interval': interval,
        'price': 100.0,
        'strength': 1,
        'indicators': 'TEST_INDICATOR',
        'rsi': 50.0,
        'macd': 0.5,
        'momentum': 1.2,
        'atr': 2.0,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 1.8,
        'vpms_score': 0.7,
        'vpm_confirmed': True
    }


@pytest.mark.asyncio
async def test_add_first_signal(lifecycle_manager, clean_test_signals):
    """İlk sinyal ekleme testi - supersede olmamalı"""
    
    signal_data = create_test_signal_data()
    
    # İlk sinyali ekle
    signal_id = await lifecycle_manager.add_new_signal(signal_data)
    
    assert signal_id is not None
    
    # Veritabanından kontrol et
    async with get_session() as session:
        stmt = select(Signal).where(
            and_(
                Signal.symbol == signal_data['symbol'],
                Signal.timestamp == signal_data['timestamp']
            )
        )
        result = await session.execute(stmt)
        signal = result.scalar_one_or_none()
        
        assert signal is not None
        assert signal.status == 'active'
        assert signal.superseded_by is None
        assert signal.superseded_at is None


@pytest.mark.asyncio
async def test_supersede_same_symbol_same_direction(lifecycle_manager, clean_test_signals):
    """Aynı sembol, aynı yön - eski sinyal supersede edilmeli"""
    
    base_time = datetime.now()
    
    # İlk Long sinyal
    signal_data_1 = create_test_signal_data(
        timestamp=base_time
    )
    signal_id_1 = await lifecycle_manager.add_new_signal(signal_data_1)
    
    # 5 dakika sonra ikinci Long sinyal
    signal_data_2 = create_test_signal_data(
        timestamp=base_time + timedelta(minutes=5)
    )
    signal_id_2 = await lifecycle_manager.add_new_signal(signal_data_2)
    
    # Kontrol et
    async with get_session() as session:
        # İlk sinyal supersede edilmeli
        stmt_1 = select(Signal).where(Signal.id == signal_id_1)
        result_1 = await session.execute(stmt_1)
        signal_1 = result_1.scalar_one()
        
        assert signal_1.status == 'superseded'
        assert signal_1.superseded_by == signal_id_2
        assert signal_1.superseded_at is not None
        assert signal_1.lifecycle_end_reason == 'supersede'
        
        # İkinci sinyal aktif olmalı
        stmt_2 = select(Signal).where(Signal.id == signal_id_2)
        result_2 = await session.execute(stmt_2)
        signal_2 = result_2.scalar_one()
        
        assert signal_2.status == 'active'
        assert signal_2.superseded_by is None


@pytest.mark.asyncio
async def test_supersede_same_symbol_opposite_direction(lifecycle_manager, clean_test_signals):
    """Aynı sembol, ters yön - eski sinyal supersede edilmeli"""
    
    base_time = datetime.now()
    
    # İlk Long sinyal
    signal_data_1 = create_test_signal_data(
        signal_type="Long",
        timestamp=base_time
    )
    signal_id_1 = await lifecycle_manager.add_new_signal(signal_data_1)
    
    # 3 dakika sonra Short sinyal
    signal_data_2 = create_test_signal_data(
        signal_type="Short",
        timestamp=base_time + timedelta(minutes=3)
    )
    signal_id_2 = await lifecycle_manager.add_new_signal(signal_data_2)
    
    # Kontrol et
    async with get_session() as session:
        # İlk sinyal supersede edilmeli
        stmt_1 = select(Signal).where(Signal.id == signal_id_1)
        result_1 = await session.execute(stmt_1)
        signal_1 = result_1.scalar_one()
        
        assert signal_1.status == 'superseded'
        assert signal_1.superseded_by == signal_id_2
        
        # İkinci sinyal aktif olmalı
        stmt_2 = select(Signal).where(Signal.id == signal_id_2)
        result_2 = await session.execute(stmt_2)
        signal_2 = result_2.scalar_one()
        
        assert signal_2.status == 'active'


@pytest.mark.asyncio
async def test_different_symbols_no_supersede(lifecycle_manager, clean_test_signals):
    """Farklı semboller - supersede olmamalı"""
    
    base_time = datetime.now()
    
    # TESTUSDT Long sinyal
    signal_data_1 = create_test_signal_data(
        symbol="TESTUSDT",
        timestamp=base_time
    )
    signal_id_1 = await lifecycle_manager.add_new_signal(signal_data_1)
    
    # TEST2USDT Long sinyal
    signal_data_2 = create_test_signal_data(
        symbol="TEST2USDT",
        timestamp=base_time + timedelta(minutes=1)
    )
    signal_id_2 = await lifecycle_manager.add_new_signal(signal_data_2)
    
    # Kontrol et - ikisi de aktif olmalı
    async with get_session() as session:
        stmt_1 = select(Signal).where(Signal.id == signal_id_1)
        result_1 = await session.execute(stmt_1)
        signal_1 = result_1.scalar_one()
        
        stmt_2 = select(Signal).where(Signal.id == signal_id_2)
        result_2 = await session.execute(stmt_2)
        signal_2 = result_2.scalar_one()
        
        assert signal_1.status == 'active'
        assert signal_2.status == 'active'
        assert signal_1.superseded_by is None
        assert signal_2.superseded_by is None


@pytest.mark.asyncio
async def test_different_intervals_no_supersede(lifecycle_manager, clean_test_signals):
    """Farklı interval'lar - supersede olmamalı"""
    
    base_time = datetime.now()
    
    # 15m interval Long sinyal
    signal_data_1 = create_test_signal_data(
        interval="15m",
        timestamp=base_time
    )
    signal_id_1 = await lifecycle_manager.add_new_signal(signal_data_1)
    
    # 1h interval Long sinyal
    signal_data_2 = create_test_signal_data(
        interval="1h",
        timestamp=base_time + timedelta(minutes=1)
    )
    signal_id_2 = await lifecycle_manager.add_new_signal(signal_data_2)
    
    # Kontrol et - ikisi de aktif olmalı
    async with get_session() as session:
        stmt_1 = select(Signal).where(Signal.id == signal_id_1)
        result_1 = await session.execute(stmt_1)
        signal_1 = result_1.scalar_one()
        
        stmt_2 = select(Signal).where(Signal.id == signal_id_2)
        result_2 = await session.execute(stmt_2)
        signal_2 = result_2.scalar_one()
        
        assert signal_1.status == 'active'
        assert signal_2.status == 'active'


@pytest.mark.asyncio
async def test_get_active_signals(lifecycle_manager, clean_test_signals):
    """Aktif sinyal getirme testi"""
    
    base_time = datetime.now()
    
    # 3 sinyal ekle, 1 tanesini supersede et
    signal_data_1 = create_test_signal_data(
        symbol="TESTUSDT",
        timestamp=base_time
    )
    signal_id_1 = await lifecycle_manager.add_new_signal(signal_data_1)
    
    signal_data_2 = create_test_signal_data(
        symbol="TEST2USDT",
        timestamp=base_time + timedelta(minutes=1)
    )
    signal_id_2 = await lifecycle_manager.add_new_signal(signal_data_2)
    
    # TESTUSDT için yeni sinyal (ilkini supersede eder)
    signal_data_3 = create_test_signal_data(
        symbol="TESTUSDT",
        timestamp=base_time + timedelta(minutes=2)
    )
    signal_id_3 = await lifecycle_manager.add_new_signal(signal_data_3)
    
    # Aktif sinyalleri getir
    active_signals = await lifecycle_manager.get_active_signals()
    
    # 2 aktif sinyal olmalı (signal_2 ve signal_3)
    assert len(active_signals) == 2
    
    active_ids = [s.id for s in active_signals]
    assert signal_id_2 in active_ids
    assert signal_id_3 in active_ids
    assert signal_id_1 not in active_ids


@pytest.mark.asyncio
async def test_manual_close_signal(lifecycle_manager, clean_test_signals):
    """Manuel sinyal kapatma testi"""
    
    signal_data = create_test_signal_data()
    signal_id = await lifecycle_manager.add_new_signal(signal_data)
    
    # Manuel kapat
    result = await lifecycle_manager.manual_close_signal(signal_id, "manual")
    assert result is True
    
    # Kontrol et
    async with get_session() as session:
        stmt = select(Signal).where(Signal.id == signal_id)
        result = await session.execute(stmt)
        signal = result.scalar_one()
        
        assert signal.status == 'completed'
        assert signal.lifecycle_end_reason == 'manual'
        assert signal.superseded_at is not None


@pytest.mark.asyncio
async def test_signal_statistics(lifecycle_manager, clean_test_signals):
    """Sinyal istatistikleri testi"""
    
    base_time = datetime.now()
    
    # 2 aktif, 1 superseded sinyal oluştur
    signal_data_1 = create_test_signal_data(
        symbol="TESTUSDT",
        timestamp=base_time
    )
    await lifecycle_manager.add_new_signal(signal_data_1)
    
    signal_data_2 = create_test_signal_data(
        symbol="TEST2USDT",
        timestamp=base_time + timedelta(minutes=1)
    )
    await lifecycle_manager.add_new_signal(signal_data_2)
    
    # TESTUSDT için yeni sinyal (ilkini supersede eder)
    signal_data_3 = create_test_signal_data(
        symbol="TESTUSDT",
        timestamp=base_time + timedelta(minutes=2)
    )
    await lifecycle_manager.add_new_signal(signal_data_3)
    
    # İstatistikleri al
    stats = await lifecycle_manager.get_signal_statistics()
    
    assert stats['active_signals'] >= 2
    assert stats['superseded_24h'] >= 1
    assert 'avg_performance_period_minutes' in stats


if __name__ == "__main__":
    # Test'leri çalıştır
    pytest.main([__file__, "-v"])
