"""
signals/risk_manager.py — DB'ye bağlı async metodlar için testler
(load_active_symbols, check_all_prices).

Gerçek DB'ye karşı çalışır (bkz. tests/conftest.py), ama SADECE TEST%
prefix'li sembollerle — canlı `signals` verisine dokunmaz. Her testten
önce/sonra TEST% satırları temizlenir.
"""

import pytest
from sqlalchemy import select

from database.engine import get_session
from database.models import Signal
from signals.risk_manager import RiskManager
from tests.conftest import create_test_signal

pytestmark = pytest.mark.database

# risk_manager._early_exit_check SADECE 5m/15m interval'lerde tetiklenir —
# check_all_prices testleri saf SL/trailing mantığını izole test etmek için
# 1h kullanır (early-exit bulaşmasın).


# --- load_active_symbols ---


@pytest.mark.asyncio
async def test_load_active_symbols_includes_active_with_sl(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(session, symbol="TESTUSDT1", status="active", stop_loss_price=95.0)

    rm = RiskManager()
    await rm.load_active_symbols()

    assert "TESTUSDT1" in rm._active_symbols


@pytest.mark.asyncio
async def test_load_active_symbols_excludes_without_stop_loss(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(session, symbol="TESTUSDT2", status="active", stop_loss_price=None)

    rm = RiskManager()
    await rm.load_active_symbols()

    assert "TESTUSDT2" not in rm._active_symbols


@pytest.mark.asyncio
async def test_load_active_symbols_excludes_closed(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(session, symbol="TESTUSDT3", status="closed", stop_loss_price=90.0)

    rm = RiskManager()
    await rm.load_active_symbols()

    assert "TESTUSDT3" not in rm._active_symbols


# --- check_all_prices ---


@pytest.mark.asyncio
async def test_check_all_prices_noop_without_active_symbols(clean_test_signals):
    rm = RiskManager()
    # _active_symbols boş -> DB'ye hiç dokunmadan sessizce dönmeli
    await rm.check_all_prices({"TESTUSDT": 100.0})
    assert rm._active_symbols == set()


@pytest.mark.asyncio
async def test_check_all_prices_closes_on_stop_loss(clean_test_signals):
    async with get_session() as session:
        sig = await create_test_signal(
            session,
            symbol="TESTUSDT",
            interval="1h",
            signal_type="Long",
            open_price=100.0,
            atr=2.0,
            sl_multiplier=1.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )
        sig_id = sig.id

    rm = RiskManager()
    await rm.load_active_symbols()
    assert "TESTUSDT" in rm._active_symbols

    await rm.check_all_prices({"TESTUSDT": 94.0})

    async with get_session() as session:
        result = await session.execute(select(Signal).where(Signal.id == sig_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "stop_loss"
        assert updated.close_price == 94.0
        assert updated.realized_pnl == -6.0  # (94-100)/100*100

    assert "TESTUSDT" not in rm._active_symbols


@pytest.mark.asyncio
async def test_check_all_prices_activates_trailing_without_closing(clean_test_signals):
    async with get_session() as session:
        sig = await create_test_signal(
            session,
            symbol="TESTUSDT",
            interval="1h",
            signal_type="Long",
            open_price=100.0,
            atr=2.0,
            sl_multiplier=1.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )
        sig_id = sig.id

    rm = RiskManager()
    await rm.load_active_symbols()

    await rm.check_all_prices({"TESTUSDT": 112.0})

    async with get_session() as session:
        result = await session.execute(select(Signal).where(Signal.id == sig_id))
        updated = result.scalar_one()
        assert updated.status == "active"
        assert updated.trailing_stop_price == 110.0  # price(112) - dist(2.0)

    assert "TESTUSDT" in rm._active_symbols


@pytest.mark.asyncio
async def test_check_all_prices_triggers_trailing_stop(clean_test_signals):
    async with get_session() as session:
        sig = await create_test_signal(
            session,
            symbol="TESTUSDT",
            interval="1h",
            signal_type="Long",
            open_price=100.0,
            atr=2.0,
            sl_multiplier=1.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            trailing_stop_price=113.0,
        )
        sig_id = sig.id

    rm = RiskManager()
    await rm.load_active_symbols()

    await rm.check_all_prices({"TESTUSDT": 113.0})

    async with get_session() as session:
        result = await session.execute(select(Signal).where(Signal.id == sig_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "trailing_stop"
        assert updated.close_price == 113.0

    assert "TESTUSDT" not in rm._active_symbols


@pytest.mark.asyncio
async def test_check_all_prices_ignores_symbols_not_in_price_map(clean_test_signals):
    async with get_session() as session:
        sig = await create_test_signal(
            session,
            symbol="TESTUSDT",
            interval="1h",
            signal_type="Long",
            open_price=100.0,
            stop_loss_price=95.0,
        )
        sig_id = sig.id

    rm = RiskManager()
    await rm.load_active_symbols()

    await rm.check_all_prices({"OTHERUSDT": 50.0})

    async with get_session() as session:
        result = await session.execute(select(Signal).where(Signal.id == sig_id))
        updated = result.scalar_one()
        assert updated.status == "active"

    assert "TESTUSDT" in rm._active_symbols
