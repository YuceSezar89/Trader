"""
utils/data_health.py::check_null_ratios — testler.

symbol_pattern='TEST%' ile gerçek DB'ye karşı çalışır ama SADECE TEST%
sembollerini sayar — canlı signals/paper_trades verisiyle hiç karışmaz,
gerçek üretim oranlarını etkilemez/okumaz.
"""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete

from database.engine import get_session
from database.models import PaperTrade, Signal
from tests.conftest import create_test_signal
from utils.data_health import check_null_ratios

pytestmark = pytest.mark.database


@pytest_asyncio.fixture(scope="function")
async def clean_test_data():
    async def _cleanup():
        async with get_session() as session:
            await session.execute(delete(Signal).where(Signal.symbol.like("TEST%")))
            await session.execute(delete(PaperTrade).where(PaperTrade.symbol.like("TEST%")))
            await session.commit()

    await _cleanup()
    yield
    await _cleanup()


async def _create_closed_trade(session, **overrides) -> PaperTrade:
    defaults = dict(
        strategy="TEST_STRATEGY",
        source="signal",
        symbol="TESTUSDT",
        signal_type="Long",
        interval="5m",
        position_usd=100.0,
        entry_price=100.0,
        status="closed",
        opened_at=datetime.now() - timedelta(minutes=10),
        closed_at=datetime.now(),
        exit_price=105.0,
        close_reason="take_profit",
        pnl_pct=5.0,
        pnl_usd=5.0,
    )
    defaults.update(overrides)
    trade = PaperTrade(**defaults)
    session.add(trade)
    await session.flush()
    await session.commit()
    return trade


# --- signals_closed ---


@pytest.mark.asyncio
async def test_no_closed_rows_returns_zero_ratio(clean_test_data):
    results = await check_null_ratios(symbol_pattern="TEST%")
    null_count, total, ratio = results["signals_closed"]
    assert (null_count, total, ratio) == (0, 0, 0.0)


@pytest.mark.asyncio
async def test_complete_closed_signals_have_zero_null_ratio(clean_test_data):
    async with get_session() as session:
        for i in range(3):
            await create_test_signal(
                session,
                symbol=f"TESTUSDT{i}",
                status="closed",
                closed_at=datetime.now(),
                close_reason="stop_loss",
                close_price=95.0,
                realized_pnl=-5.0,
            )

    _, total, ratio = (await check_null_ratios(symbol_pattern="TEST%"))["signals_closed"]
    assert total == 3
    assert ratio == 0.0


@pytest.mark.asyncio
async def test_incomplete_closed_signal_detected_as_null(clean_test_data):
    async with get_session() as session:
        await create_test_signal(
            session,
            symbol="TESTUSDT1",
            status="closed",
            closed_at=datetime.now(),
            close_reason="stop_loss",
            close_price=95.0,
            realized_pnl=-5.0,
        )
        # yarım kalmış kapanış: close_reason/realized_pnl NULL ama status='closed'
        await create_test_signal(
            session,
            symbol="TESTUSDT2",
            status="closed",
            closed_at=datetime.now(),
            close_reason=None,
            close_price=None,
            realized_pnl=None,
        )

    null_count, total, ratio = (await check_null_ratios(symbol_pattern="TEST%"))["signals_closed"]
    assert total == 2
    assert null_count == 1
    assert ratio == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_signal_outside_window_excluded(clean_test_data):
    async with get_session() as session:
        await create_test_signal(
            session,
            symbol="TESTUSDT_OLD",
            status="closed",
            closed_at=datetime.now() - timedelta(minutes=120),
            close_reason="stop_loss",
            close_price=95.0,
            realized_pnl=-5.0,
        )

    _, total, _ = (await check_null_ratios(window_minutes=60, symbol_pattern="TEST%"))[
        "signals_closed"
    ]
    assert total == 0  # pencere dışında kaldı


@pytest.mark.asyncio
async def test_active_signal_not_counted_as_closed(clean_test_data):
    async with get_session() as session:
        await create_test_signal(session, symbol="TESTUSDT1", status="active")

    _, total, _ = (await check_null_ratios(symbol_pattern="TEST%"))["signals_closed"]
    assert total == 0


# --- paper_trades_closed ---


@pytest.mark.asyncio
async def test_complete_closed_trades_have_zero_null_ratio(clean_test_data):
    async with get_session() as session:
        await _create_closed_trade(session, symbol="TESTUSDT1")
        await _create_closed_trade(session, symbol="TESTUSDT2")

    _, total, ratio = (await check_null_ratios(symbol_pattern="TEST%"))["paper_trades_closed"]
    assert total == 2
    assert ratio == 0.0


@pytest.mark.asyncio
async def test_incomplete_closed_trade_detected_as_null(clean_test_data):
    async with get_session() as session:
        await _create_closed_trade(session, symbol="TESTUSDT1")
        await _create_closed_trade(
            session,
            symbol="TESTUSDT2",
            close_reason=None,
            pnl_usd=None,
            pnl_pct=None,
            exit_price=None,
        )

    null_count, total, ratio = (await check_null_ratios(symbol_pattern="TEST%"))[
        "paper_trades_closed"
    ]
    assert total == 2
    assert null_count == 1
    assert ratio == pytest.approx(0.5)
