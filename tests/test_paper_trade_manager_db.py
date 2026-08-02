"""
signals/paper_trade_manager.py — DB'ye bağlı async metodlar için testler
(load_open_symbols, check_all_prices, async _trail_distance).

Gerçek DB'ye karşı çalışır ama izole veriyle: semboller TEST% prefix'li,
strateji adı "TEST_STRATEGY" (rsi_cross_live/ta_kovalama_live/conf_100/
do_kirilimi gibi CANLI stratejilere asla dokunulmaz — PaperPortfolio.strategy
unique olduğu için ayrı bir satır kullanılır, testler sonunda silinir).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from database.engine import get_session
from database.models import PaperPortfolio, PaperTrade
from signals.paper_trade_manager import PaperTradeManager
from utils.redis_client import RedisClient

TEST_STRATEGY = "TEST_STRATEGY"

pytestmark = pytest.mark.database


@pytest_asyncio.fixture(scope="function")
async def clean_test_trades():
    async def _cleanup():
        async with get_session() as session:
            await session.execute(delete(PaperTrade).where(PaperTrade.symbol.like("TEST%")))
            await session.execute(
                delete(PaperPortfolio).where(PaperPortfolio.strategy == TEST_STRATEGY)
            )
            await session.commit()

    await _cleanup()
    yield
    await _cleanup()
    try:
        client = RedisClient.get_client()
        await client.delete("atr_live:TESTUSDT:5m")
    except Exception:  # pylint: disable=broad-exception-caught
        pass


async def _create_trade(session, **overrides) -> PaperTrade:
    defaults = dict(
        strategy=TEST_STRATEGY,
        source="signal",
        symbol="TESTUSDT",
        signal_type="Long",
        interval="5m",
        position_usd=100.0,
        entry_price=100.0,
        status="open",
        opened_at=datetime.now(),
    )
    defaults.update(overrides)
    trade = PaperTrade(**defaults)
    session.add(trade)
    await session.flush()
    await session.commit()
    return trade


# --- load_open_symbols ---


@pytest.mark.asyncio
async def test_load_open_symbols_includes_own_strategy_open_trade(clean_test_trades):
    async with get_session() as session:
        await _create_trade(session, symbol="TESTUSDT1", status="open")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    assert "TESTUSDT1" in mgr._open_symbols


@pytest.mark.asyncio
async def test_load_open_symbols_excludes_closed_trade(clean_test_trades):
    async with get_session() as session:
        await _create_trade(session, symbol="TESTUSDT2", status="closed")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    assert "TESTUSDT2" not in mgr._open_symbols


@pytest.mark.asyncio
async def test_load_open_symbols_excludes_other_strategy(clean_test_trades):
    async with get_session() as session:
        await _create_trade(
            session, symbol="TESTUSDT3", status="open", strategy="OTHER_TEST_STRATEGY"
        )

    try:
        mgr = PaperTradeManager(strategy=TEST_STRATEGY)
        await mgr.load_open_symbols()
        assert "TESTUSDT3" not in mgr._open_symbols
    finally:
        async with get_session() as session:
            await session.execute(
                delete(PaperTrade).where(PaperTrade.strategy == "OTHER_TEST_STRATEGY")
            )
            await session.commit()


# --- check_all_prices ---


@pytest.mark.asyncio
async def test_check_all_prices_noop_without_open_symbols(clean_test_trades):
    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.check_all_prices({"TESTUSDT": 100.0})
    assert mgr._open_symbols == set()


@pytest.mark.asyncio
async def test_check_all_prices_closes_on_stop_loss_without_portfolio(clean_test_trades):
    async with get_session() as session:
        trade = await _create_trade(
            session,
            symbol="TESTUSDT",
            signal_type="Long",
            entry_price=100.0,
            atr=2.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )
        trade_id = trade.id

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()
    assert "TESTUSDT" in mgr._open_symbols

    await mgr.check_all_prices({"TESTUSDT": 94.0})

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "stop_loss"
        assert updated.exit_price == 94.0
        assert updated.pnl_pct == -6.0
        assert updated.balance_after is None  # portföy satırı yok

    assert "TESTUSDT" not in mgr._open_symbols


@pytest.mark.asyncio
async def test_check_all_prices_updates_portfolio_on_close(clean_test_trades):
    async with get_session() as session:
        portfolio = PaperPortfolio(
            strategy=TEST_STRATEGY,
            balance=10000.0,
            initial_balance=10000.0,
            peak_balance=10000.0,
        )
        session.add(portfolio)
        trade = await _create_trade(
            session,
            symbol="TESTUSDT",
            signal_type="Long",
            entry_price=100.0,
            atr=2.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )
        trade_id = trade.id
        await session.commit()

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    await mgr.check_all_prices({"TESTUSDT": 94.0})

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        trade_after = result.scalar_one()
        pf_result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.strategy == TEST_STRATEGY)
        )
        pf_after = pf_result.scalar_one()

        assert trade_after.balance_after == pf_after.balance
        assert pf_after.total_trades == 1
        assert pf_after.winning_trades == 0
        assert pf_after.balance < 10000.0


@pytest.mark.asyncio
async def test_check_all_prices_activates_trailing_without_closing(clean_test_trades):
    async with get_session() as session:
        trade = await _create_trade(
            session,
            symbol="TESTUSDT",
            signal_type="Long",
            entry_price=100.0,
            atr=2.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )
        trade_id = trade.id

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    await mgr.check_all_prices({"TESTUSDT": 112.0})

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        updated = result.scalar_one()
        assert updated.status == "open"
        assert updated.trailing_stop_price is not None

    assert "TESTUSDT" in mgr._open_symbols


@pytest.mark.asyncio
async def test_check_all_prices_triggers_trailing_stop(clean_test_trades):
    async with get_session() as session:
        trade = await _create_trade(
            session,
            symbol="TESTUSDT",
            signal_type="Long",
            entry_price=100.0,
            atr=2.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            trailing_stop_price=113.0,
        )
        trade_id = trade.id

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    await mgr.check_all_prices({"TESTUSDT": 113.0})

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "trailing_stop"

    assert "TESTUSDT" not in mgr._open_symbols


@pytest.mark.asyncio
async def test_check_all_prices_closes_on_max_hold_timeout(clean_test_trades):
    old_open_time = datetime.now() - timedelta(hours=5)
    async with get_session() as session:
        trade = await _create_trade(
            session,
            symbol="TESTUSDT",
            signal_type="Long",
            entry_price=100.0,
            stop_loss_price=95.0,
            opened_at=old_open_time,
        )
        trade_id = trade.id

    mgr = PaperTradeManager(strategy=TEST_STRATEGY, max_hold_hours=4.0)
    await mgr.load_open_symbols()

    await mgr.check_all_prices({"TESTUSDT": 101.0})

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "timeout"


@pytest.mark.asyncio
async def test_check_all_prices_ignores_symbols_not_in_price_map(clean_test_trades):
    async with get_session() as session:
        await _create_trade(
            session, symbol="TESTUSDT", signal_type="Long", entry_price=100.0, stop_loss_price=95.0
        )

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.load_open_symbols()

    await mgr.check_all_prices({"OTHERUSDT": 50.0})

    assert "TESTUSDT" in mgr._open_symbols


# --- async _trail_distance ---


@pytest.mark.asyncio
async def test_trail_distance_uses_live_atr_from_redis(clean_test_trades):
    client = RedisClient.get_client()
    await client.set("atr_live:TESTUSDT:5m", "3.0")

    trade = await _create_dummy_trade()
    dist = await PaperTradeManager._trail_distance(trade)

    assert dist == pytest.approx(3.0 * 1.5)  # Config.RISK_SL_MULTIPLIER


@pytest.mark.asyncio
async def test_trail_distance_falls_back_to_static_without_redis_key(clean_test_trades):
    client = RedisClient.get_client()
    await client.delete("atr_live:TESTUSDT:5m")

    trade = await _create_dummy_trade(atr=2.0, stop_loss_price=95.0, entry_price=100.0)
    dist = await PaperTradeManager._trail_distance(trade)

    assert dist == 5.0  # abs(entry - sl), Redis key yok


async def _create_dummy_trade(**overrides):
    defaults = dict(
        symbol="TESTUSDT",
        interval="5m",
        atr=None,
        stop_loss_price=None,
        entry_price=100.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
