"""
signals/paper_trade_manager.py — pozisyon AÇMA/EVOL-erken-çıkış akışları için
testler (on_new_signal, open_direct, check_evol_exits).

İzolasyon: semboller TEST% prefix'li, strateji adı "TEST_STRATEGY" — canlı
stratejilere (ta_kovalama_live vb.) dokunulmaz. _STRATEGY_TRIGGERS ve
Config.PAPER["ENABLED_STRATEGIES"] monkeypatch ile SADECE test süresince
genişletilir (pytest otomatik geri alır).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from config import Config
from database.engine import get_session
from database.models import PaperTrade, Signal
from signals.paper_trade_manager import _STRATEGY_TRIGGERS, PaperTradeManager
from tests.conftest import create_test_signal
from utils.redis_client import RedisClient

TEST_STRATEGY = "TEST_STRATEGY"

pytestmark = pytest.mark.database


@pytest_asyncio.fixture(scope="function")
async def clean_test_data():
    async def _cleanup():
        async with get_session() as session:
            await session.execute(delete(PaperTrade).where(PaperTrade.symbol.like("TEST%")))
            await session.execute(delete(Signal).where(Signal.symbol.like("TEST%")))
            await session.commit()

    await _cleanup()
    yield
    await _cleanup()


@pytest.fixture
def enable_test_strategy(monkeypatch):
    """TEST_STRATEGY'yi geçici olarak etkin kılar ve her zaman tetiklenir hâle getirir."""
    monkeypatch.setitem(Config.PAPER, "ENABLED_STRATEGIES", [TEST_STRATEGY])
    monkeypatch.setitem(_STRATEGY_TRIGGERS, TEST_STRATEGY, lambda sd: True)


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


def _signal_data(**overrides):
    defaults = dict(
        symbol="TESTUSDT",
        signal_type="Long",
        interval="5m",
        vpms_score=50.0,
        mtf_score=50.0,
        atr=2.0,
    )
    defaults.update(overrides)
    return defaults


# --- on_new_signal ---


@pytest.mark.asyncio
async def test_on_new_signal_skipped_when_strategy_disabled(clean_test_data):
    async with get_session() as session:
        sig = await create_test_signal(session)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert result.scalars().first() is None


@pytest.mark.asyncio
async def test_on_new_signal_skipped_when_trigger_false(clean_test_data, monkeypatch):
    monkeypatch.setitem(Config.PAPER, "ENABLED_STRATEGIES", [TEST_STRATEGY])
    monkeypatch.setitem(_STRATEGY_TRIGGERS, TEST_STRATEGY, lambda sd: False)

    async with get_session() as session:
        sig = await create_test_signal(session)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert result.scalars().first() is None


@pytest.mark.asyncio
async def test_on_new_signal_skipped_when_signal_id_none(clean_test_data, enable_test_strategy):
    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(), signal_id=None, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert result.scalars().first() is None


@pytest.mark.asyncio
async def test_on_new_signal_opens_trade_with_computed_sl_tp(clean_test_data, enable_test_strategy):
    async with get_session() as session:
        sig = await create_test_signal(session, atr=2.0)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(atr=2.0), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        trade = result.scalars().one()
        assert trade.status == "open"
        assert trade.entry_price == 100.0
        # DynamicATRPolicy: sl=atr*1.5, tp=atr*3.0 (bonus yok: vpmv/mtf eşik altı, interval 5m)
        assert trade.stop_loss_price == pytest.approx(97.0)
        assert trade.take_profit_price == pytest.approx(106.0)

    assert "TESTUSDT" in mgr._open_symbols


@pytest.mark.asyncio
async def test_on_new_signal_zero_atr_skips_sl_tp(clean_test_data, enable_test_strategy):
    async with get_session() as session:
        sig = await create_test_signal(session, atr=None)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        trade = result.scalars().one()
        assert trade.stop_loss_price is None
        assert trade.take_profit_price is None


@pytest.mark.asyncio
async def test_on_new_signal_skips_when_already_open_same_symbol(
    clean_test_data, enable_test_strategy
):
    async with get_session() as session:
        sig = await create_test_signal(session)
        await _create_trade(session, symbol="TESTUSDT")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.on_new_signal(_signal_data(), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert len(result.scalars().all()) == 1


@pytest.mark.asyncio
async def test_on_new_signal_scale_in_allows_same_direction_second_leg(
    clean_test_data, enable_test_strategy
):
    async with get_session() as session:
        sig = await create_test_signal(session, signal_type="Long")
        await _create_trade(session, symbol="TESTUSDT", signal_type="Long")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY, allow_scale_in=True)
    await mgr.on_new_signal(_signal_data(signal_type="Long"), signal_id=sig.id, current_price=100.0)

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert len(result.scalars().all()) == 2


@pytest.mark.asyncio
async def test_on_new_signal_scale_in_blocks_opposite_direction(
    clean_test_data, enable_test_strategy
):
    async with get_session() as session:
        sig = await create_test_signal(session, signal_type="Short")
        await _create_trade(session, symbol="TESTUSDT", signal_type="Long")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY, allow_scale_in=True)
    await mgr.on_new_signal(
        _signal_data(signal_type="Short"), signal_id=sig.id, current_price=100.0
    )

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == TEST_STRATEGY)
        )
        assert len(result.scalars().all()) == 1


@pytest.mark.asyncio
async def test_on_new_signal_skips_when_max_open_reached(clean_test_data, enable_test_strategy):
    async with get_session() as session:
        sig = await create_test_signal(session, symbol="TESTUSDT_NEW")
        await _create_trade(session, symbol="TESTUSDT_OTHER")  # kapasiteyi doldurur

    mgr = PaperTradeManager(strategy=TEST_STRATEGY, max_open=1)
    await mgr.on_new_signal(
        _signal_data(symbol="TESTUSDT_NEW"), signal_id=sig.id, current_price=100.0
    )

    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.symbol == "TESTUSDT_NEW")
        )
        assert result.scalars().first() is None


# --- open_direct ---


@pytest.mark.asyncio
async def test_open_direct_skipped_when_strategy_disabled(clean_test_data):
    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    opened = await mgr.open_direct(
        "TESTUSDT", "Long", "5m", price=100.0, atr=2.0, sl_price=97.0, tp_price=106.0
    )
    assert opened is False

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.symbol == "TESTUSDT"))
        assert result.scalars().first() is None


@pytest.mark.asyncio
async def test_open_direct_opens_trade_successfully(clean_test_data, enable_test_strategy):
    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    opened = await mgr.open_direct(
        "TESTUSDT", "Long", "5m", price=100.0, atr=2.0, sl_price=97.0, tp_price=106.0, note="test"
    )
    assert opened is True

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.symbol == "TESTUSDT"))
        trade = result.scalars().one()
        assert trade.status == "open"
        assert trade.stop_loss_price == 97.0
        assert trade.take_profit_price == 106.0

    assert "TESTUSDT" in mgr._open_symbols


@pytest.mark.asyncio
async def test_open_direct_returns_false_when_already_open(clean_test_data, enable_test_strategy):
    async with get_session() as session:
        await _create_trade(session, symbol="TESTUSDT")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    opened = await mgr.open_direct(
        "TESTUSDT", "Long", "5m", price=100.0, atr=2.0, sl_price=97.0, tp_price=106.0
    )
    assert opened is False


@pytest.mark.asyncio
async def test_open_direct_returns_false_when_max_open_reached(
    clean_test_data, enable_test_strategy
):
    async with get_session() as session:
        await _create_trade(session, symbol="TESTUSDT_OTHER")

    mgr = PaperTradeManager(strategy=TEST_STRATEGY, max_open=1)
    opened = await mgr.open_direct(
        "TESTUSDT_NEW", "Long", "5m", price=100.0, atr=2.0, sl_price=97.0, tp_price=106.0
    )
    assert opened is False


# --- check_evol_exits ---


def _make_evol_df(low: bool) -> pd.DataFrame:
    n = 150
    if low:
        np.random.seed(7)
        price = 100 + np.cumsum(np.random.normal(0, 2.0, n))
        volume = np.full(n, 1000.0)
        k = 30
        price[-k:] = price[-k - 1]
        volume[-k:] = 100_000.0
    else:
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.normal(0, 0.3, n))
        volume = np.full(n, 1000.0)
        volume[-5:] = 200.0
        price[-5:] = price[-6] + np.linspace(0, 5.0, 5)
    return pd.DataFrame({"open_time": np.arange(n) * 300_000, "close": price, "volume": volume})


@pytest.mark.asyncio
async def test_check_evol_exits_skips_too_young_position(clean_test_data):
    async with get_session() as session:
        trade = await _create_trade(
            session, symbol="TESTUSDT", signal_type="Long", opened_at=datetime.now()
        )
        trade_id = trade.id

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.check_evol_exits()

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        assert result.scalar_one().status == "open"


@pytest.mark.asyncio
async def test_check_evol_exits_closes_on_low_evol(clean_test_data, monkeypatch):
    old_open = datetime.now() - timedelta(hours=2)
    async with get_session() as session:
        trade = await _create_trade(
            session, symbol="TESTUSDT", signal_type="Long", interval="5m", opened_at=old_open
        )
        trade_id = trade.id

    async def _fake_mtf_klines(symbol, timeframe, limit=None):
        return _make_evol_df(low=True)

    monkeypatch.setattr(RedisClient, "get_mtf_klines", _fake_mtf_klines)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    mgr._open_symbols.add("TESTUSDT")
    await mgr.check_evol_exits()

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        updated = result.scalar_one()
        assert updated.status == "closed"
        assert updated.close_reason == "evol_decay"

    assert "TESTUSDT" not in mgr._open_symbols


@pytest.mark.asyncio
async def test_check_evol_exits_keeps_open_on_high_evol(clean_test_data, monkeypatch):
    old_open = datetime.now() - timedelta(hours=2)
    async with get_session() as session:
        trade = await _create_trade(
            session, symbol="TESTUSDT", signal_type="Long", interval="5m", opened_at=old_open
        )
        trade_id = trade.id

    async def _fake_mtf_klines(symbol, timeframe, limit=None):
        return _make_evol_df(low=False)

    monkeypatch.setattr(RedisClient, "get_mtf_klines", _fake_mtf_klines)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.check_evol_exits()

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        assert result.scalar_one().status == "open"


@pytest.mark.asyncio
async def test_check_evol_exits_ignores_short_positions(clean_test_data, monkeypatch):
    old_open = datetime.now() - timedelta(hours=2)
    async with get_session() as session:
        trade = await _create_trade(
            session, symbol="TESTUSDT", signal_type="Short", interval="5m", opened_at=old_open
        )
        trade_id = trade.id

    async def _fake_mtf_klines(symbol, timeframe, limit=None):
        return _make_evol_df(low=True)

    monkeypatch.setattr(RedisClient, "get_mtf_klines", _fake_mtf_klines)

    mgr = PaperTradeManager(strategy=TEST_STRATEGY)
    await mgr.check_evol_exits()

    async with get_session() as session:
        result = await session.execute(select(PaperTrade).where(PaperTrade.id == trade_id))
        assert result.scalar_one().status == "open"
