"""
SignalFilter testleri — DB-tabanlı (signal_filter_events tablosu, bkz. migration 016).

Önceki tasarım in-memory dict + Redis persist idi; artık her check() denemesi
signal_filter_events'e yazılıyor ve referans oradan okunuyor (bkz. proje notları:
restart/çoklu-process senkron sorunlarını yapısal olarak ortadan kaldırmak için).

Senaryolar:
- İlk long / ilk short her zaman geçersiz (referans yok)
- Long geçerli: high > önceki short high
- Long geçersiz: high <= önceki short high
- Short geçerli: low < önceki long low
- Short geçersiz: low >= önceki long low
- Ardışık aynı yön: her sinyal kendi referansını günceller
- Farklı key'ler birbirini etkilemez

Her test benzersiz bir symbol kullanır (test_<uuid>) ve sonunda kendi satırlarını
siler — gerçek/paylaşılan verilere karışmaz, testler arası izolasyon sağlanır.
"""

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text

from database.engine import async_engine, get_session
from signals.signal_filter import SignalFilter

IV = "1h"
IND = "Supertrend"

pytestmark = pytest.mark.database


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine_pool():
    """pytest-asyncio 1.x her testi kendi (function-scoped) event loop'unda
    çalıştırır, ama async_engine'in bağlantı havuzu process ömrü boyunca tek bir
    global nesne — bir önceki testin (artık kapanmış) loop'unda açılmış bir
    asyncpg bağlantısı havuzda kalıp bu testte kullanılmaya çalışılırsa
    "Event loop is closed" hatası verir. Her testten SONRA havuzu boşaltmak,
    bir sonraki testin kendi loop'unda temiz bağlantı açmasını garantiler."""
    yield
    await async_engine.dispose()


@pytest.fixture
def f():
    return SignalFilter()


@pytest_asyncio.fixture
async def sym():
    """Her test için benzersiz sembol — izolasyon ve otomatik temizlik."""
    test_symbol = f"TEST{uuid.uuid4().hex[:10].upper()}"
    yield test_symbol
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM signal_filter_events WHERE symbol = :sym"),
            {"sym": test_symbol},
        )


def _bar_times(n: int):
    """n adet artan zaman damgası — ORDER BY bar_time DESC'in doğru sıralanması için."""
    base = datetime.now()
    return [base + timedelta(minutes=i) for i in range(n)]


@pytest.mark.asyncio
async def test_first_long_invalid(f, sym):
    """İlk long sinyali — önceki short high yok, geçersiz."""
    t = _bar_times(1)
    assert await f.check("Long", high=105.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0]) is False


@pytest.mark.asyncio
async def test_first_short_invalid(f, sym):
    """İlk short sinyali — önceki long low yok, geçersiz."""
    t = _bar_times(1)
    assert await f.check("Short", high=105.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0]) is False


@pytest.mark.asyncio
async def test_long_valid_after_short(f, sym):
    """Short sonrası long: high > short high → geçerli."""
    t = _bar_times(2)
    await f.check("Short", high=100.0, low=90.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    result = await f.check("Long", high=101.0, low=95.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert result is True


@pytest.mark.asyncio
async def test_long_invalid_after_short(f, sym):
    """Short sonrası long: high <= short high → geçersiz."""
    t = _bar_times(2)
    await f.check("Short", high=100.0, low=90.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    result = await f.check("Long", high=100.0, low=95.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert result is False


@pytest.mark.asyncio
async def test_long_invalid_after_short_below(f, sym):
    """Short sonrası long: high < short high → geçersiz."""
    t = _bar_times(2)
    await f.check("Short", high=100.0, low=90.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    result = await f.check("Long", high=99.0, low=95.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert result is False


@pytest.mark.asyncio
async def test_short_valid_after_long(f, sym):
    """Long sonrası short: low < long low → geçerli."""
    t = _bar_times(2)
    await f.check("Long", high=110.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    result = await f.check("Short", high=105.0, low=99.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert result is True


@pytest.mark.asyncio
async def test_short_invalid_after_long(f, sym):
    """Long sonrası short: low >= long low → geçersiz."""
    t = _bar_times(2)
    await f.check("Long", high=110.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    result = await f.check("Short", high=105.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert result is False


@pytest.mark.asyncio
async def test_sequence_long_short_long(f, sym):
    """Long → Short (geçerli) → Long (geçerli) ardışık dizi."""
    t = _bar_times(3)
    await f.check("Long", high=110.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    short_valid = await f.check("Short", high=108.0, low=99.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    assert short_valid is True
    long_valid = await f.check("Long", high=109.0, low=97.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[2])
    assert long_valid is True


@pytest.mark.asyncio
async def test_consecutive_short_updates_reference(f, sym):
    """Ardışık iki short: ikinci short, birinci short'un high'ını referans olarak kullanır."""
    t = _bar_times(5)
    # İlk long (state başlatmak için)
    await f.check("Long", high=110.0, low=100.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
    # İlk short — low=99 < long_low=100 → geçerli, last_short_high=105 güncellenir
    await f.check("Short", high=105.0, low=99.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[1])
    # İkinci short — low=98 < long_low=100 → geçerli, last_short_high=107 güncellenir
    await f.check("Short", high=107.0, low=98.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[2])
    # Long: high > last_short_high (107) gerekiyor
    assert await f.check("Long", high=106.0, low=95.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[3]) is False
    assert await f.check("Long", high=108.0, low=95.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[4]) is True


@pytest.mark.asyncio
async def test_independent_keys(f, sym):
    """Farklı symbol/interval/indicator kombinasyonları birbirini etkilemez."""
    t = _bar_times(2)
    other_sym = f"{sym}_OTHER"
    try:
        await f.check("Short", high=100.0, low=90.0, symbol=sym, interval=IV, indicator=IND, bar_time=t[0])
        # other_sym için ayrı state — ilk long geçersiz olmalı
        result = await f.check("Long", high=101.0, low=95.0, symbol=other_sym, interval=IV, indicator=IND, bar_time=t[1])
        assert result is False
    finally:
        async with get_session() as session:
            await session.execute(
                text("DELETE FROM signal_filter_events WHERE symbol = :sym"), {"sym": other_sym}
            )
