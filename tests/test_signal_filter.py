"""
SignalFilter unit testleri.

Senaryolar:
- İlk long / ilk short her zaman geçersiz (referans yok)
- Long geçerli: high > önceki short high
- Long geçersiz: high <= önceki short high
- Short geçerli: low < önceki long low
- Short geçersiz: low >= önceki long low
- Ardışık aynı yön: her sinyal kendi referansını günceller
- reset() sonrası state temizlenir
"""

import pytest
from signals.signal_filter import SignalFilter

SYM = "BTCUSDT"
IV = "1h"
IND = "Supertrend"


@pytest.fixture
def f():
    return SignalFilter()


def test_first_long_invalid(f):
    """İlk long sinyali — önceki short high yok, geçersiz."""
    assert f.check("Long", high=105.0, low=100.0, symbol=SYM, interval=IV, indicator=IND) is False


def test_first_short_invalid(f):
    """İlk short sinyali — önceki long low yok, geçersiz."""
    assert f.check("Short", high=105.0, low=100.0, symbol=SYM, interval=IV, indicator=IND) is False


def test_long_valid_after_short(f):
    """Short sonrası long: high > short high → geçerli."""
    f.check("Short", high=100.0, low=90.0, symbol=SYM, interval=IV, indicator=IND)
    result = f.check("Long", high=101.0, low=95.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is True


def test_long_invalid_after_short(f):
    """Short sonrası long: high <= short high → geçersiz."""
    f.check("Short", high=100.0, low=90.0, symbol=SYM, interval=IV, indicator=IND)
    result = f.check("Long", high=100.0, low=95.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is False


def test_long_invalid_after_short_below(f):
    """Short sonrası long: high < short high → geçersiz."""
    f.check("Short", high=100.0, low=90.0, symbol=SYM, interval=IV, indicator=IND)
    result = f.check("Long", high=99.0, low=95.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is False


def test_short_valid_after_long(f):
    """Long sonrası short: low < long low → geçerli."""
    f.check("Long", high=110.0, low=100.0, symbol=SYM, interval=IV, indicator=IND)
    result = f.check("Short", high=105.0, low=99.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is True


def test_short_invalid_after_long(f):
    """Long sonrası short: low >= long low → geçersiz."""
    f.check("Long", high=110.0, low=100.0, symbol=SYM, interval=IV, indicator=IND)
    result = f.check("Short", high=105.0, low=100.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is False


def test_sequence_long_short_long(f):
    """Long → Short (geçerli) → Long (geçerli) ardışık dizi."""
    f.check("Long", high=110.0, low=100.0, symbol=SYM, interval=IV, indicator=IND)
    short_valid = f.check("Short", high=108.0, low=99.0, symbol=SYM, interval=IV, indicator=IND)
    assert short_valid is True
    long_valid = f.check("Long", high=109.0, low=97.0, symbol=SYM, interval=IV, indicator=IND)
    assert long_valid is True


def test_consecutive_short_updates_reference(f):
    """Ardışık iki short: ikinci short, birinci short'un high'ını referans olarak kullanır."""
    # İlk long (state başlatmak için)
    f.check("Long", high=110.0, low=100.0, symbol=SYM, interval=IV, indicator=IND)
    # İlk short — low=99 < long_low=100 → geçerli, last_short_high=105 güncellenir
    f.check("Short", high=105.0, low=99.0, symbol=SYM, interval=IV, indicator=IND)
    # İkinci short — low=98 < long_low=100 → geçerli, last_short_high=107 güncellenir
    f.check("Short", high=107.0, low=98.0, symbol=SYM, interval=IV, indicator=IND)
    # Long: high > last_short_high (107) gerekiyor
    assert f.check("Long", high=106.0, low=95.0, symbol=SYM, interval=IV, indicator=IND) is False
    assert f.check("Long", high=108.0, low=95.0, symbol=SYM, interval=IV, indicator=IND) is True


def test_independent_keys(f):
    """Farklı symbol/interval/indicator kombinasyonları birbirini etkilemez."""
    f.check("Short", high=100.0, low=90.0, symbol="BTCUSDT", interval="1h", indicator=IND)
    # ETHUSDT için ayrı state — ilk long geçersiz olmalı
    result = f.check("Long", high=101.0, low=95.0, symbol="ETHUSDT", interval="1h", indicator=IND)
    assert result is False


def test_reset_clears_state(f):
    """reset() sonrası ilk sinyal tekrar geçersiz olur."""
    f.check("Short", high=100.0, low=90.0, symbol=SYM, interval=IV, indicator=IND)
    f.check("Long", high=101.0, low=95.0, symbol=SYM, interval=IV, indicator=IND)
    f.reset(SYM, IV, IND)
    result = f.check("Long", high=999.0, low=1.0, symbol=SYM, interval=IV, indicator=IND)
    assert result is False
