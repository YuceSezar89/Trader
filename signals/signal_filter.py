"""
Sinyal Filtresi — PineScript Supertrend Filtered Signals mantığı.

Kural:
- Long geçerli  : long sinyalinin mumunun high'ı > önceki short sinyalinin high'ı
- Short geçerli : short sinyalinin mumunun low'u  < önceki long sinyalinin low'u
- İlk sinyal (karşı referans yok): her zaman geçersiz (PineScript na koruması)

State key: (symbol, interval, indicator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class _FilterState:
    last_short_high: Optional[float] = None
    last_long_low: Optional[float] = None


class SignalFilter:
    """
    Bağımsız, in-memory sinyal filtresi.

    Kullanım:
        f = SignalFilter()
        valid = f.check("Long", high=105.0, low=100.0,
                        symbol="BTCUSDT", interval="1h", indicator="RSI_Cross")
    """

    def __init__(self) -> None:
        self._state: dict[tuple, _FilterState] = {}

    def _key(self, symbol: str, interval: str, indicator: str) -> tuple:
        return (symbol, interval, indicator)

    def check(
        self,
        signal_type: str,
        high: float,
        low: float,
        symbol: str,
        interval: str,
        indicator: str,
    ) -> bool:
        """
        Sinyalin geçerli olup olmadığını döner ve state'i günceller.

        PineScript sırasını korur:
          1. Önceki referans değerleri yakala (prev*)
          2. State'i güncelle
          3. Filtre koşulunu kontrol et (prev* kullanılır)
        """
        key = self._key(symbol, interval, indicator)
        if key not in self._state:
            self._state[key] = _FilterState()

        state = self._state[key]

        prev_short_high = state.last_short_high
        prev_long_low = state.last_long_low

        if signal_type == "Short":
            state.last_short_high = high
            if prev_long_low is None:
                return False
            return low < prev_long_low

        if signal_type == "Long":
            state.last_long_low = low
            if prev_short_high is None:
                return False
            return high > prev_short_high

        return False

    def reset(self, symbol: str, interval: str, indicator: str) -> None:
        """Belirli bir key'in state'ini sıfırlar."""
        self._state.pop(self._key(symbol, interval, indicator), None)

    def reset_all(self) -> None:
        self._state.clear()
