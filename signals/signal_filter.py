"""
Sinyal Filtresi — PineScript Supertrend Filtered Signals mantığı.

Kural:
- Long geçerli  : long sinyalinin mumunun high'ı > önceki short sinyalinin high'ı
- Short geçerli : short sinyalinin mumunun low'u  < önceki long sinyalinin low'u
- İlk sinyal (karşı referans yok): her zaman geçersiz (PineScript na koruması)

State key: (symbol, interval, indicator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_REDIS_KEY = "signal_filter_state"


@dataclass
class _FilterState:
    last_short_high: Optional[float] = None
    last_long_low: Optional[float] = None


class SignalFilter:
    """
    Sinyal filtresi — state Redis'e persist edilir, restart-proof.

    Kullanım:
        f = SignalFilter()
        valid = f.check("Long", high=105.0, low=100.0,
                        symbol="BTCUSDT", interval="1h", indicator="RSI_Cross")
    """

    def __init__(self) -> None:
        self._state: dict[tuple, _FilterState] = {}
        self._dirty: bool = False

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
            self._dirty = True
            if prev_long_low is None:
                return False
            return low < prev_long_low

        if signal_type == "Long":
            state.last_long_low = low
            self._dirty = True
            if prev_short_high is None:
                return False
            return high > prev_short_high

        return False

    async def save_to_redis(self, redis_client: Any) -> None:
        if not self._dirty:
            return
        data: Dict[str, Dict[str, Optional[float]]] = {
            f"{k[0]}:{k[1]}:{k[2]}": {
                "last_short_high": v.last_short_high,
                "last_long_low": v.last_long_low,
            }
            for k, v in self._state.items()
        }
        try:
            await redis_client.set_json(_REDIS_KEY, data)
            self._dirty = False
            logger.debug("SignalFilter state Redis'e kaydedildi (%d key)", len(data))
        except Exception as e:
            logger.warning("SignalFilter state kaydetme hatası: %s", e)

    async def load_from_redis(self, redis_client: Any) -> None:
        try:
            data = await redis_client.get_json(_REDIS_KEY)
            if not data:
                logger.info("SignalFilter: Redis'te state yok, sıfırdan başlanıyor.")
                return
            for key_str, vals in data.items():
                parts = key_str.split(":", 2)
                if len(parts) == 3:
                    self._state[tuple(parts)] = _FilterState(
                        last_short_high=vals.get("last_short_high"),
                        last_long_low=vals.get("last_long_low"),
                    )
            logger.info("SignalFilter state Redis'ten yüklendi (%d key)", len(self._state))
        except Exception as e:
            logger.warning("SignalFilter state yükleme hatası: %s", e)

    def reset(self, symbol: str, interval: str, indicator: str) -> None:
        """Belirli bir key'in state'ini sıfırlar."""
        self._state.pop(self._key(symbol, interval, indicator), None)

    def reset_all(self) -> None:
        self._state.clear()
