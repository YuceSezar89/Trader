"""
Sinyal Filtresi — PineScript Supertrend Filtered Signals mantığı.

Kural:
- Long geçerli  : long sinyalinin mumunun high'ı > önceki short sinyalinin high'ı
- Short geçerli : short sinyalinin mumunun low'u  < önceki long sinyalinin low'u
- İlk sinyal (karşı referans yok): her zaman geçersiz (PineScript na koruması)

Referans noktaları signal_filter_events tablosunda tutulur (bkz. migration 016)
— her deneme (kabul/red fark etmeksizin) buraya loglanır. Önceki tasarım
(in-memory dict + Redis persist) restart'ta ve iki process (live_data_manager +
signal_service) arasında senkron/staleness sorunlarına yol açıyordu; DB tek
kaynak olduğu için bu sorun yapısal olarak ortadan kalkar.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from database.engine import get_session

logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Sinyal filtresi — referans noktaları DB'den (signal_filter_events) okunur.

    Kullanım:
        f = SignalFilter()
        valid = await f.check("Long", high=105.0, low=100.0,
                              symbol="BTCUSDT", interval="1h", indicator="RSI_Cross",
                              bar_time=datetime.now())
    """

    async def check(
        self,
        signal_type: str,
        high: float,
        low: float,
        symbol: str,
        interval: str,
        indicator: str,
        bar_time: datetime,
    ) -> bool:
        """
        Sinyalin geçerli olup olmadığını DB'den sorgulayarak döner ve bu
        denemeyi (kabul/red fark etmeksizin) signal_filter_events'e kaydeder.
        """
        if signal_type not in ("Long", "Short"):
            return False

        opposite = "Long" if signal_type == "Short" else "Short"

        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT high, low FROM signal_filter_events
                        WHERE symbol = :symbol AND interval = :interval
                          AND indicator = :indicator AND signal_type = :opposite
                        ORDER BY bar_time DESC LIMIT 1
                    """),
                    {"symbol": symbol, "interval": interval, "indicator": indicator, "opposite": opposite},
                )
                row = result.fetchone()

                if signal_type == "Short":
                    passed = row is not None and low < row[1]
                else:
                    passed = row is not None and high > row[0]

                await session.execute(
                    text("""
                        INSERT INTO signal_filter_events
                        (symbol, interval, indicator, signal_type, high, low, passed, bar_time, created_at)
                        VALUES (:symbol, :interval, :indicator, :signal_type, :high, :low, :passed, :bar_time, :created_at)
                    """),
                    {
                        "symbol": symbol, "interval": interval, "indicator": indicator,
                        "signal_type": signal_type, "high": high, "low": low,
                        "passed": passed, "bar_time": bar_time, "created_at": datetime.now(),
                    },
                )
            return passed
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "SignalFilter DB hatası [%s:%s:%s]: %s — fail-closed (reddedildi)",
                symbol, interval, indicator, e,
            )
            return False

    async def last_reference(
        self, symbol: str, interval: str, indicator: str, signal_type: str
    ) -> Optional[tuple[float, float]]:
        """Diagnostik/görselleştirme amaçlı: bu key için en son (high, low) olayını döner."""
        async with get_session() as session:
            result = await session.execute(
                text("""
                    SELECT high, low FROM signal_filter_events
                    WHERE symbol = :symbol AND interval = :interval
                      AND indicator = :indicator AND signal_type = :signal_type
                    ORDER BY bar_time DESC LIMIT 1
                """),
                {"symbol": symbol, "interval": interval, "indicator": indicator, "signal_type": signal_type},
            )
            row = result.fetchone()
            return (row[0], row[1]) if row else None

    async def cleanup(self, symbol: str, interval: str, indicator: str) -> None:
        """Bir (symbol, interval, indicator) key'inin tüm olaylarını siler —
        analiz/backtest scriptlerinin (ör. scripts/compare_filter_output.py)
        kendi izole symbol'lerini temizlemesi için, canlı verilere dokunmaz."""
        async with get_session() as session:
            await session.execute(
                text("""
                    DELETE FROM signal_filter_events
                    WHERE symbol = :symbol AND interval = :interval AND indicator = :indicator
                """),
                {"symbol": symbol, "interval": interval, "indicator": indicator},
            )
