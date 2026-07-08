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

import asyncio
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from database.engine import get_session

logger = logging.getLogger(__name__)

# database/engine.py'nin kendi pool_timeout=30/command_timeout=30'u var ama bu
# check() HER sinyal değerlendirmesinde (hot path) çağrılıyor — bu iç timeout'lar
# beklenmedik şekilde tetiklenmezse (ör. ağ kesintisi sırasında pool bağlantısı
# yarım kalmış bir durumda askıda kalırsa) dıştan bağımsız bir üst sınır olmalı.
# 8 Tem gece: hem live_data_manager hem signal_service'in sinyal değerlendirme
# döngüsü, bu çağrının hiç dışarıdan zaman aşımı olmaması yüzünden saatlerce
# (13:40-18:20) sessizce askıda kaldı — heartbeat bile bunu yakalayamadı çünkü
# heartbeat ayrı bir task. Bu haftaki Redis çağrılarına eklenen
# SAFE_EXTERNAL_TIMEOUT dersinin DB tarafına uygulanmamış hâliydi.
_DB_TIMEOUT = 5.0


async def _run_with_timeout(coro):
    try:
        return await asyncio.wait_for(coro, timeout=_DB_TIMEOUT)
    except asyncio.TimeoutError:
        raise TimeoutError(f"SignalFilter DB çağrısı {_DB_TIMEOUT}s içinde tamamlanmadı") from None


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

        async def _do_check() -> bool:
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

        try:
            return await _run_with_timeout(_do_check())
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
        async def _do_query():
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

        return await _run_with_timeout(_do_query())

    async def cleanup(self, symbol: str, interval: str, indicator: str) -> None:
        """Bir (symbol, interval, indicator) key'inin tüm olaylarını siler —
        analiz/backtest scriptlerinin (ör. scripts/compare_filter_output.py)
        kendi izole symbol'lerini temizlemesi için, canlı verilere dokunmaz."""
        async def _do_delete() -> None:
            async with get_session() as session:
                await session.execute(
                    text("""
                        DELETE FROM signal_filter_events
                        WHERE symbol = :symbol AND interval = :interval AND indicator = :indicator
                    """),
                    {"symbol": symbol, "interval": interval, "indicator": indicator},
                )

        await _run_with_timeout(_do_delete())
