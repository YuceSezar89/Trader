"""
utils/data_health.py — kritik tablolarda (signals/paper_trades) beklenmedik
NULL oranını izler (Fable 5 mimari denetimi madde 4, 2 Ağu 2026: mevcut
heartbeat/throughput altyapısı sadece "süreç canlı mı" sorusuna cevap
veriyordu, "veri doğru yazılıyor mu" sorusuna değil).

database/signal_repository.py::close_signal() ve
signals/paper_trade_manager.py::PaperTradeManager._apply_close() bir
sinyal/paper-trade kapanırken bir grup alanı BİRLİKTE, TEK yerden set eder —
status='closed' olduğu hâlde bunlardan biri NULL'sa bu ya yarım kalmış bir
yazma ya da helper'ı atlayan alternatif bir kod yolu demektir.

NOT: Signal.duration_minutes KASITLI OLARAK kontrol edilmiyor — kod tabanında
hiçbir yerde set edilmiyor (ölü kolon, close_signal() bile atamıyor), her
zaman NULL olması beklenen bir durum, "bozuk" değil.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import text

from database.engine import get_session, run_with_db_timeout
from utils.telegram_notify import send_telegram_message

logger = logging.getLogger(__name__)

_alerted: set = set()

# SQL now() KULLANILMAZ: DB session'ı UTC'de (database/engine.py connect_args
# timezone="UTC"), ama closed_at KOLONU CLAUDE.md'nin datetime kuralı gereği
# Python'un yerel (İstanbul, UTC+3) datetime.now()'uyla yazılıyor — now()'a
# karşı karşılaştırma ~3 saatlik kaymayla yanlış pencere sonucu verir (testle
# doğrulandı, bkz. tests/test_data_health.py). Kesme zamanı Python'da
# hesaplanıp parametre olarak geçirilir.
_SIGNAL_CLOSED_NULL_SQL = """
    SELECT COUNT(*) AS total,
           COUNT(*) FILTER (
               WHERE close_reason IS NULL OR realized_pnl IS NULL
                  OR close_price IS NULL OR closed_at IS NULL
           ) AS null_count
    FROM signals
    WHERE status = 'closed'
      AND symbol LIKE :symbol_pattern
      AND closed_at >= :cutoff
"""

_PAPER_TRADE_CLOSED_NULL_SQL = """
    SELECT COUNT(*) AS total,
           COUNT(*) FILTER (
               WHERE close_reason IS NULL OR pnl_usd IS NULL
                  OR pnl_pct IS NULL OR exit_price IS NULL OR closed_at IS NULL
           ) AS null_count
    FROM paper_trades
    WHERE status = 'closed'
      AND symbol LIKE :symbol_pattern
      AND closed_at >= :cutoff
"""

_CHECKS = (
    ("signals_closed", _SIGNAL_CLOSED_NULL_SQL),
    ("paper_trades_closed", _PAPER_TRADE_CLOSED_NULL_SQL),
)


async def check_null_ratios(window_minutes: int = 60, symbol_pattern: str = "%") -> dict:
    """Son `window_minutes` içinde kapanmış signals/paper_trades satırlarında
    "kapanırken birlikte set edilmesi gereken" alanların NULL oranını döner:
    {check_name: (null_count, total, ratio)}. total=0 ise ratio=0.0 (veri yok,
    alarm değil). symbol_pattern: SQL LIKE deseni (testlerde 'TEST%' ile
    izolasyon için)."""

    cutoff = datetime.now() - timedelta(minutes=window_minutes)

    async def _do_check(sql: str) -> tuple:
        async with get_session() as session:
            result = await session.execute(
                text(sql),
                {"cutoff": cutoff, "symbol_pattern": symbol_pattern},
            )
            row = result.fetchone()
            return row[0], row[1]

    results = {}
    for name, sql in _CHECKS:
        total, null_count = await run_with_db_timeout(_do_check(sql))
        ratio = (null_count / total) if total else 0.0
        results[name] = (null_count, total, ratio)
    return results


async def data_health_loop(
    check_interval: int = 300,
    window_minutes: int = 60,
    warn_ratio: float = 0.1,
    min_total: int = 5,
) -> None:
    """Periyodik olarak check_null_ratios() çağırır, eşiği aşan her check için
    Telegram alarmı verir (utils/heartbeat.py::watchdog_loop ile aynı histerezis
    deseni: eşik-aşımında bir kez alarm, normale dönüşte bir kez "düzeldi",
    tekrar tekrar spam etmez). min_total: örneklem bu sayının altındaysa
    (ör. sakin gece saatleri) karar verilmez — küçük örneklemde %50 gibi
    yanıltıcı oranlar yanlış alarm üretmesin diye."""
    logger.info(
        "[DataHealth] izleyici başlatıldı: pencere=%ddk, eşik=%%%.0f, min_örneklem=%d",
        window_minutes,
        warn_ratio * 100,
        min_total,
    )
    while True:
        try:
            results = await check_null_ratios(window_minutes=window_minutes)
            for name, (null_count, total, ratio) in results.items():
                if total < min_total:
                    continue
                if ratio > warn_ratio and name not in _alerted:
                    _alerted.add(name)
                    logger.error(
                        "[DataHealth] %s: %d/%d satırda beklenen alan NULL (%%%.1f, eşik %%%.0f)",
                        name,
                        null_count,
                        total,
                        ratio * 100,
                        warn_ratio * 100,
                    )
                    await send_telegram_message(
                        f"⚠️ {name}: {null_count}/{total} satırda beklenen alan NULL "
                        f"(%{ratio * 100:.1f}, eşik %{warn_ratio * 100:.0f})"
                    )
                elif ratio <= warn_ratio and name in _alerted:
                    _alerted.discard(name)
                    logger.info("[DataHealth] %s normale döndü", name)
                    await send_telegram_message(f"✅ {name} normale döndü")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[DataHealth] kontrol hatası: %s", e, exc_info=True)
        await asyncio.sleep(check_interval)
