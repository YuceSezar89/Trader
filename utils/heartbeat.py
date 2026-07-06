"""Bileşen bazlı heartbeat: son başarılı çalışma zamanını Redis'e yazar,
uzun süre güncellenmeyen bileşenler için Telegram alarmı üretir."""

import asyncio
from datetime import datetime

from utils.logger import get_logger
from utils.redis_client import RedisClient
from utils.telegram_notify import send_telegram_message

logger = get_logger("Heartbeat")

_HEARTBEAT_PREFIX = "heartbeat:"
_alerted: set[str] = set()


async def beat(component: str) -> None:
    """En iyi çaba (best-effort) — Redis geçici olarak meşgulse (ör. başlangıç
    burst'ü) sessizce loglayıp geçer, çağıranı asla etkilemez.

    Kısa timeout ŞART: paylaşımlı pool tıkanırsa (bilinen risk, redis_client.py'de
    socket_timeout yok) çağıran döngüyü (ör. _batch_flush_loop, 500ms periyodu)
    onlarca saniye bekletmemek için."""
    try:
        client = RedisClient.get_client()
        await asyncio.wait_for(
            client.set(f"{_HEARTBEAT_PREFIX}{component}", datetime.now().isoformat()),
            timeout=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("beat(%s) yazılamadı: %s", component, e)


async def _read_all() -> dict[str, datetime]:
    client = RedisClient.get_client()
    result: dict[str, datetime] = {}
    try:
        keys = await asyncio.wait_for(client.keys(f"{_HEARTBEAT_PREFIX}*"), timeout=3)
        for key in keys:
            raw = await asyncio.wait_for(client.get(key), timeout=3)
            if not raw:
                continue
            try:
                result[key[len(_HEARTBEAT_PREFIX):]] = datetime.fromisoformat(raw)
            except ValueError:
                continue
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("_read_all okunamadı: %s", e)
    return result


async def watchdog_loop(max_age_seconds: dict[str, int], check_interval: int = 60) -> None:
    """max_age_seconds: {bileşen_adı: izin verilen maksimum bayatlık (saniye)}."""
    logger.info("Heartbeat watchdog başlatıldı: %s", max_age_seconds)

    # Restart öncesinden kalan eski heartbeat değerlerini temizle — yoksa ilk
    # kontrolde yanlış "bayat" alarmı üretilir (yeni process henüz kendi beat()
    # çağrısını yapmadan, Redis'teki eski değer okunur).
    try:
        client = RedisClient.get_client()
        for component in max_age_seconds:
            await asyncio.wait_for(client.delete(f"{_HEARTBEAT_PREFIX}{component}"), timeout=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("Başlangıç heartbeat temizliği başarısız: %s", e)

    while True:
        try:
            heartbeats = await _read_all()
            now = datetime.now()
            for component, limit_secs in max_age_seconds.items():
                last = heartbeats.get(component)
                if last is None:
                    continue
                age = (now - last).total_seconds()
                if age > limit_secs and component not in _alerted:
                    _alerted.add(component)
                    logger.error("[Heartbeat] %s bayat: %.0fs (limit %ds)", component, age, limit_secs)
                    await send_telegram_message(
                        f"⚠️ {component} {int(age)}s'dir güncellenmiyor (limit: {limit_secs}s)"
                    )
                elif age <= limit_secs and component in _alerted:
                    _alerted.discard(component)
                    logger.info("[Heartbeat] %s normale döndü", component)
                    await send_telegram_message(f"✅ {component} normale döndü")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Heartbeat watchdog hatası: %s", e, exc_info=True)
        await asyncio.sleep(check_interval)
