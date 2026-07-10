"""Bileşen bazlı heartbeat: son başarılı çalışma zamanını Redis'e yazar,
uzun süre güncellenmeyen bileşenler için Telegram alarmı üretir."""

import asyncio
import os
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis

from config import Config
from utils.logger import get_logger
from utils.redis_client import RedisClient
from utils.telegram_notify import send_telegram_message

logger = get_logger("Heartbeat")

_HEARTBEAT_PREFIX = "heartbeat:"
_THROUGHPUT_PREFIX = "throughput:"
_alerted: set[str] = set()
_activity_counts: dict[str, int] = {}

# Paylaşımlı ana pool'dan bağımsız, kendi bağlantısı — iptal ortasında
# zehirlenen paylaşımlı pool bağlantısının heartbeat'i sonsuza dek askıda
# bırakmasını önler (_price_publish_loop'taki 3 Tem çözümüyle aynı desen).
_dedicated_conn: Optional[aioredis.Redis] = None


def _get_dedicated_connection() -> aioredis.Redis:
    global _dedicated_conn  # pylint: disable=global-statement
    if _dedicated_conn is None:
        _dedicated_conn = aioredis.from_url(
            Config.REDIS_URL, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5,
        )
    return _dedicated_conn


async def _reset_dedicated_connection() -> None:
    global _dedicated_conn  # pylint: disable=global-statement
    try:
        if _dedicated_conn is not None:
            await _dedicated_conn.aclose()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    _dedicated_conn = None


async def beat(component: str) -> None:
    """En iyi çaba (best-effort) — Redis geçici olarak meşgulse (ör. başlangıç
    burst'ü) sessizce loglayıp geçer, çağıranı asla etkilemez.

    Kısa timeout ŞART, ama artık paylaşımlı pool'a değil kendi bağlantısına
    karşı: paylaşımlı pool'daki bir çağrının iptal ortasında bağlantı
    sızdırması heartbeat'i etkilemesin diye izole edildi."""
    try:
        client = _get_dedicated_connection()
        await asyncio.wait_for(
            client.set(f"{_HEARTBEAT_PREFIX}{component}", datetime.now().isoformat()),
            timeout=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("beat(%s) yazılamadı, bağlantı yenilenecek: %s", component, e)
        await _reset_dedicated_connection()


def record_activity(component: str) -> None:
    """Bellek-içi sayaç — bir bileşenin GERÇEKTEN iş yaptığının kanıtı. beat()'ten
    farkı: beat() ayrı, hafif bir asyncio task'tan çağrılır ve asıl işlem döngüsü
    tıkansa bile taze görünmeye devam edebilir (8 Tem vakası — 4.5 saatlik sessiz
    donmada heartbeat hiç bayat olmadı). Bu fonksiyon asıl iş akışının İÇİNDEN
    çağrılmalı (ör. her bar kapanışı/event işleme), senkron ve çok ucuz olmalı."""
    _activity_counts[component] = _activity_counts.get(component, 0) + 1


async def throughput_watchdog_loop(
    min_expected: dict[str, int],
    check_interval: int = 60,
    self_heal_after: int = 0,
) -> None:
    """min_expected: {bileşen_adı: check_interval içinde beklenen minimum
    record_activity() çağrısı}. Sayı bu eşiğin altında kalırsa (heartbeat taze
    olsa bile) asıl iş akışının durmuş olabileceğini loglar — rutin uyarı için
    Telegram alarmı BİLEREK yok, sadece görünürlük (bkz. proje hafızası, 9 Tem kararı).

    self_heal_after: bir bileşen art arda kaç kontrolde eşiğin altında kalırsa
    process'in kendini sonlandıracağı (os._exit(1)) — launchd KeepAlive.Crashed=true
    bunu yakalayıp otomatik yeniden başlatır. 0 = kapalı (varsayılan, eski davranış).
    Sadece launchd/benzeri bir supervisor altında çalışan servislerde kullanılmalı."""
    logger.info(
        "Throughput watchdog başlatıldı: %s (self_heal_after=%s)",
        min_expected, self_heal_after or "kapalı",
    )
    last_counts = {component: _activity_counts.get(component, 0) for component in min_expected}
    consecutive_breaches = {component: 0 for component in min_expected}
    while True:
        await asyncio.sleep(check_interval)
        for component, min_count in min_expected.items():
            current = _activity_counts.get(component, 0)
            delta = current - last_counts[component]
            last_counts[component] = current
            try:
                client = _get_dedicated_connection()
                await asyncio.wait_for(
                    client.set(f"{_THROUGHPUT_PREFIX}{component}", delta, ex=check_interval * 3),
                    timeout=2,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("throughput(%s) yazılamadı: %s", component, e)
                await _reset_dedicated_connection()
            if delta < min_count:
                consecutive_breaches[component] += 1
                logger.warning(
                    "[Throughput] %s son %ds'de %d işlem yaptı (beklenen >= %d, art arda %d. ihlal) — "
                    "heartbeat taze olsa bile asıl iş akışı durmuş olabilir",
                    component, check_interval, delta, min_count, consecutive_breaches[component],
                )
                if self_heal_after and consecutive_breaches[component] >= self_heal_after:
                    stalled_secs = consecutive_breaches[component] * check_interval
                    logger.critical(
                        "[Throughput] %s %ds'dir 0 işlem yapıyor — self-heal: process "
                        "sonlandırılıyor, launchd yeniden başlatacak", component, stalled_secs,
                    )
                    try:
                        await asyncio.wait_for(
                            send_telegram_message(
                                f"🔴 {component} {stalled_secs}s'dir donmuş görünüyor — "
                                f"self-heal restart tetiklendi"
                            ),
                            timeout=5,
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    os._exit(1)  # pylint: disable=protected-access
            else:
                consecutive_breaches[component] = 0


async def _read_all() -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    try:
        client = _get_dedicated_connection()
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
        await _reset_dedicated_connection()
    return result


async def watchdog_loop(max_age_seconds: dict[str, int], check_interval: int = 60) -> None:
    """max_age_seconds: {bileşen_adı: izin verilen maksimum bayatlık (saniye)}."""
    logger.info("Heartbeat watchdog başlatıldı: %s", max_age_seconds)

    # Restart öncesinden kalan eski heartbeat değerlerini temizle — yoksa ilk
    # kontrolde yanlış "bayat" alarmı üretilir (yeni process henüz kendi beat()
    # çağrısını yapmadan, Redis'teki eski değer okunur).
    try:
        client = _get_dedicated_connection()
        for component in max_age_seconds:
            await asyncio.wait_for(client.delete(f"{_HEARTBEAT_PREFIX}{component}"), timeout=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("Başlangıç heartbeat temizliği başarısız: %s", e)
        await _reset_dedicated_connection()

    while True:
        try:
            pool = RedisClient._get_pool_for_current_loop()  # pylint: disable=protected-access
            logger.info(
                "[Heartbeat] pool durumu: available=%d, in_use=%d, max=%d",
                len(pool._available_connections),  # pylint: disable=protected-access
                len(pool._in_use_connections),  # pylint: disable=protected-access
                pool.max_connections,
            )
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
