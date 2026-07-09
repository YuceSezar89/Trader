"""Signal/Feature servisi — Faz 4: dry-run modda çalışır, gerçek sinyal işleme
yolunu (process_and_enrich_signals) çağırır ama dry_run=True ile DB'ye yazmaz,
paper trade tetiklemez. Mevcut (DB'ye yazan) yol live_data_manager.py'de hâlâ
aktif — cutover (Faz 4 Adım 7) ayrı bir feature flag ile, ayrı onayla yapılacak.
"""

import asyncio
import functools
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from utils.logger import get_logger
from utils.redis_client import RedisClient, SAFE_EXTERNAL_TIMEOUT
from utils.heartbeat import beat, watchdog_loop, record_activity, throughput_watchdog_loop
from indicators.financial_metrics import calculate_metrics
from signals.signal_processor import process_and_enrich_signals
from signals.risk_manager import risk_manager
from signals.paper_trade_manager import (
    paper_trade_manager, ha_cross_manager, rsi_15m_manager, manual_manager, do_kirilimi_manager,
)
from config import Config

logger = get_logger("SignalService")

_STREAM = "kline_closed"
_GROUP = "signal_service"
_CONSUMER = "signal_service-1"
_PID_FILE = "signal_service.pid"
_METRICS_POOL_WORKERS = 5
_IDEMPOTENCY_TTL = 3600  # saniye — aynı bar'ın crash/redelivery sonrası tekrar işlenmesini bu pencerede engeller
_QUEUE_LAG_KEY = "metrics:signal_service:pending"
_QUEUE_LAG_CHECK_INTERVAL = 30  # saniye
_QUEUE_LAG_WARN_THRESHOLD = 500  # bu sayıyı aşarsa tüketim üretimin gerisinde kalıyor demektir
_CONCURRENCY = 10  # aynı anda işlenecek maksimum event sayısı (5 process-pool worker + I/O örtüşmesi payı)
_CLAIM_IDLE_MS = 60_000  # bu süreden uzun süredir ack'lenmemiş mesajlar crash sonrası sahipsiz sayılır
_CLAIM_CHECK_INTERVAL = 30  # saniye

_metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)
_process_semaphore = asyncio.Semaphore(_CONCURRENCY)


async def _calculate_metrics_via_pool(
    df_prepared: pd.DataFrame, ref_df_prepared: pd.DataFrame, interval: str
) -> pd.DataFrame:
    """calculate_metrics'i ProcessPoolExecutor'da çalıştırır. Pool çökerse (BrokenProcessPool)
    pool yeniden oluşturulur VE aynı event senkron olarak (yavaş ama veri kaybetmeden)
    hesaplanır — process_and_enrich_signals artık gerçek DB yazımı/paper trade tetiklemesi
    içerdiği için bu event'in sessizce kaybolması kabul edilebilir değil (dry_run/gölge
    dönemindeki eski davranıştan bilinçli fark)."""
    global _metrics_pool  # pylint: disable=global-statement
    loop = asyncio.get_running_loop()
    fn = functools.partial(calculate_metrics, df_prepared, ref_df_prepared, interval=interval)
    try:
        return await loop.run_in_executor(_metrics_pool, fn)
    except BrokenProcessPool as e:
        logger.warning(
            "ENDİŞE: Metrics process pool çöktü — pool yeniden oluşturuluyor, "
            "bu event senkron fallback ile hesaplanıyor (yavaş yol): %s", e,
        )
        _metrics_pool.shutdown(wait=False)
        _metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)
        return calculate_metrics(df_prepared, ref_df_prepared, interval=interval)


async def _queue_lag_loop() -> None:
    """kline_closed stream'inde signal_service grubunun bekleyen (ack'lenmemiş) mesaj
    sayısını periyodik ölçüp Redis'e + loga yazar — Faz 4 Adım 6'daki dry-run gözlem
    penceresinde tüketimin üretimin gerisinde kalıp kalmadığını görünür kılar."""
    client = RedisClient.get_client()
    while True:
        try:
            info = await asyncio.wait_for(client.xpending(_STREAM, _GROUP), timeout=SAFE_EXTERNAL_TIMEOUT)
            count = info.get("pending", 0) if isinstance(info, dict) else 0
            await asyncio.wait_for(client.set(_QUEUE_LAG_KEY, count, ex=120), timeout=SAFE_EXTERNAL_TIMEOUT)
            if count > _QUEUE_LAG_WARN_THRESHOLD:
                logger.warning(
                    "ENDİŞE: kuyruk gecikmesi yüksek — %d mesaj bekliyor (grup=%s, eşik=%d)",
                    count, _GROUP, _QUEUE_LAG_WARN_THRESHOLD,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Kuyruk gecikmesi ölçülemedi: %s", e)
        await asyncio.sleep(_QUEUE_LAG_CHECK_INTERVAL)


async def _ensure_group(client) -> None:
    try:
        await client.xgroup_create(_STREAM, _GROUP, id="$", mkstream=True)
        logger.info("Consumer group '%s' oluşturuldu", _GROUP)
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group '%s' zaten var", _GROUP)
        else:
            raise


async def _claim_event(symbol: str, interval: str, open_time: str) -> bool:
    """Aynı olayın (crash sonrası stream redelivery ile) iki kez işlenip DB'de çift
    kayıt/çift paper-trade pozisyonu açmasını önler. SET NX atomik: True dönerse bu
    çağıran işlemeyi üstlenir, False dönerse zaten işlenmiş, atlanır.

    ENDİŞE: Redis kontrolünün kendisi başarısız olursa fail-open (işlemeye devam)
    davranıyoruz — gerçek sinyali sessizce kaybetmek, nadir bir çift-işleme riskinden
    daha kötü kabul edildi. Bu bilinçli bir tercih, "en güvenli" değil."""
    key = f"processed:signal_service:{symbol}:{interval}:{open_time}"
    try:
        client = RedisClient.get_client()
        claimed = await asyncio.wait_for(client.set(key, "1", nx=True, ex=_IDEMPOTENCY_TTL), timeout=SAFE_EXTERNAL_TIMEOUT)
        return bool(claimed)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "ENDİŞE: idempotency kontrolü başarısız [%s:%s:%s], fail-open ile işleniyor: %s",
            symbol, interval, open_time, e,
        )
        return True


async def _incr_metric(key: str) -> None:
    """Dosya loguna alternatif, rotasyon yarışından bağımsız sayaç — atomik INCR,
    log dosyalarının çoklu process çakışmasıyla satır kaybedebildiği tespit
    edildiği için (ProcessPoolExecutor worker'ları) ground-truth ölçüm burada."""
    try:
        await asyncio.wait_for(RedisClient.get_client().incr(key), timeout=SAFE_EXTERNAL_TIMEOUT)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


async def _process_event(fields: dict) -> None:
    symbol = fields["symbol"]
    interval = fields["interval"]
    open_time = fields.get("open_time", "")

    await _incr_metric("metrics:sigsvc:invocation")
    record_activity("signal_service")

    if not await _claim_event(symbol, interval, open_time):
        logger.debug("[%s] %s open_time=%s zaten işlenmiş, atlanıyor (idempotency)", symbol, interval, open_time)
        await _incr_metric("metrics:sigsvc:idempotency_skip")
        return

    df = await RedisClient.get_mtf_klines(symbol, interval)
    ref_df = await RedisClient.get_mtf_klines(Config.MARKET_REFERENCE_SYMBOL, interval)
    if df is None or ref_df is None or df.empty or ref_df.empty:
        logger.debug("[%s] %s buffer eksik, atlanıyor", symbol, interval)
        await _incr_metric("metrics:sigsvc:buffer_eksik")
        return

    oi_data_json = None
    try:
        oi_data_json = await asyncio.wait_for(RedisClient.get_client().get(f"oi:{symbol}"), timeout=SAFE_EXTERNAL_TIMEOUT)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("oi_data okunamadı [%s]: %s", symbol, e)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await process_and_enrich_signals(
        symbol, df, ref_df, interval,
        oi_data=oi_data_json,
        metrics_calculator=_calculate_metrics_via_pool,
        dry_run=(Config.SIGNAL_SOURCE != "yeni"),
    )
    elapsed_ms = (loop.time() - t0) * 1000
    logger.info("[DRY-RUN] [%s] %s işlendi (%.1fms)", symbol, interval, elapsed_ms)


async def _handle_message(client, msg_id: str, fields: dict) -> None:
    """Tek bir event'i semaphore ile sınırlı eşzamanlılıkta işler, sonunda ack eder.
    xack'in kendisi başarısız olursa mesaj PEL'de kalır — _reclaim_stale_loop onu
    daha sonra geri alıp yeniden dener (idempotency zaten çift-işlemeyi önlüyor)."""
    async with _process_semaphore:
        try:
            await _process_event(fields)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[%s] event işleme hatası: %s", fields, e, exc_info=True)
        finally:
            try:
                await asyncio.wait_for(client.xack(_STREAM, _GROUP, msg_id), timeout=SAFE_EXTERNAL_TIMEOUT)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("ENDİŞE: xack başarısız [%s], mesaj PEL'de kalıp reclaim'e düşecek: %s", msg_id, e)


async def _consume_loop() -> None:
    client = RedisClient.get_client()
    await _ensure_group(client)
    logger.info(
        "Signal service tüketimi başladı (grup=%s, consumer=%s, eşzamanlılık=%d)",
        _GROUP, _CONSUMER, _CONCURRENCY,
    )

    while True:
        try:
            resp = await asyncio.wait_for(
                client.xreadgroup(_GROUP, _CONSUMER, {_STREAM: ">"}, count=50, block=2000),
                timeout=5,
            )
            if not resp:
                await beat("signal_service")
                continue
            handlers = [
                asyncio.create_task(_handle_message(client, msg_id, fields))
                for _stream_name, messages in resp
                for msg_id, fields in messages
            ]
            if handlers:
                await asyncio.gather(*handlers)
            await beat("signal_service")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Consume loop hatası: %s", e, exc_info=True)
            await asyncio.sleep(5)


async def _reclaim_stale_loop() -> None:
    """XAUTOCLAIM ile crash sonrası PEL'de (pending entries list) sahipsiz kalmış
    mesajları geri alıp yeniden işler. Önceden bu servis çökerse (xack'e ulaşmadan)
    o mesaj sonsuza dek PEL'de kalır, kimse yeniden denemezdi — idempotency
    (_claim_event) çift-işlemeyi zaten önlediği için reclaim güvenlidir."""
    client = RedisClient.get_client()
    cursor = "0-0"
    while True:
        try:
            cursor, claimed, _deleted = await asyncio.wait_for(
                client.xautoclaim(
                    _STREAM, _GROUP, _CONSUMER, min_idle_time=_CLAIM_IDLE_MS, start_id=cursor, count=50,
                ),
                timeout=5,
            )
            if claimed:
                logger.warning(
                    "ENDİŞE: %d sahipsiz (muhtemelen crash sonrası) mesaj geri alınıp yeniden işleniyor",
                    len(claimed),
                )
            for msg_id, fields in claimed:
                await _handle_message(client, msg_id, fields)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Stale mesaj kurtarma hatası: %s", e)
        await asyncio.sleep(_CLAIM_CHECK_INTERVAL)


async def _supervised(coro, name: str) -> None:
    """Bir görev beklenmedik şekilde patlarsa diğer görevleri etkilememesi için izole eder —
    yoksa asyncio.wait(..., FIRST_EXCEPTION) tek bir görevin hatasında tüm servisi kapatır."""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[%s] görev beklenmedik şekilde sonlandı, izole edildi: %s", name, e, exc_info=True)


async def run_all() -> None:
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Gözlem sayaçlarını her başlangıçta sıfırla — bu process'in ölçtüğü pencere
    # net olsun diye (bkz. _incr_metric).
    try:
        client = RedisClient.get_client()
        await asyncio.wait_for(
            client.delete(
                "metrics:sigsvc:invocation", "metrics:sigsvc:idempotency_skip",
                "metrics:sigsvc:buffer_eksik", "metrics:sigsvc:dry_run_signal",
            ),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Gözlem sayaçları sıfırlanamadı: %s", e)

    # Açık pozisyon/risk state'i belleğe yükle — live_data_manager.py:main() ile
    # aynı desen, yoksa cutover anında bu servis boş state ile başlar.
    await risk_manager.load_active_symbols()
    await paper_trade_manager.load_open_symbols()
    await ha_cross_manager.load_open_symbols()
    await rsi_15m_manager.load_open_symbols()
    await manual_manager.load_open_symbols()
    await do_kirilimi_manager.load_open_symbols()

    consume_task = asyncio.create_task(
        _supervised(_consume_loop(), "signal_service_consume"), name="signal_service_consume"
    )
    watchdog_task = asyncio.create_task(
        _supervised(
            watchdog_loop(max_age_seconds={"signal_service": 120}), "signal_service_watchdog"
        ),
        name="signal_service_watchdog",
    )
    queue_lag_task = asyncio.create_task(
        _supervised(_queue_lag_loop(), "signal_service_queue_lag"), name="signal_service_queue_lag"
    )
    reclaim_task = asyncio.create_task(
        _supervised(_reclaim_stale_loop(), "signal_service_reclaim"), name="signal_service_reclaim"
    )
    throughput_task = asyncio.create_task(
        _supervised(
            throughput_watchdog_loop(min_expected={"signal_service": 1}),
            "signal_service_throughput",
        ),
        name="signal_service_throughput",
    )
    tasks = {consume_task, watchdog_task, queue_lag_task, reclaim_task, throughput_task}

    def _handler(sig_name: str) -> None:
        logger.info("Sinyal alındı: %s. Signal service kapanıyor...", sig_name)
        for t in list(tasks):
            if not t.done():
                t.cancel()
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig.name)  # pylint: disable=no-member
        except NotImplementedError:
            pass

    logger.info("Signal service başlatıldı (dry-run mod — DB'ye yazmıyor, paper trade tetiklemiyor).")
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        pass
    finally:
        for t in list(tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Signal service kapandı.")


def _acquire_pid_lock() -> bool:
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            logger.error("signal_service zaten çalışıyor (PID %d). Yeni instance başlatılmıyor.", old_pid)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


def _release_pid_lock() -> None:
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass


if __name__ == "__main__":
    if not _acquire_pid_lock():
        raise SystemExit(0)
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
    finally:
        _release_pid_lock()
