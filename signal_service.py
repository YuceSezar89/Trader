"""Signal/Feature servisi — Faz 2: gölge modda çalışır, DB'ye yazmaz.

kline_closed stream'ini (consumer group ile) tüketir, calculate_metrics +
calculate_all_signals'ı çağırıp sonucu sadece loglar. Mevcut (DB'ye yazan) yol
live_data_manager.py'de hâlâ aktif — bu servis sadece paralelde doğrulama içindir.
"""

import asyncio
import functools
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from utils.logger import get_logger
from utils.redis_client import RedisClient
from utils.heartbeat import beat, watchdog_loop
from indicators.financial_metrics import calculate_metrics
from signals.signal_engine import signal_engine
from config import Config

logger = get_logger("SignalService")

_STREAM = "kline_closed"
_GROUP = "signal_service"
_CONSUMER = "signal_service-1"
_PID_FILE = "signal_service.pid"
_METRICS_POOL_WORKERS = 5

_metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)


async def _run_calculate_metrics(df_prepared: pd.DataFrame, ref_df_prepared: pd.DataFrame, interval: str) -> None:
    """calculate_metrics'i ProcessPoolExecutor'da çalıştırır. Pool çökerse (BrokenProcessPool)
    yeniden oluşturur — bu event kaybedilir (gölge modda kritik değil) ama servis çalışmaya devam eder."""
    global _metrics_pool  # pylint: disable=global-statement
    loop = asyncio.get_running_loop()
    fn = functools.partial(calculate_metrics, df_prepared, ref_df_prepared, interval=interval)
    try:
        await loop.run_in_executor(_metrics_pool, fn)
    except BrokenProcessPool as e:
        logger.error("Metrics process pool çöktü, yeniden oluşturuluyor: %s", e)
        _metrics_pool.shutdown(wait=False)
        _metrics_pool = ProcessPoolExecutor(max_workers=_METRICS_POOL_WORKERS)


async def _ensure_group(client) -> None:
    try:
        await client.xgroup_create(_STREAM, _GROUP, id="$", mkstream=True)
        logger.info("Consumer group '%s' oluşturuldu", _GROUP)
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group '%s' zaten var", _GROUP)
        else:
            raise


async def _process_event(fields: dict) -> None:
    symbol = fields["symbol"]
    interval = fields["interval"]

    df = await RedisClient.get_mtf_klines(symbol, interval)
    ref_df = await RedisClient.get_mtf_klines(Config.MARKET_REFERENCE_SYMBOL, interval)
    if df is None or ref_df is None or df.empty or ref_df.empty:
        logger.debug("[%s] %s buffer eksik, atlanıyor (gölge)", symbol, interval)
        return

    df_prepared = df.copy()
    df_prepared.index = pd.Index(pd.to_datetime(df_prepared["open_time"], unit="ms"))
    ref_df_prepared = ref_df.copy()
    ref_df_prepared.index = pd.Index(pd.to_datetime(ref_df_prepared["open_time"], unit="ms"))

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await _run_calculate_metrics(df_prepared, ref_df_prepared, interval)
    technical_signals = await signal_engine.calculate_all_signals(df, symbol=symbol, interval=interval)
    elapsed_ms = (loop.time() - t0) * 1000

    signal_count = sum(len(v) for v in technical_signals.values() if isinstance(v, list))
    logger.info(
        "[GÖLGE] [%s] %s işlendi (%.1fms) — %d sinyal üretildi (DB'ye yazılmadı)",
        symbol, interval, elapsed_ms, signal_count,
    )


async def _consume_loop() -> None:
    client = RedisClient.get_client()
    await _ensure_group(client)
    logger.info("Signal service tüketimi başladı (grup=%s, consumer=%s)", _GROUP, _CONSUMER)

    while True:
        try:
            resp = await client.xreadgroup(
                _GROUP, _CONSUMER, {_STREAM: ">"}, count=50, block=2000,
            )
            if not resp:
                await beat("signal_service")
                continue
            for _stream_name, messages in resp:
                for msg_id, fields in messages:
                    try:
                        await _process_event(fields)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error("[%s] event işleme hatası: %s", fields, e, exc_info=True)
                    finally:
                        await client.xack(_STREAM, _GROUP, msg_id)
            await beat("signal_service")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Consume loop hatası: %s", e, exc_info=True)
            await asyncio.sleep(5)


async def run_all() -> None:
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    consume_task = asyncio.create_task(_consume_loop(), name="signal_service_consume")
    watchdog_task = asyncio.create_task(
        watchdog_loop(max_age_seconds={"signal_service": 120}), name="signal_service_watchdog"
    )
    tasks = {consume_task, watchdog_task}

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

    logger.info("Signal service başlatıldı (gölge mod — DB'ye yazmıyor).")
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
