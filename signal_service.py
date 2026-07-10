"""Signal/Feature servisi — Faz 4: sinyal işleme yolunu (process_and_enrich_signals)
Config.SIGNAL_SOURCE="yeni" iken gerçek yazar (aktif). Paper trading'in son iki
parçası (10 Tem 2026, ingestion/paper-trading ayrıştırması): risk kontrolü
(_risk_check_loop) ve detector-tabanlı stratejiler (do_kirilimi/do_open_streak)
Config.PAPER_TRADING_SOURCE="yeni" iken burada gerçek yazar; "eski" (varsayılan)
iken live_data_manager.py'deki eski yol aktif kalır — aynı feature-flag deseni,
geri dönüş tek satır + iki servis restart.
"""

import asyncio
import functools
import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from utils.logger import get_logger
from utils.redis_client import RedisClient, SAFE_EXTERNAL_TIMEOUT
from utils.heartbeat import beat, watchdog_loop, record_activity, throughput_watchdog_loop
from utils.telegram_notify import send_telegram_message
from indicators.financial_metrics import calculate_metrics
from signals.signal_processor import process_and_enrich_signals
from signals.risk_manager import risk_manager
from signals.paper_trade_manager import (
    paper_trade_manager, ha_cross_manager, rsi_15m_manager, manual_manager,
    do_kirilimi_manager, do_open_streak_manager,
)
from signals.do_kirilimi import do_kirilimi_detector, btc_day_context
from signals.do_open_streak import do_open_streak_detector
from config import Config

_RISK_CHECK_INTERVAL = 5  # saniye — live_data_manager.py::_risk_check_loop ile aynı

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

    if Config.PAPER_TRADING_SOURCE == "yeni":
        if interval == "5m":
            await _check_do_kirilimi(symbol, df, ref_df)
        elif interval == "15m":
            await _check_do_open_streak(symbol, df)


async def _check_do_kirilimi(symbol: str, df_5m: pd.DataFrame, btc_df_5m: pd.DataFrame) -> None:
    """DO Kırılımı dedektörü — live_data_manager.py::_check_do_kirilimi'nin
    signal_service.py eşdeğeri (10 Tem 2026). BTC context artık in-memory
    self.mtf_buffers'tan değil, zaten elde olan ref_df'ten (MARKET_REFERENCE_SYMBOL
    = BTCUSDT) hesaplanıyor — process ayrımı sonrası in-process state erişimi yok."""
    try:
        btc_ctx = btc_day_context(btc_df_5m) if btc_df_5m is not None and not btc_df_5m.empty else None
        loop = asyncio.get_running_loop()
        entry = await loop.run_in_executor(None, do_kirilimi_detector.check, symbol, df_5m, btc_ctx)
        if not entry:
            return
        opened = await do_kirilimi_manager.open_direct(
            symbol=symbol,
            signal_type="Long",
            interval="5m",
            price=entry["price"],
            atr=entry["atr"],
            sl_price=entry["sl_price"],
            tp_price=entry["tp_price"],
            note=f"{entry['pattern']} {entry['ayrisma']:+.1f}%",
        )
        if opened:
            await send_telegram_message(
                f"🎯 DO Kırılımı — {symbol}\n"
                f"Giriş: {entry['price']:.6g}\n"
                f"SL: {entry['sl_price']:.6g} · TP: {entry['tp_price']:.6g}\n"
                f"Pattern: {entry['pattern']} · Ayrışma: {entry['ayrisma']:+.1f}%"
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("[DOKirilimi] %s kanca hatası: %s", symbol, exc, exc_info=True)


async def _check_do_open_streak(symbol: str, df_15m: pd.DataFrame) -> None:
    """DO Kırılımı + Ardışık Yeşil Mum + Gauss dedektörü — live_data_manager.py::
    _check_do_open_streak'in signal_service.py eşdeğeri (10 Tem 2026)."""
    try:
        loop = asyncio.get_running_loop()
        entry = await loop.run_in_executor(None, do_open_streak_detector.check, symbol, df_15m)
        if not entry:
            return

        cfg = Config.PAPER["DO_OPEN_STREAK"]
        sl_dist = entry["price"] - entry["sl_price"]
        if sl_dist <= 0:
            return
        position_usd = cfg["TARGET_RISK_USD"] * entry["price"] / sl_dist

        opened = await do_open_streak_manager.open_direct(
            symbol=symbol,
            signal_type="Long",
            interval="15m",
            price=entry["price"],
            atr=entry["atr"],
            sl_price=entry["sl_price"],
            tp_price=entry["tp_price"],
            note=f"gauss={entry['gauss_val']:.1f} hareket={entry['long_perc']:+.1f}%",
            position_usd=position_usd,
        )
        if opened:
            await send_telegram_message(
                f"📈 DO Streak — {symbol}\n"
                f"Giriş: {entry['price']:.6g}\n"
                f"SL: {entry['sl_price']:.6g} (TP yok — 24h timeout)\n"
                f"Pozisyon: ${position_usd:.0f} · Gauss: {entry['gauss_val']:.1f} · "
                f"3-mum hareket: {entry['long_perc']:+.1f}%"
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("[DoOpenStreak] %s kanca hatası: %s", symbol, exc, exc_info=True)


async def _risk_check_loop() -> None:
    """live_data_manager.py::_risk_check_loop'un signal_service.py eşdeğeri (10 Tem
    2026) — her 5 saniyede paper trade pozisyonlarını kontrol eder. Fiyatlar artık
    in-memory değil, live_data_manager.py'nin zaten her saniye yazdığı Redis
    'prices:live' key'inden okunuyor. Sadece PAPER_TRADING_SOURCE="yeni" iken aktif —
    "eski" iken live_data_manager.py'nin kendi döngüsü çalışmaya devam eder."""
    while True:
        await asyncio.sleep(_RISK_CHECK_INTERVAL)
        if Config.PAPER_TRADING_SOURCE != "yeni":
            continue
        try:
            raw = await asyncio.wait_for(RedisClient.get_client().get("prices:live"), timeout=SAFE_EXTERNAL_TIMEOUT)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("[RiskCheck] prices:live okunamadı: %s", e)
            continue
        if not raw:
            continue
        try:
            prices = json.loads(raw)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("[RiskCheck] prices:live parse hatası: %s", e)
            continue
        if not prices:
            continue
        await paper_trade_manager.check_all_prices(prices)
        await ha_cross_manager.check_all_prices(prices)
        await rsi_15m_manager.check_all_prices(prices)
        await manual_manager.check_all_prices(prices)
        await do_kirilimi_manager.check_all_prices(prices)
        await do_open_streak_manager.check_all_prices(prices)
        await beat("paper_trading_risk_check")


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
    await do_open_streak_manager.load_open_symbols()

    consume_task = asyncio.create_task(
        _supervised(_consume_loop(), "signal_service_consume"), name="signal_service_consume"
    )
    risk_check_task = asyncio.create_task(
        _supervised(_risk_check_loop(), "signal_service_risk_check"), name="signal_service_risk_check"
    )
    watchdog_task = asyncio.create_task(
        _supervised(
            watchdog_loop(max_age_seconds={"signal_service": 120, "paper_trading_risk_check": 60}),
            "signal_service_watchdog",
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
            throughput_watchdog_loop(min_expected={"signal_service": 1}, self_heal_after=3),
            "signal_service_throughput",
        ),
        name="signal_service_throughput",
    )
    tasks = {consume_task, risk_check_task, watchdog_task, queue_lag_task, reclaim_task, throughput_task}

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
