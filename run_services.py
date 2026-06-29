"""Tüm backend servislerini (PgBouncer, LiveDataManager, PerformanceUpdater) başlatır."""

import asyncio
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timedelta
from utils.logger import get_logger

# Çalıştırılacak servislerin ana fonksiyonlarını import et
from live_data_manager import main as live_data_main
from signals.signal_performance_analyzer import SignalPerformanceAnalyzer
from signals.signal_lifecycle_manager import signal_lifecycle_manager

logger = get_logger("ServiceRunner")


def start_pgbouncer():  # pylint: disable=too-many-branches,too-many-statements
    """PgBouncer'ı başlatır ve hazır olmasını bekler."""

    # Önce PgBouncer'ın zaten çalışıp çalışmadığını kontrol et
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 6432))
        sock.close()
        if result == 0:
            logger.info("PgBouncer zaten çalışıyor ve hazır")
            return True
    except OSError:
        pass

    # Eğer mevcut dizinde bir pidfile varsa, kontrol et ve stale (çalışmayan) ise yedekle
    pidfile = os.path.join(os.getcwd(), "pgbouncer.pid")
    if os.path.exists(pidfile):
        try:
            with open(pidfile, "r", encoding="utf-8") as f:
                existing_pid = int(f.read().strip())
        except (OSError, ValueError):
            existing_pid = None

        if existing_pid:
            try:
                # `ps -p <pid> -o comm=` ile süreç adını al
                out = subprocess.run(
                    ["ps", "-p", str(existing_pid), "-o", "comm="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                proc_name = out.stdout.strip()
                if "pgbouncer" not in proc_name.lower():
                    # pidfile stale görünüyorsa yedekle ve devam et
                    bak = pidfile + ".bak"
                    try:
                        os.replace(pidfile, bak)
                        logger.warning(
                            "Stale pidfile bulundu (PID=%s). Yedeklendi: %s",
                            existing_pid,
                            bak,
                        )
                    except OSError:
                        logger.warning(
                            "Stale pidfile bulundu, fakat yedekleme başarısız. El ile kontrol edin."
                        )
                else:
                    logger.info("PgBouncer zaten çalışıyor (pidfile işaret ediyor).")
                    return True
            except OSError:
                logger.warning(
                    "pidfile okundu fakat süreç doğrulanamadı; pidfile taşınacak"
                )
                try:
                    os.replace(pidfile, pidfile + ".bak")
                except OSError:
                    pass

    logger.info("PgBouncer başlatılıyor...")
    try:
        # PgBouncer'ı başlat
        subprocess.run(
            ["pgbouncer", "-d", "pgbouncer.ini"], check=True, cwd=os.getcwd()
        )

        # PgBouncer'ın hazır olmasını bekle (basit port kontrolü)
        for _ in range(10):  # 10 saniye bekle
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", 6432))
                sock.close()
                if result == 0:
                    logger.info("PgBouncer başarıyla başlatıldı ve hazır")
                    return True
            except OSError:
                pass
            time.sleep(1)

        logger.error("PgBouncer başlatılamadı veya hazır değil")
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"PgBouncer başlatma hatası: {e}")
        return False


def _run_performance_update(perf_logger) -> None:
    """Performans güncelleme işini çalıştırır (senkron, executor'da çağrılır)."""
    perf_logger.info("=" * 60)
    perf_logger.info("Performance update başlıyor...")
    perf_logger.info("=" * 60)
    try:
        analyzer = SignalPerformanceAnalyzer()
        result = analyzer.batch_update_performance(hours_back=8760, max_signals=None)
        perf_logger.info("Performance update tamamlandı:")
        perf_logger.info("  Total: %d", result.get("total", 0))
        perf_logger.info("  Success: %d", result.get("success", 0))
        perf_logger.info("  Failed: %d", result.get("failed", 0))
    except Exception as e:  # pylint: disable=broad-exception-caught
        perf_logger.error("Performance update hatası: %s", e, exc_info=True)
    finally:
        perf_logger.info("=" * 60)


async def daily_performance_update_task():
    """Her gün saat 02:00'da signal performance güncelleme görevi."""
    perf_logger = get_logger("PerformanceUpdater")
    target_hour = 2
    target_minute = 0

    perf_logger.info(
        "Performance update task başlatıldı (hedef: %02d:%02d)", target_hour, target_minute
    )

    # Startup: hesaplanmamış kayıt varsa hemen çalıştır
    try:
        import psycopg2
        from config import Config
        _conn = psycopg2.connect(
            host=Config.DB_HOST, port=Config.DB_PORT, database=Config.DB_NAME,
            user=Config.DB_USER, password=Config.DB_PASSWORD,
        )
        _cur = _conn.cursor()
        _cur.execute("SELECT COUNT(*) FROM signal_performance WHERE is_calculated = FALSE")
        pending = _cur.fetchone()[0]
        _cur.close()
        _conn.close()
        if pending > 0:
            perf_logger.info("Startup: %d hesaplanmamış sinyal — hemen başlıyor...", pending)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_performance_update, perf_logger)
        else:
            perf_logger.info("Startup: hesaplanmamış sinyal yok.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        perf_logger.error("Startup performance check hatası: %s", e, exc_info=True)

    while True:
        try:
            now = datetime.now()
            target_time = now.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0
            )
            if now >= target_time:
                target_time += timedelta(days=1)

            wait_seconds = (target_time - now).total_seconds()
            perf_logger.info(
                "Sonraki performance update: %s (%.1f saat)",
                target_time.strftime("%Y-%m-%d %H:%M:%S"),
                wait_seconds / 3600,
            )

            await asyncio.sleep(wait_seconds)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_performance_update, perf_logger)

        except asyncio.CancelledError:
            perf_logger.info("Performance update task iptal edildi")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            perf_logger.error("Performance update task hatası: %s", e, exc_info=True)
            await asyncio.sleep(3600)


async def periodic_gap_scan_task():
    """Her 6 saatte bir tüm MTF intervallar için gap taraması yapar."""
    gap_logger = get_logger("GapScanner")
    _INTERVAL_MS_MAP = {
        "1m": 60_000,
    }
    _SCAN_INTERVAL_HOURS = 1
    _LOOKBACK_DAYS = 7

    gap_logger.info("Periyodik gap scanner başlatıldı (her %d saatte bir)", _SCAN_INTERVAL_HOURS)

    while True:
        try:
            gap_logger.info("Gap taraması başlıyor (son %d gün)...", _LOOKBACK_DAYS)

            from database.engine import get_session
            from sqlalchemy import text
            from binance_client import BinanceClientManager
            from database.crud import bulk_insert_price_data

            async with get_session() as session:
                result = await session.execute(
                    text("SELECT DISTINCT symbol FROM price_data WHERE interval = '1m' "
                         "AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - INTERVAL '1 day'")
                )
                symbols = [r[0] for r in result.fetchall()]

            if not symbols:
                gap_logger.warning("Aktif sembol bulunamadı, tarama atlandı.")
                continue

            gap_logger.info("%d aktif sembol taranacak", len(symbols))
            total_filled = 0
            import time as _t

            for interval, interval_ms in _INTERVAL_MS_MAP.items():
                for symbol in symbols:
                    try:
                        async with get_session() as session:
                            r = await session.execute(
                                text("""
                                    SELECT symbol, prev_ts, curr_ts
                                    FROM (
                                        SELECT symbol, timestamp AS curr_ts,
                                               LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                                        FROM price_data
                                        WHERE symbol = :sym AND interval = :iv
                                          AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - (:days * INTERVAL '1 day')
                                    ) t
                                    WHERE prev_ts IS NOT NULL
                                      AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > :thresh
                                    ORDER BY prev_ts
                                """),
                                {"sym": symbol, "iv": interval, "days": _LOOKBACK_DAYS, "thresh": interval_ms * 2},
                            )
                            gaps = [
                                (int(row[1].timestamp() * 1000), int(row[2].timestamp() * 1000))
                                for row in r.fetchall()
                            ]

                            r2 = await session.execute(
                                text("SELECT MAX(timestamp) FROM price_data WHERE symbol = :sym AND interval = :iv"),
                                {"sym": symbol, "iv": interval},
                            )
                            last_row = r2.fetchone()
                            last_dt = last_row[0] if last_row and last_row[0] else None
                            last_ms = int(last_dt.timestamp() * 1000) if last_dt else None

                        now_ms = int(_t.time() * 1000)
                        lookback_start_ms = now_ms - int(_LOOKBACK_DAYS * 86_400_000)
                        if last_ms and (now_ms - last_ms) > interval_ms * 2:
                            tail_start = max(last_ms, lookback_start_ms)
                            if (now_ms - tail_start) > interval_ms * 2:
                                if not any(g[0] >= tail_start for g in gaps):
                                    gaps.append((tail_start, now_ms))

                    except Exception as exc:
                        gap_logger.debug("[GapScan] %s %s sorgu hatası: %s", symbol, interval, exc)
                        continue

                    for gap_start_ms, gap_end_ms in gaps:
                        fetch_start = gap_start_ms + interval_ms
                        while fetch_start < gap_end_ms:
                            await asyncio.sleep(0.5)
                            try:
                                df = await BinanceClientManager.fetch_klines(
                                    symbol=symbol, interval=interval, limit=1000, startTime=fetch_start,
                                )
                            except Exception:
                                break
                            if df is None or df.empty:
                                break
                            df = df[df["open_time"] < gap_end_ms]
                            if df.empty:
                                break
                            await bulk_insert_price_data(symbol, df, interval=interval)
                            total_filled += len(df)
                            last_ts = int(df["open_time"].iloc[-1])
                            if last_ts <= fetch_start or len(df) < 1000:
                                break
                            fetch_start = last_ts + interval_ms

            if total_filled:
                gap_logger.info("[GapScan] %d bar eklendi", total_filled)

            gap_logger.info("Gap taraması tamamlandı: toplam %d bar eklendi", total_filled)

        except asyncio.CancelledError:
            gap_logger.info("Gap scanner iptal edildi.")
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            gap_logger.error("Gap scanner hatası: %s", exc, exc_info=True)

        await asyncio.sleep(_SCAN_INTERVAL_HOURS * 3600)


async def run_all_services():
    """Tüm servisleri doğru sırada başlatır ve yönetir."""
    logger.info("Tüm servisler başlatılıyor...")

    # 1. PgBouncer'ı başlat ve hazır olmasını bekle
    if not start_pgbouncer():
        logger.error("PgBouncer başlatılamadı, çıkılıyor...")
        return

    # 3. LiveDataManager'ı başlat
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    live_task = asyncio.create_task(live_data_main(), name="live_data_manager")

    # 4. Daily Performance Update Task
    perf_task = asyncio.create_task(
        daily_performance_update_task(), name="performance_updater"
    )

    # 5. Periyodik Gap Scanner (her 6 saatte bir)
    gap_task = asyncio.create_task(
        periodic_gap_scan_task(), name="gap_scanner"
    )

    # 6. Sinyal timeout sweeper (her 5 dakikada bir)
    async def _sweep_loop():
        sweep_logger = get_logger("SignalSweeper")
        while True:
            await asyncio.sleep(300)
            try:
                closed = await signal_lifecycle_manager.sweep_timeouts()
                if closed:
                    sweep_logger.info(f"Sweep tamamlandı: {closed} sinyal kapatıldı")
            except Exception as exc:
                sweep_logger.error(f"Sweep hatası: {exc}", exc_info=True)

    sweep_task = asyncio.create_task(_sweep_loop(), name="signal_sweeper")

    tasks = {live_task, perf_task, gap_task, sweep_task}

    # Sinyal yakalayıcı: görevleri iptal et ve shutdown akışını başlat
    def _signal_handler(sig_name: str):
        logger.info(
            f"Sinyal alındı: {sig_name}. Servisler düzgün şekilde kapatılıyor..."
        )
        for t in list(tasks):
            if not t.done():
                t.cancel()
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, _signal_handler, sig.name  # pylint: disable=no-member
                )
            except NotImplementedError:
                # Windows vb. ortamlarda add_signal_handler desteklenmeyebilir.
                pass

        logger.info("Tüm servisler başlatıldı. Çalışıyor...")

        # Ana bekleme: herhangi bir görev biterse ya da shutdown tetiklenirse çık
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    except asyncio.CancelledError:
        logger.info("Servisler iptal edildi (CancelledError).")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            f"Servis yöneticisinde beklenmedik bir hata oluştu: {e}", exc_info=True
        )
    finally:
        for t in list(tasks):
            if not t.done():
                t.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Tüm servisler kapatıldı.")


_PID_FILE = "run_services.pid"


def _acquire_pid_lock() -> bool:
    """PID dosyası ile tek instance garantisi. Çakışma varsa False döner."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as f:
                old_pid = int(f.read().strip())
            # Eski process hâlâ çalışıyor mu?
            os.kill(old_pid, 0)
            logger.error(
                "run_services zaten çalışıyor (PID %d). Yeni instance başlatılmıyor.", old_pid
            )
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale PID — üzerine yaz

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
        raise SystemExit(0)  # temiz çıkış → launchctl yeniden başlatmaz
    try:
        asyncio.run(run_all_services())
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından servisler durduruluyor...")
    finally:
        _release_pid_lock()
