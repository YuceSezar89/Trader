import asyncio
import signal
import subprocess
import time
import os
from datetime import datetime, time as dt_time
from utils.logger import get_logger
from database.crud import initialize_database

# Çalıştırılacak servislerin ana fonksiyonlarını import et
from live_data_manager import main as live_data_main
from signals.signal_performance_analyzer import SignalPerformanceAnalyzer

logger = get_logger("ServiceRunner")


def start_pgbouncer():
    """PgBouncer'ı başlatır ve hazır olmasını bekler."""
    import socket
    import subprocess
    import json
    
    # Önce PgBouncer'ın zaten çalışıp çalışmadığını kontrol et
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6432))
        sock.close()
        if result == 0:
            logger.info("PgBouncer zaten çalışıyor ve hazır")
            return True
    except:
        pass
    
    # Eğer mevcut dizinde bir pidfile varsa, kontrol et ve stale (çalışmayan) ise yedekle
    pidfile = os.path.join(os.getcwd(), "pgbouncer.pid")
    if os.path.exists(pidfile):
        try:
            with open(pidfile, "r") as f:
                existing_pid = int(f.read().strip())
        except Exception:
            existing_pid = None

        if existing_pid:
            try:
                # `ps -p <pid> -o comm=` ile süreç adını al
                out = subprocess.run(["ps", "-p", str(existing_pid), "-o", "comm="], capture_output=True, text=True)
                proc_name = out.stdout.strip()
                if "pgbouncer" not in proc_name.lower():
                    # pidfile stale görünüyorsa yedekle ve devam et
                    bak = pidfile + ".bak"
                    try:
                        os.replace(pidfile, bak)
                        logger.warning(f"Stale pidfile bulundu (PID={existing_pid}). Yedeklendi: {bak}")
                    except Exception:
                        logger.warning("Stale pidfile bulundu, fakat yedekleme başarısız oldu. El ile kontrol edin.")
                else:
                    # PgBouncer gerçekten çalışıyor
                    logger.info("PgBouncer zaten çalışıyor (pidfile işaret ediyor).")
                    return True
            except Exception:
                # Güvenli taraf için devam et
                logger.warning("pidfile okundu fakat süreç doğrulanamadı; pidfile taşınacak")
                try:
                    os.replace(pidfile, pidfile + ".bak")
                except Exception:
                    pass

    logger.info("PgBouncer başlatılıyor...")
    try:
        # PgBouncer'ı başlat
        subprocess.run(["pgbouncer", "-d", "pgbouncer.ini"], check=True, cwd=os.getcwd())
        
        # PgBouncer'ın hazır olmasını bekle (basit port kontrolü)
        for i in range(10):  # 10 saniye bekle
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 6432))
                sock.close()
                if result == 0:
                    logger.info("PgBouncer başarıyla başlatıldı ve hazır")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("PgBouncer başlatılamadı veya hazır değil")
        return False
    except Exception as e:
        logger.error(f"PgBouncer başlatma hatası: {e}")
        return False

def start_streamlit():
    """Streamlit uygulamasını arka planda başlatır."""
    logger.info("Streamlit uygulaması başlatılıyor...")
    try:
        # Streamlit'i arka planda başlat
        process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port=8502"],
            cwd=os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Streamlit başlatıldı (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Streamlit başlatma hatası: {e}")
        return None


async def daily_performance_update_task():
    """Her gün saat 02:00'da signal performance güncelleme görevi."""
    # Ayrı logger oluştur
    perf_logger = get_logger("PerformanceUpdater")
    
    target_hour = 2  # Saat 02:00
    target_minute = 0
    
    perf_logger.info(f"Performance update task başlatıldı (hedef: {target_hour:02d}:{target_minute:02d})")
    logger.info(f"Performance update task başlatıldı (hedef: {target_hour:02d}:{target_minute:02d})")
    
    while True:
        try:
            # Şu anki zaman
            now = datetime.now()
            
            # Hedef zaman (bugün veya yarın saat 02:00)
            target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            
            # Eğer hedef zaman geçmişse, yarına ayarla
            if now >= target_time:
                target_time = target_time.replace(day=target_time.day + 1)
            
            # Bekleme süresi
            wait_seconds = (target_time - now).total_seconds()
            perf_logger.info(f"Sonraki performance update: {target_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds/3600:.1f} saat sonra)")
            logger.info(f"Sonraki performance update: {target_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds/3600:.1f} saat sonra)")
            
            # Hedef zamana kadar bekle
            await asyncio.sleep(wait_seconds)
            
            # Performance update çalıştır
            perf_logger.info("=" * 60)
            perf_logger.info("Günlük performance update başlıyor...")
            perf_logger.info("=" * 60)
            
            try:
                analyzer = SignalPerformanceAnalyzer()
                result = analyzer.batch_update_performance(
                    hours_back=720,  # 30 gün
                    max_signals=1000
                )
                
                perf_logger.info("Performance update tamamlandı:")
                perf_logger.info(f"  Total: {result.get('total', 0)}")
                perf_logger.info(f"  Success: {result.get('success', 0)}")
                perf_logger.info(f"  Failed: {result.get('failed', 0)}")
                perf_logger.info(f"  Skipped: {result.get('skipped', 0)}")
                
            except Exception as e:
                perf_logger.error(f"Performance update hatası: {e}", exc_info=True)
            
            perf_logger.info("=" * 60)
            
        except asyncio.CancelledError:
            perf_logger.info("Performance update task iptal edildi")
            logger.info("Performance update task iptal edildi")
            break
        except Exception as e:
            perf_logger.error(f"Performance update task hatası: {e}", exc_info=True)
            logger.error(f"Performance update task hatası: {e}", exc_info=True)
            # Hata olsa bile devam et, 1 saat sonra tekrar dene
            await asyncio.sleep(3600)

async def run_all_services():
    """Tüm servisleri doğru sırada başlatır ve yönetir."""
    logger.info("Tüm servisler başlatılıyor...")
    
    # 1. Veritabanını başlat
    await initialize_database()
    
    # 2. PgBouncer'ı başlat ve hazır olmasını bekle
    if not start_pgbouncer():
        logger.error("PgBouncer başlatılamadı, çıkılıyor...")
        return
    
    # 3. Streamlit'i başlat
    streamlit_process = start_streamlit()
    
    # 4. LiveDataManager'ı başlat
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    
    live_task = asyncio.create_task(live_data_main(), name="live_data_manager")
    
    # 5. Daily Performance Update Task'ı başlat
    perf_task = asyncio.create_task(daily_performance_update_task(), name="performance_updater")
    
    tasks = {live_task, perf_task}

    # Sinyal yakalayıcı: görevleri iptal et ve shutdown akışını başlat
    def _signal_handler(sig_name: str):
        logger.info(f"Sinyal alındı: {sig_name}. Servisler düzgün şekilde kapatılıyor...")
        # Streamlit'i kapat
        if streamlit_process:
            streamlit_process.terminate()
        # LiveDataManager'ı kapat
        for t in list(tasks):
            if not t.done():
                t.cancel()
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler, sig.name)
            except NotImplementedError:
                # Windows vb. ortamlarda add_signal_handler desteklenmeyebilir.
                pass

        logger.info("Tüm servisler başlatıldı. Çalışıyor...")
        logger.info("Streamlit arayüzü: http://localhost:8502")
        
        # Ana bekleme: herhangi bir görev biterse ya da shutdown tetiklenirse çık
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    except asyncio.CancelledError:
        logger.info("Servisler iptal edildi (CancelledError).")
    except Exception as e:
        logger.error(f"Servis yöneticisinde beklenmedik bir hata oluştu: {e}", exc_info=True)
    finally:
        # Streamlit'i kapat
        if streamlit_process:
            logger.info("Streamlit kapatılıyor...")
            streamlit_process.terminate()
        
        # Tüm görevlerin kapanmasını bekle
        for t in list(tasks):
            if not t.done():
                t.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Tüm servisler kapatıldı.")


if __name__ == "__main__":
    try:
        asyncio.run(run_all_services())
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından servisler durduruluyor...")
