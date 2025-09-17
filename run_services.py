import asyncio
import signal
import subprocess
import time
import os
from utils.logger import get_logger
from database.crud import initialize_database

# Çalıştırılacak servislerin ana fonksiyonlarını import et
from live_data_manager import main as live_data_main

logger = get_logger("ServiceRunner")


def start_pgbouncer():
    """PgBouncer'ı başlatır ve hazır olmasını bekler."""
    import socket
    
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
    tasks = {live_task}

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
