import asyncio
import signal
from utils.logger import get_logger
from database.crud import initialize_database

# Çalıştırılacak servislerin ana fonksiyonlarını import et
from live_data_manager import main as live_data_main
from signal_performance_tracker import main as performance_tracker_main

logger = get_logger("ServiceRunner")


async def run_all_services():
    """Tüm arka plan servislerini eşzamanlı olarak başlatır ve yönetir.
    SIGINT/SIGTERM sinyallerinde görevleri iptal ederek düzgün kapanış yapar.
    """
    logger.info("Tüm arka plan servisleri başlatılıyor...")
    # Veritabanını ve olası migration'ları başlat
    await initialize_database()

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Çalışacak görevleri oluştur
    live_task = asyncio.create_task(live_data_main(), name="live_data_manager")
    perf_task = asyncio.create_task(performance_tracker_main(), name="performance_tracker")
    tasks = {live_task, perf_task}

    # Sinyal yakalayıcı: görevleri iptal et ve shutdown akışını başlat
    def _signal_handler(sig_name: str):
        logger.info(f"Sinyal alındı: {sig_name}. Servisler düzgün şekilde kapatılıyor...")
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

        # Ana bekleme: herhangi bir görev biterse ya da shutdown tetiklenirse çık
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    except asyncio.CancelledError:
        logger.info("Servisler iptal edildi (CancelledError).")
    except Exception as e:
        logger.error(f"Servis yöneticisinde beklenmedik bir hata oluştu: {e}", exc_info=True)
    finally:
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
