import asyncio
import sys
import os
import pandas as pd

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance_client import BinanceClientManager
from signals.signal_processor import process_and_enrich_signals
from indicators.core import add_all_indicators
from database.crud import save_price_data_batch
from config import Config
from utils.logger import get_logger

logger = get_logger("manual_scan")

async def main():
    """Manuel olarak sinyal taraması yapar ve veritabanını doldurur."""
    INTERVAL = "15m"
    LIMIT = 300 # İndikatörler için yeterli veri
    COIN_LIMIT = 50 # Taranacak coin sayısı
    REFERENCE_SYMBOL = "BTCUSDT"

    logger.info(f"Manuel tarama başlatıldı. Zaman aralığı: {INTERVAL}, Coin limiti: {COIN_LIMIT}")

    # 1. Taranacak sembol listesini al
    try:
        symbols = await BinanceClientManager.get_top_volume_symbols_async(limit=COIN_LIMIT)
        if REFERENCE_SYMBOL not in symbols:
            symbols.append(REFERENCE_SYMBOL)
        logger.info(f"{len(symbols)} adet sembol tarama için belirlendi.")
    except Exception as e:
        logger.error(f"Sembol listesi alınamadı: {e}")
        return

    # 2. Tüm sembollerin verilerini toplu olarak çek
    logger.info("Tüm semboller için mum verileri çekiliyor...")
    results = await BinanceClientManager.fetch_all_klines(symbols, INTERVAL, LIMIT)
    
    data_map = {res[0]: res[1] for res in results if res[1] is not None and not res[1].empty}

    # 4. Ham verileri veritabanına kaydet
    if data_map:
        logger.info(f"{len(data_map)} sembol için ham fiyat verileri veritabanına kaydediliyor...")
        await save_price_data_batch(data_map, interval=INTERVAL)

    if REFERENCE_SYMBOL not in data_map:
        logger.error(f"Referans sembol ({REFERENCE_SYMBOL}) için veri alınamadı. Tarama durduruldu.")
        return
    
    ref_df = data_map[REFERENCE_SYMBOL]

    # 3. Her sembol için sinyalleri işle
    tasks = []
    for symbol, df in data_map.items():
        if symbol == REFERENCE_SYMBOL:
            continue
        
        logger.debug(f"[{symbol}] için sinyal işleme görevi oluşturuluyor...")
        # Önce tüm indikatörleri hesapla
        df_with_indicators = add_all_indicators(df)
        tasks.append(process_and_enrich_signals(symbol, df_with_indicators, ref_df, INTERVAL))

    logger.info(f"{len(tasks)} adet sembol için sinyal işleme görevleri başlatılıyor.")
    await asyncio.gather(*tasks)

    logger.info("Manuel tarama tamamlandı.")

if __name__ == "__main__":
    asyncio.run(main())
