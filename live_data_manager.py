import asyncio
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import pandas as pd

from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from binance_client import BinanceClientManager
from utils.exceptions import BinanceAPIError
from indicators.core import add_all_indicators
from database.crud import (bulk_insert_price_data,
                           get_last_timestamp, initialize_database, delete_symbol_data)
from signals.signal_processor import process_and_enrich_signals
from utils.exceptions import BinanceAPIError, DatabaseError
from config import Config
from utils.redis_client import RedisClient

def setup_logging():
    """live_data_manager için özel loglama ayarlarını yapar."""
    log_dir = Config.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(Config.LOG_LEVEL)
    logger.propagate = False  # Root logger'a logların gitmesini engelle

    # Handler'ların tekrar tekrar eklenmesini önle
    if logger.hasHandlers():
        logger.handlers.clear()

    # Dosya Handler'ı (Rotating)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'live_data_manager.log'),
        maxBytes=Config.LOG_FILE_MAX_SIZE,
        backupCount=Config.LOG_FILE_BACKUP_COUNT
    )
    # Konsol Handler'ı
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# --- Logging Kurulumu ---
logger = setup_logging()


class LiveDataManager:
    """
    Tarihsel verileri senkronize eden ve ardından WebSocket üzerinden canlı veri alarak
    sinyal üreten yönetici sınıfı.
    """

    def __init__(self, symbols: List[str], interval: str = Config.KLINE_INTERVAL):
        self.ref_symbol = Config.MARKET_REFERENCE_SYMBOL
        # Referans sembolün listede olduğundan emin ol
        if self.ref_symbol not in symbols:
            symbols.insert(0, self.ref_symbol) # Başa ekle
        self.symbols = list(dict.fromkeys(symbols)) # Duplike varsa kaldır

        self.interval = interval
        # WebSocket istemcisi: tek bağlantı üzerinde combined-stream kullanacağız
        self.ws_client = UMFuturesWebsocketClient(
            on_message=self._handle_websocket_message,
            on_close=self._handle_ws_close,
            on_error=self._handle_ws_error
        )
        self.is_ws_connected = False
        self.last_message_time: Optional[float] = None  # Son WebSocket mesajının zamanını takip et
        self.reconnect_attempt = 0  # Üstel backoff için sayaç
        self.db_lock = asyncio.Lock()  # Veritabanı yazma işlemleri için kilit
        self.kline_data: Dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}
        self.processing_tasks: set[asyncio.Task] = set()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    async def sync_historical_data(self):
        """
        Tüm semboller için geçmiş verileri senkronize eder.
        - Her sembol için veritabanından son zaman damgasını alır.
        - Binance'ten son zaman damgasından bu yana eksik olan mumları çeker.
        - Çekilen verileri veritabanına kaydeder.
        """
        logger.info("Tarihsel veri senkronizasyonu başlatılıyor...")
        tasks = [self._sync_symbol_data(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                logger.error(f"[{symbol}] Tarihsel veri senkronizasyonu sırasında kritik hata: {result}")
            else:
                logger.info(f"[{symbol}] Tarihsel veri senkronizasyonu tamamlandı.")

    async def _sync_symbol_data(self, symbol: str):
        """Helper method to sync historical data for a single symbol."""
        try:
            last_timestamp = await get_last_timestamp(symbol, interval=self.interval)
            start_time = last_timestamp + 1 if last_timestamp else None

            if start_time:
                # start_time bir sonraki ms olduğundan, görüntüleme için 1 ms geri alıyoruz
                logger.info(f"[{symbol}] Son kayıt: {pd.to_datetime(start_time - 1, unit='ms')}. Eksik veriler çekiliyor...")
            else:
                logger.info(f"[{symbol}] Veritabanında kayıt bulunamadı. Son 1500 mum çekiliyor...")

            df_missing = await BinanceClientManager.fetch_klines(
                symbol=symbol,
                interval=self.interval,
                limit=1500,
                startTime=start_time
            )

            if not df_missing.empty:
                logger.info(f"[{symbol}] {len(df_missing)} adet yeni mum verisi bulundu. Veritabanına kaydediliyor...")
                # İndikatör hesaplamadan doğrudan ham veriyi kaydet
                async with self.db_lock:
                    await bulk_insert_price_data(symbol, df_missing, interval=self.interval)
            else:
                logger.info(f"[{symbol}] Yeni veri bulunamadı, sistem güncel.")

        except DatabaseError as e:
            logger.error(f"[{symbol}] Veritabanı hatası oluştu: {e}")
            raise
        except BinanceAPIError as e:
            logger.error(f"[{symbol}] Veri senkronizasyonu sırasında Binance API hatası: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"[{symbol}] Veri senkronizasyonunda beklenmedik hata: {e}", exc_info=True)
            raise

    async def _initialize_dataframes(self):
        """Initializes in-memory DataFrames with the last 500 klines for signal calculation."""
        logger.info("Sinyal hesaplaması için başlangıç verileri yükleniyor...")
        tasks = [self._load_initial_data(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        symbols_to_remove = []
        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                logger.error(f"[{symbol}] Başlangıç verisi yüklenirken hata: {result}")
                symbols_to_remove.append(symbol)
            elif isinstance(result, pd.DataFrame) and not result.empty:
                # Son 24 saatlik (96 * 15dk) veride hacim kontrolü
                recent_data = result.tail(96)
                if recent_data['volume'].sum() < Config.MIN_VOLUME_THRESHOLD:
                    logger.warning(f"[{symbol}] Düşük hacimli (son 24s hacim < {Config.MIN_VOLUME_THRESHOLD}), izlemeden çıkarılıyor.")
                    symbols_to_remove.append(symbol)
                    # Bu sembol için veritabanından da temizlik yapalım
                    task = asyncio.create_task(self._purge_symbol_data(symbol))
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
                else:
                    df = add_all_indicators(result)
                    self.kline_data[symbol] = df
                    logger.info(f"[{symbol}] {len(df)} adet mum başlangıç verisi olarak yüklendi ve göstergeler hesaplandı.")
            else:
                logger.warning(f"[{symbol}] için başlangıç verisi yüklenemedi veya veri boş, izlemeden çıkarılıyor.")
                symbols_to_remove.append(symbol)

        if symbols_to_remove:
            self.symbols = [s for s in self.symbols if s not in symbols_to_remove]
            for s in symbols_to_remove:
                del self.kline_data[s]
            logger.info(f"Düşük hacimli/hatalı semboller temizlendi. Güncel izleme listesi: {self.symbols}")

    async def _load_initial_data(self, symbol: str) -> pd.DataFrame:
        """Helper to fetch initial kline data for one symbol."""
        try:
            # We fetch 500 to have enough data for indicators like MA200
            return await BinanceClientManager.fetch_klines(symbol, self.interval, limit=500)
        except BinanceAPIError as e:
            logger.error(f"[{symbol}] Başlangıç verisi çekilirken Binance API hatası: {e}", exc_info=True)
            raise # Hatayı yukarıya ilet
        except Exception as e:
            logger.error(f"[{symbol}] Başlangıç verisi çekilemedi: {e}", exc_info=True)
            raise # Hatayı yukarıya ilet

    def _handle_websocket_message(self, _, msg: str):
        """WebSocket'ten gelen her mesajı işler."""
        self.last_message_time = self.loop.time() # Her mesajda zamanı güncelle
        try:
            data = json.loads(msg)
            if data.get('e') == 'kline':
                kline = data['k']
                symbol = kline['s']
                is_closed = kline['x']

                if is_closed:
                    logger.info(f"🕯️ [{symbol}] Mum kapandı. Fiyat: {kline['c']}")
                    # WebSocket thread'inden ana event loop'a güvenli coroutine çağrısı
                    asyncio.run_coroutine_threadsafe(self._update_and_process_symbol(symbol, kline), self.loop)

        except json.JSONDecodeError:
            logger.error(f"WebSocket'ten bozuk JSON verisi alındı: {msg}")
        except Exception as e:
            logger.error(f"WebSocket mesaj işleme hatası: {e} | Mesaj: {msg}", exc_info=True)

    async def _update_and_process_symbol(self, symbol: str, kline_data: Dict):
        """Updates the DataFrame with the new kline and triggers signal processing."""
        try:
            new_row = {
                'open_time': int(kline_data['t']),
                'open': float(kline_data['o']),
                'high': float(kline_data['h']),
                'low': float(kline_data['l']),
                'close': float(kline_data['c']),
                'volume': float(kline_data['v']),
                'close_time': int(kline_data['T']),
                'quote_asset_volume': float(kline_data['q']),
                'number_of_trades': int(kline_data['n']),
                'taker_buy_base_asset_volume': float(kline_data['V']),
                'taker_buy_quote_asset_volume': float(kline_data['Q']),
            }
            
            # Using pd.concat instead of append
            new_df = pd.DataFrame([new_row])
            self.kline_data[symbol] = pd.concat([self.kline_data[symbol], new_df], ignore_index=True)

            # Bellekteki veri setini belirli bir boyutta tut (örneğin son 1000 mum)
            self.kline_data[symbol] = self.kline_data[symbol].tail(1000)

            # Önce yeni kapanan mumu (ham veri) veritabanına kaydet
            try:
                async with self.db_lock:
                    await bulk_insert_price_data(symbol, new_df, interval=self.interval)
                logger.info(f"[{symbol}] Yeni kapanan mum (ham veri) veritabanına kaydedildi.")
            except Exception as db_err:
                logger.error(f"[{symbol}] Canlı veri veritabanına kaydedilirken hata: {db_err}", exc_info=True)

            # Hafızadaki veri üzerinde sinyal üretimi için göstergeleri hesapla
            self.kline_data[symbol] = add_all_indicators(self.kline_data[symbol])

            # Göstergelerle zenginleştirilmiş DataFrame'i Redis'e yaz
            # Yeni standart anahtar: <prefix>:<symbol>:<interval>
            new_redis_key = f"{Config.REDIS_LIVE_DATA_KEY_PREFIX}:{symbol}:{self.interval}"
            await RedisClient.set_df(new_redis_key, self.kline_data[symbol])
            # Geriye uyum: eski anahtara da yaz (yakında kaldırılabilir)
            legacy_redis_key = f"{Config.REDIS_LIVE_DATA_KEY_PREFIX}:{symbol}"
            await RedisClient.set_df(legacy_redis_key, self.kline_data[symbol])
            logger.info(f"[{symbol}] Canlı veri Redis'e yazıldı. Yeni: {new_redis_key} | Legacy: {legacy_redis_key}")

            task = asyncio.create_task(self._process_signal_for_symbol(symbol))
            self.processing_tasks.add(task)
            task.add_done_callback(self.processing_tasks.discard)

        except Exception as e:
            logger.error(f"Failed to update and process symbol {symbol}: {e}", exc_info=True)

    async def _purge_symbol_data(self, symbol: str):
        """Deletes all data for a given symbol from the database."""
        try:
            # from database.crud import delete_symbol_data # Artık gerekli değil, global scope'a taşınacak.
            logger.info(f"[{symbol}] Veritabanından temizleniyor...")
            async with self.db_lock:
                await delete_symbol_data(symbol)
            logger.info(f"[{symbol}] Veritabanından başarıyla temizlendi.")
        except Exception as e:
            logger.error(f"[{symbol}] Veritabanı temizliği sırasında hata: {e}")

    def _handle_ws_close(self, _):
        """Callback function for when the websocket connection is closed."""
        logger.warning("WebSocket bağlantısı kapandı.")
        self.is_ws_connected = False

    def _handle_ws_error(self, _, error):
        """Callback function for websocket errors."""
        logger.error(f"WebSocket hatası: {error}", exc_info=True)
        self.is_ws_connected = False

    async def _process_signal_for_symbol(self, symbol: str):
        """Belirli bir sembol için sinyal hesaplamasını ve zenginleştirmesini tetikler."""
        try:
            if symbol == self.ref_symbol:
                return  # Referans sembol için sinyal üretme

            df = self.kline_data.get(symbol)
            ref_df = self.kline_data.get(self.ref_symbol)

            if df is None or ref_df is None or df.empty or ref_df.empty:
                logger.warning(f"[{symbol}] Sinyal işleme için yeterli veri bulunamadı, atlanıyor.")
                return

            # Kilit mekanizmasını `process_and_enrich_signals` fonksiyonuna devretmek yerine burada yönetebiliriz.
            # Ancak `create_signal` zaten kendi içinde atomik olmalı. Şimdilik kilitsiz devam edelim.
            # async with self.db_lock:
            await process_and_enrich_signals(
                symbol=symbol,
                df=df.copy(),
                ref_df=ref_df.copy(),
                interval=self.interval
            )
        except Exception as e:
            logger.error(f"Sinyal işleme ana hatası - {symbol}: {e}", exc_info=True)

    async def start_streams(self):
        """Starts the WebSocket streams for all symbols using a single combined stream."""
        if not self.symbols:
            logger.warning("İzlenecek sembol kalmadı, WebSocket başlatılmıyor.")
            return

        logger.info(f"WebSocket yayınları başlatılıyor: {self.symbols}")
        # Combined stream (tek bağlantı) için stream listesi
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        
        # Yeni bir ws istemcisi oluşturmak genellikle daha temiz bir yeniden bağlanma sağlar
        self.ws_client = UMFuturesWebsocketClient(
            on_message=self._handle_websocket_message,
            on_close=self._handle_ws_close,
            on_error=self._handle_ws_error
        )
        # Not: binance-futures-connector, bir liste geçildiğinde combined stream açar (tek WS).
        # Ping/pong için client internal ayarları kullanır; watchdog'u biz üstten izliyoruz.
        self.ws_client.subscribe(stream=streams, id=1)
        self.is_ws_connected = True
        self.reconnect_attempt = 0  # Başarılı bağlantıda backoff'u sıfırla
        logger.info("Tüm WebSocket yayınlarına başarıyla abone olundu.")

    async def run(self):
        """Ana çalıştırma döngüsü."""
        try:
            # 1. Eksik geçmiş verileri tamamla
            await self.sync_historical_data()
            # 2. Sinyal hesaplaması için hafızadaki DataFrame'leri ilk verilerle doldur
            await self._initialize_dataframes()
            # 3. Canlı veri akışını başlat (bu bloklamaz)
            await self.start_streams()
            
            logger.info("Canlı veri yöneticisi çalışıyor. Bağlantı izleniyor... Çıkmak için CTRL+C.")

            # Başlangıçta son mesaj zamanını ayarla
            if self.is_ws_connected:
                self.last_message_time = self.loop.time()

            while True:
                reconnect_reason = None
                if not self.is_ws_connected:
                    reconnect_reason = "WebSocket bağlantısı koptu."
                elif self.last_message_time and (self.loop.time() - self.last_message_time) > Config.WEBSOCKET_TIMEOUT:
                    reconnect_reason = f"WebSocket zaman aşımına uğradı ({Config.WEBSOCKET_TIMEOUT}s)."

                if reconnect_reason:
                    # Üstel backoff + jitter
                    backoff_base = getattr(Config, 'WS_RECONNECT_BACKOFF_BASE', 5)
                    backoff_max = getattr(Config, 'WS_RECONNECT_BACKOFF_MAX', 60)
                    delay = min(backoff_max, backoff_base * (2 ** self.reconnect_attempt))
                    # Basit jitter: +/- 20%
                    jitter = max(1.0, delay * 0.2)
                    import random
                    sleep_for = max(1.0, delay + random.uniform(-jitter, jitter))
                    logger.warning(f"{reconnect_reason} {sleep_for:.1f} saniye içinde yeniden bağlanma denenecek... (attempt={self.reconnect_attempt})")
                    self.is_ws_connected = False  # Yeniden bağlanma sürecini başlatmak için
                    await asyncio.sleep(sleep_for)
                    try:
                        # Eski istemciyi durdurmayı dene, ancak takılırsa devam et
                        if self.ws_client:
                           await asyncio.to_thread(self.ws_client.stop)
                        
                        logger.info("Yeni WebSocket bağlantısı kuruluyor...")
                        await self.start_streams()
                        # Yeniden bağlandıktan sonra zamanı sıfırla
                        if self.is_ws_connected:
                           self.last_message_time = self.loop.time()
                           self.reconnect_attempt = 0
                           logger.info("WebSocket bağlantısı başarıyla yeniden kuruldu.")
                        else:
                           self.reconnect_attempt += 1
                           logger.error("Yeniden bağlanma denemesi başarısız oldu.")

                    except Exception as e:
                        logger.error(f"WebSocket yeniden başlatma sırasında kritik hata: {e}", exc_info=True)
                        self.reconnect_attempt += 1
                        # Hata durumunda da backoff uygulanır
                        backoff_base = getattr(Config, 'WS_RECONNECT_BACKOFF_BASE', 5)
                        backoff_max = getattr(Config, 'WS_RECONNECT_BACKOFF_MAX', 60)
                        delay = min(backoff_max, backoff_base * (2 ** self.reconnect_attempt))
                        jitter = max(1.0, delay * 0.2)
                        import random
                        sleep_for = max(1.0, delay + random.uniform(-jitter, jitter))
                        logger.info(f"{sleep_for:.1f} saniye sonra tekrar denenecek.")
                        await asyncio.sleep(sleep_for)
                else:
                    # Bağlantı sağlamsa, döngüyü tıkamadan bekle
                    # Ping/Pong watchdog: belirli aralıkla heartbeat kontrolü
                    await asyncio.sleep(getattr(Config, 'WS_HEARTBEAT_CHECK_INTERVAL', 5))
        except asyncio.CancelledError:
            logger.info("Ana çalıştırma döngüsü iptal edildi.")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Tüm görevleri ve servisleri düzgünce kapatır."""
        logger.info("Kapatma işlemi başlatılıyor...")
        # WebSocket istemcisini durdur (non-blocking)
        if self.ws_client:
            await asyncio.to_thread(self.ws_client.stop)
        logger.info("WebSocket istemcisi durduruldu.")

        # Sadece bizim oluşturduğumuz işlem görevlerini iptal et
        tasks = list(self.processing_tasks)
        if tasks:
            logger.info(f"{len(tasks)} adet bekleyen görev iptal ediliyor...")
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Tüm bekleyen görevler başarıyla iptal edildi.")
        else:
            logger.info("İptal edilecek bekleyen görev bulunamadı.")

async def main():
    """Uygulamanın ana giriş noktası."""
    # Veritabanını ve tabloları oluştur
    await initialize_database()

    logger.info("En yüksek hacimli semboller Binance'ten çekiliyor...")
    symbols_to_track = await BinanceClientManager.get_top_volume_symbols_async(limit=Config.SYMBOL_LIMIT)
    # Referans sembolün izleme listesinde olduğundan emin ol
    if Config.MARKET_REFERENCE_SYMBOL not in symbols_to_track:
        symbols_to_track.insert(0, Config.MARKET_REFERENCE_SYMBOL)
    logger.info(f"{len(symbols_to_track)} adet sembol bulundu.")

    if not symbols_to_track:
        logger.error("İzlenecek sembol bulunamadı. Binance API veya bağlantı sorunu olabilir.")
        return

    manager = LiveDataManager(symbols=symbols_to_track, interval=Config.KLINE_INTERVAL)
    main_task = asyncio.create_task(manager.run())

    try:
        await main_task
    except asyncio.CancelledError:
        logger.info("Ana görev (main_task) iptal edildi.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Program kullanıcı tarafından sonlandırıldı.")
