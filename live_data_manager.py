import asyncio
import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import pandas as pd

from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from binance_client import BinanceClientManager
from utils.exceptions import BinanceAPIError
from indicators.core import add_all_indicators
from database.crud import (
    bulk_insert_price_data,
    get_last_timestamp,
    get_recent_klines,
    initialize_database,
    delete_symbol_data,
)
from database.engine import get_session
from sqlalchemy import text
from signals.signal_processor import process_and_enrich_signals
from utils.exceptions import BinanceAPIError, DatabaseError
from config import Config
from utils.redis_client import RedisClient
from utils.timeframe_aggregator import TimeframeAggregator

# MTF init/refresh için ayrı thread pool — default executor'ı (WS sinyalleri) bloklamaz
_MTF_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mtf_init")


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
        os.path.join(log_dir, "live_data_manager.log"),
        maxBytes=Config.LOG_FILE_MAX_SIZE,
        backupCount=Config.LOG_FILE_BACKUP_COUNT,
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
            symbols.insert(0, self.ref_symbol)  # Başa ekle
        self.symbols = list(dict.fromkeys(symbols))  # Duplike varsa kaldır

        self.interval = interval
        
        # MTF Configuration
        self.mtf_enabled = getattr(Config, 'MTF_ENABLED', True)
        self.supported_timeframes = getattr(Config, 'MTF_TIMEFRAMES', ['1m', '5m', '15m'])
        self.mtf_buffer_limits = getattr(Config, 'MTF_BUFFER_LIMITS', {
            '1m': 1000,   # 16+ hours
            '5m': 200,    # 16+ hours  
            '15m': 67,    # 16+ hours
            '1h': 24,     # 24 hours
            '4h': 12,     # 48 hours
            '1d': 7       # 7 days
        })
        # Multi-WebSocket istemcileri: Her connection için ayrı client
        self.ws_clients: Dict[int, Any] = {}  # connection_id -> ws_client
        self.is_ws_connected = False
        self.last_message_time: Optional[float] = (
            None  # Son WebSocket mesajının zamanını takip et
        )
        self.reconnect_attempt = 0  # Üstel backoff için sayaç
        self.connection_reset_count = 0  # Connection reset sayacı
        self.last_error_type = None  # Son hata türü
        self.consecutive_errors = 0  # Ardışık hata sayısı
        self.db_lock = asyncio.Lock()  # Veritabanı yazma işlemleri için kilit
        self._startup_lookback_days: float = 1.0
        self._startup_fill_end_ms: int = 0

        # Multi-WebSocket configuration
        self.max_streams_per_connection = 200  # Binance limit
        
        # Keep-Alive Ping/Pong Tracking
        self.ping_task: Optional[asyncio.Task] = None
        self.last_ping_time: Optional[float] = None
        self.ping_interval = getattr(Config, 'WS_PING_INTERVAL', 20)
        self.connection_health_ok = True
        
        # Legacy single timeframe buffer (backward compatibility)
        self.kline_data: Dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame() for symbol in symbols
        }
        
        # NEW: Multi-timeframe buffers
        if self.mtf_enabled:
            self.mtf_buffers: Dict[str, Dict[str, pd.DataFrame]] = {}
            for symbol in symbols:
                self.mtf_buffers[symbol] = {}
                for tf in self.supported_timeframes:
                    self.mtf_buffers[symbol][tf] = pd.DataFrame()
            logger.info(f"MTF buffers initialized for {len(symbols)} symbols, {len(self.supported_timeframes)} timeframes")
        
        self.processing_tasks: set[asyncio.Task] = set()
        self._tick_last_sent: Dict[str, float] = {}

        # Batch insert için buffer sistemi
        self.kline_buffer: List[Dict] = []  # Bekleyen kline verilerini toplar
        self.buffer_lock = asyncio.Lock()  # Buffer erişimi için kilit
        self.batch_size = 100  # Kaç kline toplandığında insert yapılacak
        self.batch_timeout = 30  # Saniye - timeout sonrası zorla flush
        self.last_flush_time: Optional[float] = None  # Son flush zamanı

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

        Hızlı paralel işleme ile optimum performans.
        """
        logger.info("Tarihsel veri senkronizasyonu başlatılıyor...")

        # Paralel işleme - maksimum hız için
        # Semaphore ile eşzamanlı istek sayısını kontrol et
        semaphore = asyncio.Semaphore(2)  # Aynı anda max 2 istek (arka plan görevi, rate limit dostu)

        async def sync_with_semaphore(symbol):
            async with semaphore:
                try:
                    await self._sync_symbol_data(symbol)
                    logger.info(f"[{symbol}] Tarihsel veri senkronizasyonu tamamlandı.")
                    return True
                except Exception as e:
                    logger.error(
                        f"[{symbol}] Tarihsel veri senkronizasyonu sırasında hata: {e}"
                    )
                    return False

        # Tüm sembolleri paralel olarak işle
        tasks = [sync_with_semaphore(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_count = sum(1 for r in results if r is True)
        failed_count = len(results) - successful_count

        logger.info(
            f"Tarihsel veri senkronizasyonu tamamlandı. Başarılı: {successful_count}, Başarısız: {failed_count}"
        )

    async def _sync_symbol_data(self, symbol: str):
        """Helper method to sync historical data for a single symbol."""
        try:
            last_timestamp = await get_last_timestamp(symbol, interval=self.interval)
            start_time = last_timestamp + 1 if last_timestamp else None

            if start_time:
                logger.info(
                    f"[{symbol}] Son kayıt: {pd.to_datetime(start_time - 1, unit='ms')}. Eksik veriler çekiliyor..."
                )
            else:
                logger.info(
                    f"[{symbol}] Veritabanında kayıt bulunamadı. Son 1500 mum çekiliyor..."
                )

            total_inserted = 0
            while True:
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        df_missing = await BinanceClientManager.fetch_klines(
                            symbol=symbol,
                            interval=self.interval,
                            limit=1500,
                            startTime=start_time,
                        )
                        break
                    except BinanceAPIError as e:
                        if "Timeout" in str(e) and attempt < max_retries - 1:
                            await asyncio.sleep(0.5)
                            continue
                        raise

                if df_missing.empty:
                    break

                async with self.db_lock:
                    await bulk_insert_price_data(
                        symbol, df_missing, interval=self.interval
                    )
                total_inserted += len(df_missing)
                logger.info(f"[{symbol}] {len(df_missing)} mum kaydedildi (toplam: {total_inserted})")

                if len(df_missing) < 1500:
                    break

                start_time = int(df_missing["open_time"].iloc[-1]) + 1

            if total_inserted:
                logger.info(f"[{symbol}] Senkronizasyon tamamlandı: {total_inserted} mum eklendi.")
                if self.mtf_enabled:
                    await self._refresh_mtf_redis(symbol)
            else:
                logger.info(f"[{symbol}] Yeni veri bulunamadı, sistem güncel.")

        except DatabaseError as e:
            logger.error(f"[{symbol}] Veritabanı hatası oluştu: {e}")
            raise
        except BinanceAPIError as e:
            logger.error(
                f"[{symbol}] Veri senkronizasyonu sırasında Binance API hatası: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"[{symbol}] Veri senkronizasyonunda beklenmedik hata: {e}",
                exc_info=True,
            )
            raise

    async def _add_to_batch_buffer(self, symbol: str, kline_row: Dict):
        """Kline verisini batch buffer'a ekler ve gerekirse flush yapar."""

        async with self.buffer_lock:
            # Kline verisine symbol ve interval bilgisi ekle
            kline_row["symbol"] = symbol
            kline_row["interval"] = self.interval

            self.kline_buffer.append(kline_row)

            # İlk ekleme ise flush zamanını başlat
            current_time = time.time()
            if self.last_flush_time is None:
                self.last_flush_time = current_time

            time_since_last_flush = current_time - self.last_flush_time

            # Buffer doldu veya timeout geçti ise flush yap
            should_flush = (
                len(self.kline_buffer) >= self.batch_size
                or time_since_last_flush >= self.batch_timeout
            )

            if should_flush:
                await self._flush_batch_buffer()

    async def _flush_batch_buffer(self):
        """Buffer'daki tüm kline verilerini veritabanına toplu olarak yazar."""

        if not self.kline_buffer:
            return

        buffer_copy = self.kline_buffer.copy()
        self.kline_buffer.clear()
        self.last_flush_time = time.time()

        try:
            # Verileri symbol'e göre grupla
            symbol_groups = {}
            for kline in buffer_copy:
                symbol = kline["symbol"]
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(kline)

            # Her symbol için ayrı ayrı batch insert
            for symbol, klines in symbol_groups.items():
                df = pd.DataFrame(klines)
                # symbol ve interval kolonlarını kaldır (bulk_insert_price_data bunları beklemez)
                df = df.drop(["symbol", "interval"], axis=1)

                async with self.db_lock:
                    await bulk_insert_price_data(symbol, df, interval=self.interval)

                logger.info(
                    f"[{symbol}] {len(klines)} adet kline toplu olarak veritabanına kaydedildi."
                )

        except Exception as e:
            logger.error(f"Batch insert hatası: {e}", exc_info=True)
            # Hata durumunda verileri tekrar buffer'a ekle
            async with self.buffer_lock:
                self.kline_buffer.extend(buffer_copy)

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
                # Referans sembolü asla filtreleme
                if symbol != Config.MARKET_REFERENCE_SYMBOL and recent_data["volume"].sum() < Config.MIN_VOLUME_THRESHOLD:
                    logger.info(
                        f"[{symbol}] Düşük hacimli (son 24s hacim < {Config.MIN_VOLUME_THRESHOLD}), izlemeden çıkarılıyor."
                    )
                    symbols_to_remove.append(symbol)
                    # Bu sembol için veritabanından da temizlik yapalım
                    task = asyncio.create_task(self._purge_symbol_data(symbol))
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
                else:
                    df = add_all_indicators(result)
                    self.kline_data[symbol] = df
                    logger.info(
                        f"[{symbol}] {len(df)} adet mum başlangıç verisi olarak yüklendi ve göstergeler hesaplandı."
                    )
            else:
                logger.warning(
                    f"[{symbol}] için başlangıç verisi yüklenemedi veya veri boş, izlemeden çıkarılıyor."
                )
                symbols_to_remove.append(symbol)

        if symbols_to_remove:
            self.symbols = [s for s in self.symbols if s not in symbols_to_remove]
            for s in symbols_to_remove:
                del self.kline_data[s]
            logger.info(
                f"Düşük hacimli/hatalı semboller temizlendi. Güncel izleme listesi: {self.symbols}"
            )

    async def _load_initial_data(self, symbol: str) -> pd.DataFrame:
        """Helper to fetch initial kline data for one symbol."""
        try:
            # We fetch 500 to have enough data for indicators like MA200
            return await BinanceClientManager.fetch_klines(
                symbol, self.interval, limit=500
            )
        except BinanceAPIError as e:
            logger.error(
                f"[{symbol}] Başlangıç verisi çekilirken Binance API hatası: {e}",
                exc_info=True,
            )
            raise  # Hatayı yukarıya ilet
        except Exception as e:
            logger.error(f"[{symbol}] Başlangıç verisi çekilemedi: {e}", exc_info=True)
            raise  # Hatayı yukarıya ilet

    def _handle_websocket_message(self, _, msg: str):
        """WebSocket'ten gelen multi-timeframe mesajları işler."""
        self.last_message_time = self.loop.time()  # Her mesajda zamanı güncelle
        self.connection_health_ok = True  # Mesaj geldi, bağlantı sağlıklı
        logger.debug(f"WebSocket mesajı alındı: {msg}")  # Tam mesaj
        try:
            data = json.loads(msg)
            # Combined stream formatında data nested oluyor
            if "data" in data:
                kline_data = data["data"]
                logger.debug(f"JSON parse edildi, event type: {kline_data.get('e')}")
                if kline_data.get("e") == "kline":
                    kline = kline_data["k"]
                    symbol = kline["s"]
                    interval = kline["i"]  # Timeframe bilgisi (1m, 5m, 15m, etc.)
                    is_closed = kline["x"]

                    logger.debug(f"[{symbol}] {interval} Bar closed (x): {is_closed}")

                    if is_closed:
                        logger.info(f"🕯️ [{symbol}] {interval} mum kapandı. Fiyat: {kline['c']}")
                        # WebSocket thread'inden ana event loop'a güvenli coroutine çağrısı
                        asyncio.run_coroutine_threadsafe(
                            self._update_and_process_symbol_mtf(symbol, interval, kline), self.loop
                        )
                    else:
                        tick_key = f"{symbol}:{interval}"
                        now = time.time()
                        if now - self._tick_last_sent.get(tick_key, 0) >= 2.0:
                            self._tick_last_sent[tick_key] = now
                            asyncio.run_coroutine_threadsafe(
                                self._handle_tick(symbol, interval, kline), self.loop
                            )

        except json.JSONDecodeError:
            logger.error(f"WebSocket'ten bozuk JSON verisi alındı: {msg}")
        except Exception as e:
            logger.error(
                f"WebSocket mesaj işleme hatası: {e} | Mesaj: {msg}", exc_info=True
            )

    async def _handle_tick(self, symbol: str, interval: str, kline_data: Dict) -> None:
        """Açık mumu kapalı buffer'a ekleyerek Redis'e yazar ve pub/sub tetikler."""
        try:
            if symbol not in self.mtf_buffers or interval not in self.mtf_buffers[symbol]:
                return
            buf = self.mtf_buffers[symbol][interval]
            if buf.empty:
                return
            tick_row = {
                "open_time": int(kline_data["t"]),
                "open": float(kline_data["o"]),
                "high": float(kline_data["h"]),
                "low": float(kline_data["l"]),
                "close": float(kline_data["c"]),
                "volume": float(kline_data["v"]),
                "close_time": int(kline_data["T"]),
                "quote_asset_volume": float(kline_data["q"]),
                "number_of_trades": int(kline_data["n"]),
                "taker_buy_base_asset_volume": float(kline_data["V"]),
                "taker_buy_quote_asset_volume": float(kline_data["Q"]),
            }
            limit = self.mtf_buffer_limits.get(interval, 100)
            tick_open_time = tick_row["open_time"]
            base = buf[buf["open_time"] != tick_open_time] if "open_time" in buf.columns else buf
            merged = pd.concat(
                [base, pd.DataFrame([tick_row])], ignore_index=True
            ).tail(limit)
            await RedisClient.set_mtf_klines(symbol, interval, merged)
            await RedisClient.publish_kline_update(symbol, interval)
            logger.debug("[%s] %s tick Redis'e yazıldı", symbol, interval)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("[%s] %s tick hatası: %s", symbol, interval, e)

    async def _update_and_process_symbol_mtf(self, symbol: str, interval: str, kline_data: Dict):
        """
        Multi-timeframe version: Updates the DataFrame for specific timeframe and triggers signal processing.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '15m')
            kline_data: Kline data from WebSocket
        """
        try:
            # Parse kline data
            new_row = {
                "open_time": int(kline_data["t"]),
                "open": float(kline_data["o"]),
                "high": float(kline_data["h"]),
                "low": float(kline_data["l"]),
                "close": float(kline_data["c"]),
                "volume": float(kline_data["v"]),
                "close_time": int(kline_data["T"]),
                "quote_asset_volume": float(kline_data["q"]),
                "number_of_trades": int(kline_data["n"]),
                "taker_buy_base_asset_volume": float(kline_data["V"]),
                "taker_buy_quote_asset_volume": float(kline_data["Q"]),
            }

            # MTF buffer'a ekle (her timeframe için ayrı buffer)
            if self.mtf_enabled and symbol in self.mtf_buffers:
                new_df = pd.DataFrame([new_row])
                existing = self.mtf_buffers[symbol][interval]
                if "open_time" in existing.columns:
                    existing = existing[existing["open_time"] != new_row["open_time"]]
                self.mtf_buffers[symbol][interval] = pd.concat(
                    [existing, new_df], ignore_index=True
                ).drop_duplicates(subset=["open_time"], keep="last")

                # Apply buffer limit
                limit = self.mtf_buffer_limits.get(interval, 100)
                self.mtf_buffers[symbol][interval] = self.mtf_buffers[symbol][interval].tail(limit)

                # Add indicators
                self.mtf_buffers[symbol][interval] = await asyncio.to_thread(
                    add_all_indicators, self.mtf_buffers[symbol][interval]
                )

                # Cache to Redis
                await RedisClient.set_mtf_klines(symbol, interval, self.mtf_buffers[symbol][interval])
                logger.debug(f"[{symbol}] {interval} buffer updated and cached")

            # Legacy 1m buffer için backward compatibility (1m interval'da)
            if interval == '1m':
                new_df = pd.DataFrame([new_row])
                self.kline_data[symbol] = pd.concat(
                    [self.kline_data[symbol], new_df], ignore_index=True
                )
                self.kline_data[symbol] = self.kline_data[symbol].tail(1000)
                self.kline_data[symbol] = await asyncio.to_thread(add_all_indicators, self.kline_data[symbol])

                # Legacy Redis keys
                new_redis_key = f"{Config.REDIS_LIVE_DATA_KEY_PREFIX}:{symbol}:1m"
                await RedisClient.set_df(new_redis_key, self.kline_data[symbol])
                legacy_redis_key = f"{Config.REDIS_LIVE_DATA_KEY_PREFIX}:{symbol}"
                await RedisClient.set_df(legacy_redis_key, self.kline_data[symbol])
                await RedisClient.set_hot_klines(symbol, self.kline_data[symbol])
                await RedisClient.publish_kline_update(symbol, "1m")

            # Batch insert için buffer'a ekle (sadece 1m için - diğer TF'ler opsiyonel)
            if interval == '1m':
                await self._add_to_batch_buffer(symbol, new_row)

            # 4h ve 1d kapanışlarını DB'ye yaz (Redis düşse bile veri kaybolmasın)
            if interval in ('4h', '1d'):
                try:
                    df_bar = pd.DataFrame([{
                        'open_time': new_row['open_time'],
                        'open':   new_row['open'],
                        'high':   new_row['high'],
                        'low':    new_row['low'],
                        'close':  new_row['close'],
                        'volume': new_row['volume'],
                    }])
                    async with self.db_lock:
                        await bulk_insert_price_data(symbol, df_bar, interval=interval)
                    logger.debug(f"[{symbol}] {interval} kapanışı DB'ye kaydedildi")
                except Exception as db_err:
                    logger.warning(f"[{symbol}] {interval} DB yazma hatası: {db_err}")

            # Sinyal üretimi (her timeframe için)
            if self.mtf_enabled:
                # Get reference data for this timeframe
                ref_df = pd.DataFrame()
                if self.ref_symbol in self.mtf_buffers and interval in self.mtf_buffers[self.ref_symbol]:
                    ref_df = self.mtf_buffers[self.ref_symbol][interval].copy()

                # Minimum bar requirements per timeframe
                min_bars = {'1m': 200, '5m': 100, '15m': 67, '1h': 24, '4h': 12, '1d': 7}.get(interval, 100)

                if not ref_df.empty and len(self.mtf_buffers[symbol][interval]) >= min_bars:
                    # Create async task for signal processing
                    task = asyncio.create_task(
                        process_and_enrich_signals(
                            symbol=symbol,
                            df=self.mtf_buffers[symbol][interval].copy(),
                            ref_df=ref_df,
                            interval=interval,
                        )
                    )
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
                    logger.info(f"🎯 [{symbol}] {interval} sinyal üretimi başlatıldı")

        except Exception as e:
            logger.error(
                f"[{symbol}] {interval} veri güncelleme hatası: {e}", exc_info=True
            )

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

    def _handle_ws_close(self, *args):
        """Callback function for when the websocket connection is closed."""
        try:
            # Log any provided close arguments (code, reason, etc.) for diagnostics
            logger.warning(f"WebSocket bağlantısı kapandı. args={args!r}")
            # If the websocket client provides a close code/reason, try to extract
            if args:
                try:
                    # common signatures: (ws, close_status_code, close_msg) or (code, reason)
                    # attempt a best-effort extraction
                    if len(args) >= 2:
                        close_code = args[1]
                        logger.warning(f"WebSocket close code: {close_code}")
                    if len(args) >= 3:
                        close_msg = args[2]
                        logger.warning(f"WebSocket close message: {close_msg}")
                except Exception:
                    logger.debug("Close args couldn't be parsed further.", exc_info=True)
        except Exception as e:
            logger.error(f"_handle_ws_close hata verirken: {e}", exc_info=True)
        finally:
            # Mark connection as down and set an error type for reconnect logic
            self.is_ws_connected = False
            self.last_error_type = "closed"

    def _handle_ws_error(self, error, *args):
        """Callback function for websocket errors."""
        try:
            error_str = str(error)
            logger.error(f"WebSocket hata callback tetiklendi. error={error_str}, args={args!r}")

            # Connection reset hatalarını özel olarak takip et
            if "Connection reset by peer" in error_str or "[Errno 54]" in error_str:
                self.connection_reset_count += 1
                self.last_error_type = "connection_reset"
                logger.error(
                    f"WebSocket connection reset hatası (#{self.connection_reset_count}): {error}"
                )
            elif "timeout" in error_str.lower():
                self.last_error_type = "timeout"
                logger.error(f"WebSocket timeout hatası: {error}")
            else:
                self.last_error_type = "other"
                logger.error(f"WebSocket genel hatası: {error}", exc_info=True)

            # If the error object has attributes like code/reason, log them too
            try:
                if hasattr(error, 'code'):
                    logger.debug(f"Error.code={getattr(error, 'code')}")
                if hasattr(error, 'reason'):
                    logger.debug(f"Error.reason={getattr(error, 'reason')}")
            except Exception:
                logger.debug("Ek hata öznitelikleri alınamadı.", exc_info=True)

            self.consecutive_errors += 1
            self.is_ws_connected = False
        except Exception as e:
            logger.error(f"_handle_ws_error sırasında beklenmedik hata: {e}", exc_info=True)

    async def _process_signal_for_symbol(self, symbol: str):
        """Belirli bir sembol için sinyal hesaplamasını ve zenginleştirmesini tetikler."""
        try:
            if symbol == self.ref_symbol:
                logger.debug(f"[{symbol}] Referans sembol, sinyal üretimi atlanıyor.")
                return  # Referans sembol için sinyal üretme

            df = self.kline_data.get(symbol)
            ref_df = self.kline_data.get(self.ref_symbol)

            logger.debug(
                f"[{symbol}] Sinyal işleme kontrolü - DF: {len(df) if df is not None else 'None'}, Ref DF: {len(ref_df) if ref_df is not None else 'None'}"
            )

            if df is None or ref_df is None or df.empty or ref_df.empty:
                logger.warning(
                    f"[{symbol}] Sinyal işleme için yeterli veri bulunamadı, atlanıyor."
                )
                return

            # Kilit mekanizmasını `process_and_enrich_signals` fonksiyonuna devretmek yerine burada yönetebiliriz.
            # Ancak `create_signal` zaten kendi içinde atomik olmalı. Şimdilik kilitsiz devam edelim.
            # async with self.db_lock:
            await process_and_enrich_signals(
                symbol=symbol,
                df=df.copy(),
                ref_df=ref_df.copy(),
                interval=self.interval,
            )
        except Exception as e:
            logger.error(f"Sinyal işleme ana hatası - {symbol}: {e}", exc_info=True)

    # =============================================================================
    # MULTI-TIMEFRAME FUNCTIONS (NEW!)
    # =============================================================================
    
    async def _update_mtf_data(self, symbol: str, new_row: Dict):
        """
        Updates multi-timeframe buffers when a new 1m bar is received.
        
        Args:
            symbol: Symbol to update
            new_row: New 1m OHLCV data
        """
        try:
            if not self.mtf_enabled or symbol not in self.mtf_buffers:
                return
            
            # Update 1m buffer first
            new_df = pd.DataFrame([new_row])
            self.mtf_buffers[symbol]['1m'] = pd.concat(
                [self.mtf_buffers[symbol]['1m'], new_df], ignore_index=True
            )
            
            # Apply buffer limit for 1m
            limit_1m = self.mtf_buffer_limits.get('1m', 1000)
            self.mtf_buffers[symbol]['1m'] = self.mtf_buffers[symbol]['1m'].tail(limit_1m)
            
            # Aggregate to higher timeframes
            await self._aggregate_and_cache_mtf(symbol)

            # YENİ: MTF Bar Kapanış Kontrolü ve Sinyal Üretimi
            # close_time + 1ms = bir sonraki bar'ın open_time'ı
            # Örnek: 16:30:00-16:30:59 arası bar için close_time=16:30:59999 -> next_open=16:31:00
            # Bu sayede 16:31:00 minute % 15 kontrolünde, 16:30 bar'ının kapandığını anlayabiliriz
            next_bar_open_time = datetime.fromtimestamp((new_row['close_time'] + 1) / 1000)

            # Her timeframe için bar kapanış kontrolü (1m hariç)
            for timeframe in self.supported_timeframes[1:]:  # 1m'i atla
                if self._is_mtf_bar_complete(timeframe, next_bar_open_time):
                    logger.info(f"🕯️ [{symbol}] {timeframe} bar kapandı - sinyal kontrolü başlatılıyor")

                    # Async task olarak sinyal üretimi başlat (blocking olmasın)
                    task = asyncio.create_task(
                        self._generate_mtf_signal_live(symbol, timeframe)
                    )
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
            
            logger.debug(f"[{symbol}] MTF data updated for all timeframes")
            
        except Exception as e:
            logger.error(f"[{symbol}] MTF data update error: {e}", exc_info=True)
    
    async def _aggregate_and_cache_mtf(self, symbol: str):
        """
        1m verisinden 5m, 15m, 1h'yi aggregate eder ve Redis'e yazar.
        4h ve 1d WebSocket'ten direkt geldiği için buradan atlanır —
        bu TF'ler için 1m buffer yetersiz ve WebSocket verisini ezmemek gerekir.
        """
        try:
            df_1m = self.mtf_buffers[symbol]['1m']
            if df_1m.empty:
                return

            # Yalnızca 1m buffer'dan güvenilir şekilde üretilebilen TF'ler
            aggregation_targets = ['5m', '15m', '1h']

            for target_tf in aggregation_targets:
                if not TimeframeAggregator.can_aggregate('1m', target_tf):
                    continue

                aggregated_df = TimeframeAggregator.aggregate_ohlcv(df_1m, '1m', target_tf)

                if not aggregated_df.empty:
                    limit = self.mtf_buffer_limits.get(target_tf, 100)
                    result = await asyncio.to_thread(add_all_indicators, aggregated_df.tail(limit))
                    self.mtf_buffers[symbol][target_tf] = result
                    logger.debug(f"[{symbol}] {target_tf}: {len(aggregated_df)} bar aggregated")

            # Sadece aggregate edilen TF'leri Redis'e yaz (4h/1d'yi ezme)
            for tf in ['1m'] + aggregation_targets:
                df = self.mtf_buffers[symbol].get(tf)
                if df is not None and not df.empty:
                    await RedisClient.set_mtf_klines(symbol, tf, df)

        except Exception as e:
            logger.error(f"[{symbol}] MTF aggregation error: {e}", exc_info=True)
    
    async def _cache_mtf_to_redis(self, symbol: str):
        """
        Caches all MTF data to Redis using new MTF cache functions.
        
        Args:
            symbol: Symbol to cache
        """
        try:
            cached_count = 0
            
            for timeframe in self.supported_timeframes:
                df = self.mtf_buffers[symbol].get(timeframe)
                if df is not None and not df.empty:
                    # Use new MTF cache function
                    success = await RedisClient.set_mtf_klines(symbol, timeframe, df)
                    if success:
                        cached_count += 1
                        logger.debug(f"[{symbol}] {timeframe}: Cached {len(df)} bars to Redis")
                    else:
                        logger.warning(f"[{symbol}] {timeframe}: Failed to cache to Redis")
            
            if cached_count > 0:
                logger.info(f"[{symbol}] MTF cache updated: {cached_count}/{len(self.supported_timeframes)} timeframes")
            
        except Exception as e:
            logger.error(f"[{symbol}] MTF Redis cache error: {e}", exc_info=True)
    
    async def _initialize_mtf_dataframes(self):
        """
        Hibrit batch initialization: Tarihsel tüm TF'leri batch halinde yükle + sonra WebSocket.
        API rate limit safe: 10 sembol/batch, 2 saniye delay.
        """
        if not self.mtf_enabled:
            return

        batch_size = 10  # Her batch'te 10 sembol
        delay_between_batches = 5  # 5 saniye delay (10×3TF×2weight=60/batch → 720 weight/dk, limit 1200)

        total_symbols = len(self.symbols)
        total_batches = (total_symbols + batch_size - 1) // batch_size

        logger.info(f"🚀 MTF Batch Initialization başlatılıyor:")
        logger.info(f"   📊 Toplam: {total_symbols} sembol × {len(self.supported_timeframes)} TF")
        logger.info(f"   📦 Batch: {batch_size} sembol/batch, {total_batches} batch")
        logger.info(f"   ⏱️  Tahmini süre: ~{total_batches * delay_between_batches / 60:.1f} dakika")

        # Sembolleri batch'lere böl
        for i in range(0, total_symbols, batch_size):
            batch = self.symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} sembol yükleniyor...")

            # Batch timeout: 120s — takılı semboller iptal edilerek batch ilerler
            batch_tasks = [asyncio.create_task(self._load_symbol_all_timeframes(s)) for s in batch]
            done, pending = await asyncio.wait(batch_tasks, timeout=120)
            for p in pending:
                p.cancel()
                try:
                    await p
                except (asyncio.CancelledError, Exception):
                    pass
            if pending:
                logger.warning(f"⚠️ Batch {batch_num}: {len(pending)} sembol timeout ile atlandı.")

            results = [t.result() if not t.cancelled() and t.exception() is None else None for t in done]

            # Başarı oranını hesapla
            binance_used = any(r is False for r in results if r is not None)
            success_count = sum(1 for r in results if r is not None)
            logger.info(f"✅ Batch {batch_num}/{total_batches} tamamlandı ({success_count}/{len(batch)} başarılı)")

            # Son batch değilse bekle
            if i + batch_size < total_symbols:
                wait = delay_between_batches if binance_used else 0.5
                await asyncio.sleep(wait)

        logger.info("🎉 MTF Batch Initialization tamamlandı! WebSocket canlı mod başlatılabilir.")

    async def _load_symbol_all_timeframes(self, symbol: str) -> bool:
        """
        Bir sembol için tüm timeframe'leri API'den yükle.

        Args:
            symbol: Sembol adı

        Returns:
            bool: Başarılı ise True
        """
        try:
            # Binance'ten çekilecek TF'ler (1m ve aggregate edilebilecekler hariç)
            binance_timeframe_limits = {
                '1h': 250,   # ~10 gün  — aggregate için 1m yetersiz (15k bar lazım)
                '4h': 250,   # ~41 gün
                '1d': 250,   # ~250 gün
            }

            loaded_count = 0
            binance_call_made = False

            # ── 1m: DB'den yükle ──────────────────────────────────────────────
            limit_1m = max(1500, int(self._startup_lookback_days * 24 * 60))
            df_1m = await get_recent_klines(symbol, "1m", limit_1m)
            if df_1m.empty:
                logger.warning(f"[{symbol}] 1m DB'de yok, Binance'ten çekiliyor...")
                df_1m = await BinanceClientManager.fetch_klines(symbol=symbol, interval="1m", limit=1500)

            if not df_1m.empty:
                loop = asyncio.get_event_loop()
                df_1m_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1m)
                self.mtf_buffers[symbol]['1m'] = df_1m_ind.tail(self.mtf_buffer_limits.get('1m', 250))
                await RedisClient.set_mtf_klines(symbol, '1m', self.mtf_buffers[symbol]['1m'])
                loaded_count += 1

                # ── 5m / 15m: 1m buffer'dan aggregate et (Binance isteği yok) ──
                for agg_tf in ['5m', '15m']:
                    if not TimeframeAggregator.can_aggregate('1m', agg_tf):
                        continue
                    agg_df = TimeframeAggregator.aggregate_ohlcv(df_1m, '1m', agg_tf)
                    if not agg_df.empty:
                        limit = self.mtf_buffer_limits.get(agg_tf, 250)
                        agg_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, agg_df.tail(limit))
                        self.mtf_buffers[symbol][agg_tf] = agg_ind
                        await RedisClient.set_mtf_klines(symbol, agg_tf, agg_ind)
                        loaded_count += 1
                        logger.debug(f"[{symbol}] {agg_tf}: {len(agg_df)} bar aggregated from 1m")

            # ── 1h / 4h / 1d: Redis cache → yoksa Binance ───────────────────
            for tf, limit in binance_timeframe_limits.items():
                # Önce Redis cache'e bak (7 günlük TTL — sistem kısa süreli restart'ta sıfırdan çekmez)
                cached_df = await RedisClient.get_mtf_klines(symbol, tf, limit=limit)
                if cached_df is not None and len(cached_df) >= limit // 2:
                    self.mtf_buffers[symbol][tf] = cached_df.drop_duplicates(subset=["open_time"], keep="last")
                    loaded_count += 1
                    logger.debug(f"[{symbol}] {tf}: Redis cache'den yüklendi ({len(cached_df)} bar)")
                    continue

                # Cache miss → önce DB'yi dene (4h/1d artık DB'ye yazılıyor)
                db_df = await get_recent_klines(symbol, tf, limit)
                if db_df is not None and len(db_df) >= limit // 2:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, db_df)
                    buf_limit = self.mtf_buffer_limits.get(tf, 250)
                    self.mtf_buffers[symbol][tf] = df_ind.tail(buf_limit).drop_duplicates(subset=["open_time"], keep="last")
                    await RedisClient.set_mtf_klines(symbol, tf, self.mtf_buffers[symbol][tf])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] {tf}: DB'den yüklendi ({len(db_df)} bar)")
                    continue

                # DB de boş → Binance'ten çek
                binance_call_made = True
                max_retries = 3
                df = pd.DataFrame()
                for retry in range(max_retries):
                    try:
                        df = await BinanceClientManager.fetch_klines(symbol=symbol, interval=tf, limit=limit)
                        break
                    except Exception as e:
                        if "418" in str(e) or "429" in str(e):
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 10
                                logger.warning(f"[{symbol}] {tf} Rate limit, {wait_time}s bekleniyor...")
                                await asyncio.sleep(wait_time)
                                continue
                        raise

                if not df.empty:
                    df_with_indicators = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df)
                    buffer_limit = self.mtf_buffer_limits.get(tf, 250)
                    self.mtf_buffers[symbol][tf] = df_with_indicators.tail(buffer_limit).drop_duplicates(subset=["open_time"], keep="last")
                    await RedisClient.set_mtf_klines(symbol, tf, self.mtf_buffers[symbol][tf])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] {tf}: Binance'ten çekildi ({len(df)} bar)")
                else:
                    logger.warning(f"[{symbol}] {tf}: Veri boş")

            logger.info(f"✅ [{symbol}] {loaded_count} TF yüklendi (1m+5m+15m=aggregated, 1h/4h/1d=cache→Binance)")
            # False döndür → batch'e "Binance kullanıldı" sinyali ver
            return not binance_call_made

        except Exception as e:
            logger.error(f"❌ [{symbol}] Yükleme hatası: {e}", exc_info=False)
            return False
    
    def get_mtf_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Returns MTF data for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe (1m, 5m, 15m, etc.)
            
        Returns:
            DataFrame or None if not available
        """
        if not self.mtf_enabled or symbol not in self.mtf_buffers:
            return None
        
        return self.mtf_buffers[symbol].get(timeframe)
    
    def get_mtf_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Returns statistics about MTF buffers.
        
        Returns:
            Dict with buffer sizes for each symbol and timeframe
        """
        if not self.mtf_enabled:
            return {}
        
        stats: Dict[str, Dict[str, int]] = {}
        for symbol in self.mtf_buffers:
            stats[symbol] = {}
            for tf in self.supported_timeframes:
                df = self.mtf_buffers[symbol].get(tf)
                stats[symbol][tf] = len(df) if df is not None else 0
        
        return stats
    
    def _is_mtf_bar_complete(self, timeframe: str, timestamp: datetime) -> bool:
        """
        MTF bar kapanış kontrolü - timestamp'e göre bar tamamlandı mı?
        
        Args:
            timeframe: Timeframe (5m, 15m, 1h, 4h)
            timestamp: Bar timestamp'i
            
        Returns:
            bool: Bar tamamlandıysa True
        """
        minute = timestamp.minute
        hour = timestamp.hour
        
        if timeframe == '5m':
            return minute % 5 == 0
        elif timeframe == '15m':
            return minute % 15 == 0
        elif timeframe == '1h':
            return minute == 0
        elif timeframe == '4h':
            return minute == 0 and hour % 4 == 0
        elif timeframe == '1d':
            return minute == 0 and hour == 0

        return False
    
    async def _generate_mtf_signal_live(self, symbol: str, timeframe: str):
        """
        Canlı MTF sinyal üretimi - bar kapanışında çalışır
        
        Args:
            symbol: Sembol adı
            timeframe: Timeframe (5m, 15m, 1h, 4h)
        """
        try:
            # MTF buffer'dan veri al
            if symbol not in self.mtf_buffers or timeframe not in self.mtf_buffers[symbol]:
                logger.debug(f"[{symbol}] {timeframe} buffer bulunamadı")
                return
            
            df_mtf = self.mtf_buffers[symbol][timeframe]
            if len(df_mtf) < 200:  # Yeterli veri yok
                logger.debug(f"[{symbol}] {timeframe} yetersiz veri: {len(df_mtf)} < 200")
                return
            
            # Son bar için sinyal kontrol et (mtf_backfill mantığını kullan)
            last_row = df_mtf.iloc[-1]
            signal_data = await self._check_mtf_signal_conditions(last_row, symbol, timeframe)
            
            if signal_data:
                # Mevcut pipeline: process_and_enrich_signals kullan
                # Referans sembolün aynı timeframe MTF verisi (varsa)
                ref_df = pd.DataFrame()
                try:
                    ref_tf_map = self.mtf_buffers.get(self.ref_symbol, {})
                    if isinstance(ref_tf_map, dict):
                        maybe_ref = ref_tf_map.get(timeframe)
                        if maybe_ref is not None and not maybe_ref.empty:
                            ref_df = maybe_ref
                except Exception:
                    ref_df = pd.DataFrame()

                if ref_df.empty:
                    logger.warning(f"[{symbol}] {timeframe} referans DF boş, sinyal işleme atlandı")
                else:
                    await process_and_enrich_signals(
                        symbol=symbol,
                        df=df_mtf.copy(),
                        ref_df=ref_df.copy(),
                        interval=timeframe,
                    )
                    logger.info(f"🎯 [{symbol}] {timeframe} MTF sinyali üretimi tamamlandı (process_and_enrich_signals)")
            else:
                logger.debug(f"[{symbol}] {timeframe} sinyal koşulları sağlanmadı")
                
        except Exception as e:
            await self._handle_mtf_error(symbol, timeframe, e)
    
    async def _check_mtf_signal_conditions(self, row: pd.Series, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        MTF sinyal koşullarını kontrol et (mtf_backfill mantığı)
        
        Args:
            row: DataFrame satırı
            symbol: Sembol adı
            timeframe: Timeframe
            
        Returns:
            Dict: Sinyal verisi veya None
        """
        try:
            # MTF backfill'deki sinyal mantığını kullan
            from backtest.mtf_backfill import MTFBackfillEngine
            
            # Geçici engine oluştur (sadece sinyal kontrolü için)
            temp_engine = MTFBackfillEngine()
            
            # Sinyal koşullarını kontrol et
            signal_data = temp_engine._check_signal_conditions(row, symbol, timeframe)
            
            return signal_data
            
        except Exception as e:
            logger.error(f"[{symbol}] {timeframe} sinyal kontrol hatası: {e}")
            return None
    
    async def _handle_mtf_error(self, symbol: str, timeframe: str, error: Exception):
        """
        MTF hata yönetimi
        
        Args:
            symbol: Sembol adı
            timeframe: Timeframe
            error: Hata objesi
        """
        error_msg = f"[{symbol}] {timeframe} MTF sinyal hatası: {error}"
        logger.error(error_msg, exc_info=True)
        
        # Error counter (basit implementasyon)
        if not hasattr(self, 'mtf_error_count'):
            self.mtf_error_count = 0
        
        self.mtf_error_count += 1
        
        # Circuit breaker pattern
        if self.mtf_error_count > 10:
            logger.warning("MTF circuit breaker activated - çok fazla hata")
            await asyncio.sleep(60)  # 1 dakika bekle
            self.mtf_error_count = 0
    
    async def _keep_alive_ping_loop(self):
        """
        Proaktif keep-alive: WebSocket bağlantısını canlı tutmak için
        periyodik olarak connection health check yapar.
        
        Binance sunucuları idle bağlantıları ~60 dakika sonra kapatıyor.
        Bu task her 20 saniyede kontrol yaparak bağlantının sağlıklı
        kalmasını garantiler.
        """
        logger.info(f"Keep-Alive ping task başlatıldı (interval: {self.ping_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                if not self.is_ws_connected:
                    logger.debug("WebSocket bağlı değil, ping atlanıyor")
                    continue
                
                current_time = self.loop.time()
                
                # Son mesajdan bu yana geçen süre
                if self.last_message_time:
                    time_since_last_msg = current_time - self.last_message_time
                    
                    # Eğer 30 saniyedir mesaj gelmiyorsa proaktif reconnect
                    if time_since_last_msg > 30:
                        logger.warning(
                            f"⚠️ Son mesajdan bu yana {time_since_last_msg:.1f}s geçti. "
                            "Proaktif reconnect tetikleniyor..."
                        )
                        self.connection_health_ok = False
                        self.is_ws_connected = False
                        continue
                    
                    # Health check - her 20 saniyede log
                    logger.debug(
                        f"💚 Keep-Alive Health Check: Bağlantı sağlıklı "
                        f"(son mesaj: {time_since_last_msg:.1f}s önce)"
                    )
                    self.last_ping_time = current_time
                else:
                    logger.debug("Keep-Alive: last_message_time henüz set edilmemiş")
                    
            except asyncio.CancelledError:
                logger.info("Keep-Alive ping task iptal edildi")
                break
            except Exception as e:
                logger.error(f"Keep-Alive ping task hatası: {e}", exc_info=True)
                await asyncio.sleep(5)  # Hata durumunda kısa bekle

    async def start_streams(self):
        """Starts multi-timeframe WebSocket streams for all symbols with multiple connections."""
        if not self.symbols:
            logger.warning("İzlenecek sembol kalmadı, WebSocket başlatılmıyor.")
            return

        logger.info(f"🚀 Multi-Timeframe WebSocket başlatılıyor: {len(self.symbols)} sembol × {len(self.supported_timeframes)} TF")

        # Tüm stream'leri oluştur (sembol × timeframe)
        all_streams = []
        for symbol in self.symbols:
            for tf in self.supported_timeframes:
                all_streams.append(f"{symbol.lower()}@kline_{tf}")

        total_streams = len(all_streams)
        # Allow override from central config (new tunable)
        self.max_streams_per_connection = getattr(
            Config, 'WS_MAX_STREAMS_PER_CONNECTION', self.max_streams_per_connection
        )
        connections_needed = (
            (total_streams + self.max_streams_per_connection - 1)
            // self.max_streams_per_connection
        )

        logger.info(f"📊 Toplam stream: {total_streams} ({len(self.symbols)} sembol × {len(self.supported_timeframes)} TF)")
        logger.info(f"🔌 Gerekli connection: {connections_needed} (max {self.max_streams_per_connection} stream/connection)")

        try:
            # Eski bağlantıları güvenli şekilde kapat
            await self._safe_close_websocket()

            # Stream'leri connection'lara böl (her connection max 200 stream)
            stream_chunks = [
                all_streams[i:i + self.max_streams_per_connection]
                for i in range(0, total_streams, self.max_streams_per_connection)
            ]

            # Her chunk için ayrı WebSocket connection oluştur
            for connection_id, streams in enumerate(stream_chunks):
                logger.info(f"🔌 Connection #{connection_id + 1}: {len(streams)} stream subscribe ediliyor...")

                # python-binance ile WebSocket client oluştur
                # Use configured Binance websocket base so we can target /market endpoints
                stream_url = getattr(Config, 'BINANCE_WS_BASE', None)
                if stream_url:
                    logger.info(f"Using custom Binance WS base: {stream_url} for connection #{connection_id + 1}")

                if stream_url:
                    ws_client = UMFuturesWebsocketClient(
                        stream_url=stream_url,
                        on_message=self._handle_websocket_message,
                        on_close=self._handle_ws_close,
                        on_error=self._handle_ws_error,
                        is_combined=True,  # Combined streams kullan
                    )
                else:
                    ws_client = UMFuturesWebsocketClient(
                        on_message=self._handle_websocket_message,
                        on_close=self._handle_ws_close,
                        on_error=self._handle_ws_error,
                        is_combined=True,  # Combined streams kullan
                    )

                # Bu connection'daki tüm stream'lere subscribe et
                ws_client.subscribe(stream=streams, id=connection_id + 1)

                # Client'ı sakla
                self.ws_clients[connection_id] = ws_client

                logger.info(f"✅ Connection #{connection_id + 1} başarıyla kuruldu ({len(streams)} stream)")
                # Stagger connection subscriptions slightly to avoid server-side burst handling
                await asyncio.sleep(0.25)

            # Bağlantılar kurulduktan sonra kısa bir bekleme
            await asyncio.sleep(2)

            self.is_ws_connected = True
            self.reconnect_attempt = 0  # Başarılı bağlantıda backoff'u sıfırla
            self.connection_health_ok = True  # Health durumunu sıfırla
            logger.info(f"🎉 Multi-WebSocket başarıyla başlatıldı: {connections_needed} connection, {total_streams} stream")

            # Keep-Alive ping task'ını başlat
            await self._start_ping_task()

        except Exception as e:
            logger.error(f"Multi-WebSocket başlatma hatası: {e}", exc_info=True)
            self.is_ws_connected = False
            raise

    async def _deferred_sync_historical(self, delay_seconds: int = 30):
        """sync_historical_data'yı gecikmeyle arka planda çalıştırır."""
        await asyncio.sleep(delay_seconds)
        logger.info(f"🔄 Tarihsel veri senkronizasyonu başlatılıyor (arka plan, {delay_seconds}s sonra)...")
        await self.sync_historical_data()

    async def _deferred_internal_gap_check(self, delay_seconds: int = 90) -> None:
        """WebSocket bağlantısından sonra son 1 saatte oluşan iç gap'leri doldurur.

        _deferred_sync_historical sadece kuyruktan doldurur; WebSocket başlamadan önce
        oluşan iç gap'leri (örn. PostInit penceresi) bu metot yakalar.
        """
        import time as _time
        await asyncio.sleep(delay_seconds)
        logger.info("[DeferredGapCheck] Son 1 saatin iç gap analizi başlıyor...")
        symbols_list = list(self.symbols)
        _INTERVAL_MS = 60_000

        try:
            async with get_session() as session:
                r = await session.execute(
                    text("""
                        SELECT symbol, prev_ts, curr_ts
                        FROM (
                            SELECT symbol, timestamp AS curr_ts,
                                   LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                            FROM price_data
                            WHERE symbol = ANY(:syms) AND interval = '1m'
                              AND timestamp >= NOW() - INTERVAL '1 hour'
                        ) t
                        WHERE prev_ts IS NOT NULL
                          AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > 90000
                        ORDER BY symbol, prev_ts
                    """),
                    {"syms": symbols_list},
                )
                rows = r.fetchall()
        except Exception as exc:
            logger.warning("[DeferredGapCheck] Sorgu hatası: %s", exc)
            return

        gaps_by_sym: dict = {}
        for sym, prev_dt, curr_dt in rows:
            g_start = int(prev_dt.timestamp() * 1000)
            g_end = int(curr_dt.timestamp() * 1000)
            gaps_by_sym.setdefault(sym, []).append((g_start, g_end))

        if not gaps_by_sym:
            logger.info("[DeferredGapCheck] Gap yok, sistem temiz.")
            return

        total_gaps = sum(len(g) for g in gaps_by_sym.values())
        logger.info("[DeferredGapCheck] %d sembolde %d gap bulundu.", len(gaps_by_sym), total_gaps)
        total_filled = 0

        for sym, gaps in gaps_by_sym.items():
            sym_filled = 0
            for gap_start_ms, gap_end_ms in gaps:
                fetch_start = gap_start_ms + _INTERVAL_MS
                while fetch_start < gap_end_ms:
                    await asyncio.sleep(0.5)
                    try:
                        df = await BinanceClientManager.fetch_klines(
                            symbol=sym, interval="1m", limit=1000, startTime=fetch_start,
                        )
                    except Exception:
                        break
                    if df is None or df.empty:
                        break
                    df = df[df["open_time"] < gap_end_ms]
                    if df.empty:
                        break
                    async with self.db_lock:
                        await bulk_insert_price_data(sym, df, interval="1m")
                    sym_filled += len(df)
                    last_ts = int(df["open_time"].iloc[-1])
                    if last_ts <= fetch_start or len(df) < 1000:
                        break
                    fetch_start = last_ts + _INTERVAL_MS

            if sym_filled > 0 and self.mtf_enabled:
                await self._refresh_mtf_redis(sym)
            total_filled += sym_filled

        logger.info("[DeferredGapCheck] Tamamlandı: %d bar eklendi.", total_filled)

    async def _startup_gap_fill(self) -> None:
        """Startup gap fill: 1m için dinamik lookback ile gap doldurur."""
        import time as _time

        _INTERVAL_MS = 60_000
        _THRESHOLD_MS = _INTERVAL_MS * 2
        _MAX_LOOKBACK_DAYS = 30
        symbols_list = list(self.symbols)

        # Dinamik lookback: tüm sembollerin son 1m kaydına bak.
        # NOT: EXTRACT(EPOCH FROM naive_ts) naive'i UTC gibi işler (PostgreSQL davranışı).
        # Python .timestamp() ise sistem TZ (+3) ile doğru UTC epoch'u verir.
        # Bu yüzden raw timestamp döndürüp Python'da dönüştürüyoruz.
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT MAX(timestamp)
                        FROM price_data
                        WHERE symbol = ANY(:syms) AND interval = '1m'
                    """),
                    {"syms": symbols_list},
                )
                row = result.fetchone()
                last_dt = row[0] if row and row[0] else None
                last_1m_ms = int(last_dt.timestamp() * 1000) if last_dt else None
        except Exception as exc:
            logger.warning("[Startup] Son kayıt sorgusu başarısız: %s", exc)
            last_1m_ms = None

        now_ms = int(_time.time() * 1000)
        if last_1m_ms:
            offline_ms = max(now_ms - last_1m_ms, 0)
            lookback_days = min(max(offline_ms / 86_400_000, 1), _MAX_LOOKBACK_DAYS)
            logger.info(
                "[Startup] Çevrimdışı süre: %.1f saat → %g günlük gap analizi",
                offline_ms / 3_600_000,
                round(lookback_days, 1),
            )
        else:
            lookback_days = 1

        # --- 1m gap fill: iç gap'ler (LAG sorgusu, raw timestamp → Python dönüşümü) ---
        logger.info("[Startup] 1m gap analizi yapılıyor...")
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT symbol, prev_ts, curr_ts
                        FROM (
                            SELECT symbol,
                                   timestamp AS curr_ts,
                                   LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                            FROM price_data
                            WHERE symbol = ANY(:syms) AND interval = '1m'
                              AND timestamp >= NOW() - (:days * INTERVAL '1 day')
                        ) t
                        WHERE prev_ts IS NOT NULL
                          AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > :thresh
                        ORDER BY symbol, prev_ts
                    """),
                    {"syms": symbols_list, "days": lookback_days, "thresh": _THRESHOLD_MS},
                )
                rows = result.fetchall()
        except Exception as exc:
            logger.warning("[Startup] 1m gap analizi başarısız: %s", exc)
            rows = []

        all_gaps: dict[str, list[tuple[int, int]]] = {}
        for sym, prev_dt, curr_dt in rows:
            g_start = int(prev_dt.timestamp() * 1000)
            g_end = int(curr_dt.timestamp() * 1000)
            all_gaps.setdefault(sym, []).append((g_start, g_end))

        # --- Kuyruk gap'i: her sembol için son kayıt → şu an ---
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT symbol, MAX(timestamp)
                        FROM price_data
                        WHERE symbol = ANY(:syms) AND interval = '1m'
                        GROUP BY symbol
                    """),
                    {"syms": symbols_list},
                )
                tail_rows = result.fetchall()
        except Exception as exc:
            logger.warning("[Startup] Kuyruk gap sorgusu başarısız: %s", exc)
            tail_rows = []

        for sym, last_dt in tail_rows:
            tail_ms = int(last_dt.timestamp() * 1000)
            if (now_ms - tail_ms) > _THRESHOLD_MS:
                existing = all_gaps.get(sym, [])
                if not any(gs >= tail_ms for gs, _ in existing):
                    all_gaps.setdefault(sym, []).append((tail_ms, now_ms))

        if all_gaps:
            total_gaps = sum(len(g) for g in all_gaps.values())
            logger.info("[Startup] 1m: %d sembolde %d gap bulundu, dolduruluyor...", len(all_gaps), total_gaps)
            total_filled = 0

            for sym, gaps in all_gaps.items():
                for gap_start_ms, gap_end_ms in gaps:
                    fetch_start = gap_start_ms + _INTERVAL_MS
                    while fetch_start < gap_end_ms:
                        await asyncio.sleep(0.5)
                        try:
                            df = await BinanceClientManager.fetch_klines(
                                symbol=sym, interval="1m", limit=1000, startTime=fetch_start,
                            )
                        except Exception:
                            break
                        if df is None or df.empty:
                            break
                        df = df[df["open_time"] < gap_end_ms]
                        if df.empty:
                            break
                        async with self.db_lock:
                            await bulk_insert_price_data(sym, df, interval="1m")
                        total_filled += len(df)
                        last_ts = int(df["open_time"].iloc[-1])
                        if last_ts <= fetch_start or len(df) < 1000:
                            break
                        fetch_start = last_ts + _INTERVAL_MS

            logger.info("[Startup] 1m gap fill tamamlandı: %d bar eklendi", total_filled)
        else:
            logger.info("[Startup] 1m: gap yok.")

        self._startup_fill_end_ms = int(_time.time() * 1000)
        self._startup_lookback_days = float(lookback_days)

    async def _refresh_mtf_redis(self, symbol: str) -> None:
        """1m DB verisinden 5m/15m aggregate ederek Redis'i günceller."""
        if not self.mtf_enabled or symbol not in self.mtf_buffers:
            return
        try:
            loop = asyncio.get_event_loop()
            limit_1m = max(1500, int(self._startup_lookback_days * 24 * 60))
            df_1m = await get_recent_klines(symbol, "1m", limit_1m)
            if df_1m.empty:
                return
            df_1m_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1m)
            self.mtf_buffers[symbol]["1m"] = df_1m_ind.tail(self.mtf_buffer_limits.get("1m", 1000))
            await RedisClient.set_mtf_klines(symbol, "1m", self.mtf_buffers[symbol]["1m"])
            for agg_tf in ["5m", "15m"]:
                if not TimeframeAggregator.can_aggregate("1m", agg_tf):
                    continue
                agg_df = TimeframeAggregator.aggregate_ohlcv(df_1m, "1m", agg_tf)
                if agg_df.empty:
                    continue
                limit = self.mtf_buffer_limits.get(agg_tf, 250)
                agg_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, agg_df.tail(limit * 2))
                self.mtf_buffers[symbol][agg_tf] = agg_ind.tail(limit)
                await RedisClient.set_mtf_klines(symbol, agg_tf, self.mtf_buffers[symbol][agg_tf])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[MTF-Refresh] %s hata: %s", symbol, exc)

    async def _post_init_catchup(self) -> None:
        """MTF init sonrası startup penceresindeki 1m gap'leri doldurur ve MTF Redis'i günceller."""
        import time as _time

        if not self._startup_fill_end_ms:
            return

        now_ms = int(_time.time() * 1000)
        window_ms = now_ms - self._startup_fill_end_ms
        if window_ms < 60_000:
            return

        logger.info("[PostInit] Startup penceresi gap fill başlıyor (%.1f dk)...", window_ms / 60_000)
        symbols_list = list(self.symbols)

        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT symbol, prev_ts, curr_ts
                        FROM (
                            SELECT symbol,
                                   timestamp AS curr_ts,
                                   LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                            FROM price_data
                            WHERE symbol = ANY(:syms) AND interval = '1m'
                              AND timestamp >= NOW() - INTERVAL '2 hours'
                        ) t
                        WHERE prev_ts IS NOT NULL
                          AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > 90000
                        ORDER BY symbol, prev_ts
                    """),
                    {"syms": symbols_list},
                )
                gap_rows = result.fetchall()

                result2 = await session.execute(
                    text("""
                        SELECT symbol, MAX(timestamp)
                        FROM price_data
                        WHERE symbol = ANY(:syms) AND interval = '1m'
                        GROUP BY symbol
                    """),
                    {"syms": symbols_list},
                )
                tail_rows = {row[0]: int(row[1].timestamp() * 1000) for row in result2.fetchall() if row[1]}
        except Exception as exc:
            logger.warning("[PostInit] Gap sorgusu başarısız: %s", exc)
            return

        catchup_gaps: dict[str, list[tuple[int, int]]] = {}
        for sym, prev_dt, curr_dt in gap_rows:
            g_start = int(prev_dt.timestamp() * 1000)
            g_end = int(curr_dt.timestamp() * 1000)
            catchup_gaps.setdefault(sym, []).append((g_start, g_end))

        for sym, last_ms in tail_rows.items():
            if (now_ms - last_ms) > 90_000:
                existing = catchup_gaps.get(sym, [])
                if not any(gs >= last_ms for gs, _ in existing):
                    catchup_gaps.setdefault(sym, []).append((last_ms, now_ms))

        if not catchup_gaps:
            logger.info("[PostInit] Startup penceresi temiz, gap yok.")
            return

        total_gaps = sum(len(g) for g in catchup_gaps.values())
        logger.info("[PostInit] %d sembolde %d gap bulundu, dolduruluyor...", len(catchup_gaps), total_gaps)

        _INTERVAL_MS = 60_000
        total = 0

        for sym, gaps in catchup_gaps.items():
            sym_filled = 0
            for gap_start_ms, gap_end_ms in gaps:
                fetch_start = gap_start_ms + _INTERVAL_MS
                while fetch_start < gap_end_ms:
                    await asyncio.sleep(0.5)
                    try:
                        df = await BinanceClientManager.fetch_klines(
                            symbol=sym, interval="1m", limit=1000, startTime=fetch_start,
                        )
                    except Exception:
                        break
                    if df is None or df.empty:
                        break
                    df = df[df["open_time"] < gap_end_ms]
                    if df.empty:
                        break
                    async with self.db_lock:
                        await bulk_insert_price_data(sym, df, interval="1m")
                    sym_filled += len(df)
                    last_ts = int(df["open_time"].iloc[-1])
                    if last_ts <= fetch_start or len(df) < 1000:
                        break
                    fetch_start = last_ts + _INTERVAL_MS

            if sym_filled > 0 and self.mtf_enabled:
                await self._refresh_mtf_redis(sym)
            total += sym_filled

        logger.info("[PostInit] Tamamlandı: %d bar eklendi", total)

    async def _health_loop(self):
        """Her 15 dakikada price_data tazeliğini kontrol eder; gap tespit ederse doldurur."""
        _CHECK_INTERVAL = 15 * 60
        _MAX_GAP_MS = 10 * 60 * 1000  # 10 dakika (ms cinsinden)

        await asyncio.sleep(60)  # Başlangıç stabilizasyonu için bekle

        while True:
            await asyncio.sleep(_CHECK_INTERVAL)
            stale: list[str] = []
            now_ms = time.time() * 1000

            for symbol in list(self.symbols):
                try:
                    last_ts = await get_last_timestamp(symbol, interval="1m")
                    if last_ts is None:
                        continue
                    if now_ms - last_ts > _MAX_GAP_MS:
                        gap_min = (now_ms - last_ts) / 60_000
                        logger.warning(
                            "[Health] %s — %.1f dakika gap tespit edildi, dolduruluyor",
                            symbol, gap_min,
                        )
                        stale.append(symbol)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug("[Health] %s timestamp kontrolü hatası: %s", symbol, exc)

            for symbol in stale:
                asyncio.create_task(self._sync_symbol_data(symbol))

            if stale:
                logger.info("[Health] %d sembol için gap fill başlatıldı", len(stale))
            else:
                logger.debug("[Health] Tüm semboller güncel")

    async def _background_startup(self) -> None:
        """WebSocket başladıktan sonra MTF init + gap fill arka planda çalışır.

        Eski mimari: gap_fill → mtf_init → post_init → WS  (20+ dk gap)
        Yeni mimari: WS → bu task arka planda  (~0 dk gap)
        """
        logger.info("[BackgroundStartup] Başladı (WebSocket zaten aktif).")
        if self.mtf_enabled:
            await self._initialize_mtf_dataframes()
        await self._startup_gap_fill()
        await self._post_init_catchup()
        logger.info("[BackgroundStartup] Tamamlandı.")

    async def run(self):
        """Ana çalıştırma döngüsü."""
        try:
            # 1. WebSocket önce başlat — canlı veri hemen akar, gap oluşmaz
            await self.start_streams()
            # 2. MTF init + gap fill arka planda (WS'yi bloklamaz)
            asyncio.create_task(self._background_startup())
            # 3. 30s sonra kuyruk gap fill (son timestamp → şu an)
            asyncio.create_task(self._deferred_sync_historical(delay_seconds=30))
            # 4. 120s sonra iç gap kontrolü (startup penceresi LAG analizi)
            asyncio.create_task(self._deferred_internal_gap_check(delay_seconds=120))
            # 5. Periyodik sağlık kontrolü
            asyncio.create_task(self._health_loop())

            logger.info(
                "Canlı veri yöneticisi çalışıyor. Bağlantı izleniyor... Çıkmak için CTRL+C."
            )

            # Başlangıçta son mesaj zamanını ayarla
            if self.is_ws_connected:
                self.last_message_time = self.loop.time()

            while True:
                reconnect_reason = None
                if not self.is_ws_connected:
                    reconnect_reason = "WebSocket bağlantısı koptu."
                elif (
                    self.last_message_time
                    and (self.loop.time() - self.last_message_time)
                    > Config.WEBSOCKET_TIMEOUT
                ):
                    reconnect_reason = (
                        f"WebSocket zaman aşımına uğradı ({Config.WEBSOCKET_TIMEOUT}s)."
                    )

                if reconnect_reason:
                    # Connection reset hatalarında özel backoff stratejisi
                    if (
                        self.last_error_type == "connection_reset"
                        and self.connection_reset_count
                        >= getattr(Config, "WS_CONNECTION_RESET_THRESHOLD", 5)
                    ):
                        # Çok fazla connection reset varsa daha uzun bekle
                        base_delay = getattr(Config, "WS_RECONNECT_BACKOFF_BASE", 2) * 3
                        max_delay = getattr(Config, "WS_RECONNECT_BACKOFF_MAX", 30) * 2
                    else:
                        base_delay = getattr(Config, "WS_RECONNECT_BACKOFF_BASE", 2)
                        max_delay = getattr(Config, "WS_RECONNECT_BACKOFF_MAX", 30)

                    # Üstel backoff + jitter
                    delay = min(
                        max_delay, base_delay * (2 ** min(self.reconnect_attempt, 6))
                    )
                    # Basit jitter: +/- 20%
                    jitter = max(0.5, delay * 0.2)
                    import random

                    sleep_for = max(1.0, delay + random.uniform(-jitter, jitter))
                    logger.warning(
                        f"{reconnect_reason} {sleep_for:.1f} saniye içinde yeniden bağlanma denenecek... (attempt={self.reconnect_attempt})"
                    )
                    self.is_ws_connected = (
                        False  # Yeniden bağlanma sürecini başlatmak için
                    )
                    await asyncio.sleep(sleep_for)
                    try:
                        logger.info("Yeni WebSocket bağlantısı kuruluyor...")
                        await self.start_streams()
                        # Yeniden bağlandıktan sonra zamanı sıfırla
                        if self.is_ws_connected:
                            self.last_message_time = self.loop.time()
                            # Başarılı bağlantıda sayaçları sıfırla
                            self.reconnect_attempt = 0
                            self.consecutive_errors = 0
                            # Connection reset sayacını kademeli olarak azalt
                            if self.connection_reset_count > 0:
                                self.connection_reset_count = max(
                                    0, self.connection_reset_count - 1
                                )
                            logger.info(
                                f"WebSocket bağlantısı başarıyla yeniden kuruldu. Reset count: {self.connection_reset_count}"
                            )
                            # Reconnect sırasında oluşan gap'leri arka planda doldur
                            asyncio.create_task(self._deferred_sync_historical(delay_seconds=5))
                        else:
                            self.reconnect_attempt += 1
                            logger.error(
                                f"Yeniden bağlanma denemesi başarısız oldu. (Attempt: {self.reconnect_attempt})"
                            )

                    except Exception as e:
                        logger.error(
                            f"WebSocket yeniden başlatma sırasında kritik hata: {e}",
                            exc_info=True,
                        )
                        self.reconnect_attempt += 1
                        self.consecutive_errors += 1

                        # Çok fazla ardışık hata varsa daha uzun bekle
                        if self.consecutive_errors >= getattr(
                            Config, "WS_MAX_RECONNECT_ATTEMPTS", 10
                        ):
                            logger.warning(
                                f"Çok fazla ardışık hata ({self.consecutive_errors}). Uzun bekleme moduna geçiliyor..."
                            )
                            await asyncio.sleep(120)  # 2 dakika bekle
                            self.consecutive_errors = 0  # Sayacı sıfırla
                        else:
                            # Normal backoff uygulanır
                            base_delay = getattr(Config, "WS_RECONNECT_BACKOFF_BASE", 2)
                            max_delay = getattr(Config, "WS_RECONNECT_BACKOFF_MAX", 30)
                            delay = min(
                                max_delay,
                                base_delay * (2 ** min(self.reconnect_attempt, 6)),
                            )
                            jitter = max(0.5, delay * 0.2)
                            import random

                            sleep_for = max(
                                1.0, delay + random.uniform(-jitter, jitter)
                            )
                            logger.info(
                                f"{sleep_for:.1f} saniye sonra tekrar denenecek."
                            )
                            await asyncio.sleep(sleep_for)
                else:
                    # Bağlantı sağlamsa, döngüyü tıkamadan bekle
                    # Ping/Pong watchdog: belirli aralıkla heartbeat kontrolü
                    await asyncio.sleep(
                        getattr(Config, "WS_HEARTBEAT_CHECK_INTERVAL", 5)
                    )
        except asyncio.CancelledError:
            logger.info("Ana çalıştırma döngüsü iptal edildi.")
        finally:
            await self.shutdown()

    async def _start_ping_task(self):
        """Keep-alive ping task'ını başlatır."""
        # Eski task varsa iptal et
        if self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass
        
        # Yeni ping task başlat
        self.ping_task = asyncio.create_task(self._keep_alive_ping_loop())
        logger.info("Keep-Alive ping task başlatıldı")
    
    async def _stop_ping_task(self):
        """Keep-alive ping task'ını durdurur."""
        if self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass
            logger.info("Keep-Alive ping task durduruldu")
    
    async def _safe_close_websocket(self):
        """Tüm WebSocket bağlantılarını güvenli şekilde kapatır."""
        # Önce ping task'ını durdur
        await self._stop_ping_task()

        # Tüm WebSocket client'larını kapat
        if self.ws_clients:
            for connection_id, ws_client in list(self.ws_clients.items()):
                if ws_client:
                    try:
                        # Timeout ile güvenli kapatma
                        await asyncio.wait_for(
                            asyncio.to_thread(ws_client.stop), timeout=5.0
                        )
                        logger.debug(f"WebSocket connection #{connection_id} güvenli şekilde kapatıldı.")
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"WebSocket connection #{connection_id} kapatma işlemi timeout oldu, zorla kapatılıyor."
                        )
                    except Exception as e:
                        logger.warning(
                            f"WebSocket connection #{connection_id} kapatma sırasında hata (göz ardı edildi): {e}"
                        )

            # Tüm client'ları temizle
            self.ws_clients.clear()
            logger.info(f"Tüm WebSocket bağlantıları kapatıldı.")

    async def shutdown(self):
        """Tüm görevleri ve servisleri düzgünce kapatır."""
        logger.info("Kapatma işlemi başlatılıyor...")

        # Buffer'daki kalan verileri flush et
        try:
            await self._flush_batch_buffer()
            logger.info("Buffer verileri başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Buffer flush hatası: {e}")

        # WebSocket istemcisini durdur
        await self._safe_close_websocket()
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
    symbols_to_track = await BinanceClientManager.get_top_volume_symbols_async(
        limit=Config.SYMBOL_LIMIT
    )
    # Referans sembolün izleme listesinde olduğundan emin ol
    if Config.MARKET_REFERENCE_SYMBOL not in symbols_to_track:
        symbols_to_track.insert(0, Config.MARKET_REFERENCE_SYMBOL)
    logger.info(f"{len(symbols_to_track)} adet sembol bulundu.")

    if not symbols_to_track:
        logger.error(
            "İzlenecek sembol bulunamadı. Binance API veya bağlantı sorunu olabilir."
        )
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
