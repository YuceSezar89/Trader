import asyncio
import concurrent.futures
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis

import numpy as np
import pandas as pd

from utils.asyncio_ws_client import AsyncioBinanceStreamManager

from binance_client import BinanceClientManager
from utils.exceptions import BinanceAPIError
from indicators.core import add_all_indicators
from indicators.incremental import IndicatorState, bootstrap_state, update_state, RESYNC_INTERVAL
from database.crud import (
    bulk_insert_price_data,
    bulk_insert_price_data_multi,
    get_cagg_klines,
    get_last_timestamp,
    get_oldest_timestamp,
    get_recent_klines,
    initialize_database,
    delete_symbol_data,
)
from database.engine import get_session, run_with_db_timeout
from sqlalchemy import text
from signals.signal_processor import process_and_enrich_signals
from signals.risk_manager import risk_manager
from signals.paper_trade_manager import paper_trade_manager, ha_cross_manager, rsi_15m_manager, manual_manager, do_kirilimi_manager, do_open_streak_manager
from utils.exceptions import BinanceAPIError, DatabaseError
from config import Config
from utils.kline_schema import check_kline_schema
from utils.timeframe_aggregator import TimeframeAggregator
from utils.redis_client import RedisClient
from utils.heartbeat import beat, record_activity
from utils.telegram_notify import send_telegram_message
from utils.logger import get_logger

# MTF init/refresh için ayrı thread pool — default executor'ı (WS sinyalleri) bloklamaz
# 12→4 (10 Tem 2026): incremental indikatör hesaplama (17.8x hızlanma, ~2.9ms/çağrı)
# sonrası 12 thread'e gerçek paralellik ihtiyacı kalmadı, sadece GIL çekişmesi
# yaratıyorlardı — bu da (WS artık aynı event loop'ta olduğu için) ping/pong
# gecikmesine ve DB timeout'larına yol açıyordu. Bkz. memory: project_data_layer_debt.md.
_MTF_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mtf_init")
_TICK_TF_WHITELIST = {'1m', '5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d'}
_TICK_THROTTLE_SECS = {'1m': 2, '5m': 2, '15m': 2, '30m': 2,
                       '1h': 30, '4h': 60, '6h': 60, '8h': 60, '12h': 120, '1d': 120}

# İndikatör incremental bootstrap'ı için minimum bar sayısı — SuperTrend(ATR=10) ve
# ADX(14+14) bunun altında anlamlı seed alamaz. 4h(limit=12)/1d(limit=7) gibi kısa
# buffer'lı TF'ler bu eşiğin altında kalıp hep tam hesaplamaya düşer (zararsız, ucuz).
_MIN_BOOTSTRAP_BARS = 30



def _merge_tick_row(buf: pd.DataFrame, tick_row: dict, limit: int) -> pd.DataFrame:
    """Forming bar satırını buffer'a ekler — CPU-ağır pandas işlemi, event loop'u
    bloklamaması için executor'da çalıştırılır (bkz. _handle_tick).

    Son satır zaten aynı open_time'a sahipse (forming bar zaten yerinde — en sık
    görülen durum), pd.concat'in blok birleştirme maliyetinden kaçınmak için
    yerinde satır güncellemesi yapılır; yeni bar açıldığında (nadir) concat'e
    düşer."""
    tick_open_time = tick_row["open_time"]
    if "open_time" in buf.columns and len(buf) and buf["open_time"].iat[-1] == tick_open_time:
        out = buf.copy()
        cols = list(tick_row.keys())
        out.iloc[-1, out.columns.get_indexer(cols)] = list(tick_row.values())
        return out
    base = buf[buf["open_time"] != tick_open_time] if "open_time" in buf.columns else buf
    return pd.concat([base, pd.DataFrame([tick_row])], ignore_index=True).tail(limit)


def _merge_closed_bar_and_index(
    existing: pd.DataFrame, new_row: dict, limit: int, state: Optional[IndicatorState],
    use_incremental: bool = False,
):
    """Kapanan bar'ı buffer'a ekler + indikatörleri hesaplar — executor'da çalıştırılır
    (bkz. _update_and_process_symbol_mtf).

    use_incremental=False (varsayılan, ÜRETİM): her zaman tam yeniden hesaplama
    (add_all_indicators) — Faz C öncesi davranışın AYNISI, değiştirilmedi.

    use_incremental=True (sadece gölge mod testi): state verilmişse incremental
    günceller (O(1) — tam yeniden hesaplamanın onlarca kat daha hızlısı), yoksa bu
    sembol+TF için bir kerelik bootstrap yapılır. Herhangi bir hata durumunda güvenli
    şekilde tam yeniden hesaplamaya döner — state=None döndürülür.

    Döner: (merged_df, state)
    """
    if "open_time" in existing.columns:
        existing = existing[existing["open_time"] != new_row["open_time"]]

    # 4h/1d gibi TF'lerin buffer limiti (7-12 bar) SuperTrend/ADX'in ihtiyaç duyduğu
    # minimumun (10-28 bar) altında kalabiliyor — bu durumda bootstrap HER ZAMAN
    # başarısız olur (zararsız ama gereksiz gürültü). Böyle küçük buffer'larda
    # doğrudan tam hesaplamaya git — zaten ucuz (az satır).
    if not use_incremental or len(existing) < _MIN_BOOTSTRAP_BARS:
        new_df = pd.DataFrame([new_row])
        merged = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
            subset=["open_time"], keep="last"
        ).tail(limit)
        return add_all_indicators(merged), None

    try:
        if state is None or state.steps_since_bootstrap >= RESYNC_INTERVAL:
            # İlk çağrı VEYA periyodik resync — state'in kendi içinde biriken
            # floating-point farkını ground-truth'tan (tam yeniden hesaplama) sıfırlar.
            state = bootstrap_state(existing)
        new_indicators = update_state(state, new_row)

        # ma200 / momentum: state gerektirmez, buffer'dan doğrudan lookup yeterli
        if len(existing) >= 200:
            tail_closes = existing["close"].tail(199).tolist() + [new_row["close"]]
            new_indicators["ma200"] = sum(tail_closes) / 200
        else:
            new_indicators["ma200"] = np.nan

        roc_period = Config.ROC_PERIOD
        if len(existing) >= roc_period:
            close_then = float(existing["close"].iloc[-roc_period])
            new_indicators["momentum"] = ((new_row["close"] - close_then) / close_then) * 100 if close_then else np.nan
        else:
            new_indicators["momentum"] = np.nan

        new_row_full = {**new_row, **new_indicators}
        merged = pd.concat(
            [existing, pd.DataFrame([new_row_full])], ignore_index=True
        ).drop_duplicates(subset=["open_time"], keep="last").tail(limit)
        return merged, state
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("İncremental indikatör hatası, tam yeniden hesaplamaya dönülüyor: %s", e)
        new_df = pd.DataFrame([new_row])
        merged = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
            subset=["open_time"], keep="last"
        ).tail(limit)
        return add_all_indicators(merged), None


def _build_derived_closed_bar(df_1m: pd.DataFrame, closing_tf: str) -> Optional[dict]:
    """1m-türetme projesi (10 Tem 2026, Adım 3 — hızlı yol 10 Tem akşam): 1m
    buffer'ından closing_tf'nin YENİ kapanan barını, WS-kaynaklı new_row ile
    BİREBİR AYNI şemada üretir.

    Önceki sürüm TimeframeAggregator.aggregate_ohlcv çağırıyordu — o fonksiyon
    TÜM buffer'ı (≈1000 satır) periyotlara bölüp HER periyodu ayrı ayrı
    filtreleyip yeniden hesaplıyor, oysa burada sadece SON kapanan periyot
    lazım. 543 sembolün aynı anda (5m/15m/.../1h sınırında) tetiklenmesiyle bu
    maliyet toplamda event loop'u 120s+ tıkayacak kadar büyüdü (10 Tem, batch-init
    tam durması vakası — bkz. proje notları). Artık _build_derived_forming_bar
    ile AYNI desen kullanılıyor: sadece kapanan periyodun satırlarını dilimleyip
    doğrudan max/min/sum. Gerçek üretim verisiyle (BTCUSDT/ETHUSDT/SOLUSDT/
    1000PEPEUSDT, 5m/15m/30m/1h) doğrulandı: tüm sayısal alanlar eskisiyle
    birebir aynı, 13x-196x daha hızlı.

    Döner: new_row dict'i (mevcut _merge_closed_bar_and_index'e AYNEN beslenebilir)
    veya yeterli/tam veri yoksa None."""
    minutes = TimeframeAggregator.TIMEFRAME_MINUTES.get(closing_tf)
    if not minutes or df_1m is None or df_1m.empty:
        return None
    last_open_time = int(df_1m["open_time"].iloc[-1])
    period_start = TimeframeAggregator.get_period_start(last_open_time, closing_tf)
    period_ms = minutes * 60_000
    period_end = period_start + period_ms
    period_bars = df_1m[(df_1m["open_time"] >= period_start) & (df_1m["open_time"] < period_end)]
    if len(period_bars) != minutes:
        # Boşluk var ya da periyot henüz tam değil — aggregate_ohlcv'nin
        # len(group_data) != ratio: atla davranışıyla aynı.
        return None

    def _sum_col(col: str) -> float:
        if col not in period_bars.columns:
            return 0.0
        return float(pd.to_numeric(period_bars[col], errors="coerce").fillna(0).sum())

    close_time = period_bars["close_time"].iloc[-1] if "close_time" in period_bars.columns else None
    close_time_ms = (
        int(close_time) if close_time is not None and pd.notna(close_time)
        else period_end - 1
    )

    return {
        "open_time": period_start,
        "open": float(period_bars["open"].iloc[0]),
        "high": float(period_bars["high"].max()),
        "low": float(period_bars["low"].min()),
        "close": float(period_bars["close"].iloc[-1]),
        "volume": float(period_bars["volume"].sum()),
        "close_time": close_time_ms,
        "quote_asset_volume": _sum_col("quote_asset_volume"),
        "number_of_trades": int(_sum_col("number_of_trades")),
        "taker_buy_base_asset_volume": _sum_col("taker_buy_base_asset_volume"),
        "taker_buy_quote_asset_volume": _sum_col("taker_buy_quote_asset_volume"),
        "buy_volume": _sum_col("buy_volume"),
        "sell_volume": _sum_col("sell_volume"),
    }


def _build_derived_forming_bar(df_1m: pd.DataFrame, tf: str) -> Optional[dict]:
    """1m-türetme projesi (10 Tem 2026, Adım 4): şu anki (henüz kapanmamış) 1m
    barı DAHİL, tf'nin oluşum halindeki (forming) barını türetir — panel/watchlist
    canlı gösterimi için (_handle_tick'in bugünkü davranışının eşdeğeri).

    _build_derived_closed_bar'dan (Adım 3) farkı: TimeframeAggregator.aggregate_ohlcv
    TAM periyot şartı arar (eksik grupları atlar) — forming bar TANIM GEREĞİ eksiktir,
    bu yüzden ayrı, "ne varsa topla" mantığı kullanılıyor. Kapanmış barlarla aynı
    OHLCV+hacim toplama kuralları (open=ilk, high=max, low=min, close=son, volume=toplam)
    uygulanıyor, sadece eksiksizlik şartı yok."""
    if df_1m is None or df_1m.empty:
        return None
    last_open_time = int(df_1m["open_time"].iloc[-1])
    period_start = TimeframeAggregator.get_period_start(last_open_time, tf)
    period_bars = df_1m[df_1m["open_time"] >= period_start]
    if period_bars.empty:
        return None

    period_ms = TimeframeAggregator.TIMEFRAME_MINUTES[tf] * 60_000

    def _sum_col(col: str) -> float:
        if col not in period_bars.columns:
            return 0.0
        return float(pd.to_numeric(period_bars[col], errors="coerce").fillna(0).sum())

    return {
        "open_time": period_start,
        "open": float(period_bars["open"].iloc[0]),
        "high": float(period_bars["high"].max()),
        "low": float(period_bars["low"].min()),
        "close": float(period_bars["close"].iloc[-1]),
        "volume": float(period_bars["volume"].sum()),
        "close_time": period_start + period_ms - 1,
        "quote_asset_volume": _sum_col("quote_asset_volume"),
        "number_of_trades": int(_sum_col("number_of_trades")),
        "taker_buy_base_asset_volume": _sum_col("taker_buy_base_asset_volume"),
        "taker_buy_quote_asset_volume": _sum_col("taker_buy_quote_asset_volume"),
        "buy_volume": _sum_col("buy_volume"),
        "sell_volume": _sum_col("sell_volume"),
    }


# --- Logging Kurulumu ---
# Merkezi utils.logger sistemi kullanılıyor — daha önce bu modülün kendi ayrı
# setup_logging()'i vardı (propagate=False ile izole, logs/live_data_manager.log'a
# yazan), bu yüzden bu modülün tetikleme logları (🎯, 🕯️) ana log dosyasında hiç
# görünmüyordu (7 Tem'de saatler süren bir teşhis yanlışına yol açtı).
logger = get_logger(__name__)


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
        # Asyncio-native WS taşıma katmanı yöneticisi (utils/asyncio_ws_client.py) —
        # thread-per-connection yerine aynı event loop'ta task modeli.
        self._asyncio_ws_manager: Optional[AsyncioBinanceStreamManager] = None
        self.is_ws_connected = False
        self.last_message_time: Optional[float] = (
            None  # Son WebSocket mesajının zamanını takip et
        )
        # Tekil bağlantı ölümünü yakalamak için: her bağlantının kendi son-mesaj zamanı.
        # Global last_message_time herhangi bir bağlantıdan mesaj gelince sıfırlandığı
        # için tek bir bağlantının sessizce ölmesini maskeliyordu (3 Tem vakası).
        self._socket_mgr_to_conn_id: Dict[int, int] = {}  # id(socket_manager) -> connection_id
        self._conn_last_message_time: Dict[int, float] = {}  # connection_id -> son mesaj zamanı
        self._conn_symbols: Dict[int, List[str]] = {}  # connection_id -> semboller (tanı için)
        self.reconnect_attempt = 0  # Üstel backoff için sayaç
        self.connection_reset_count = 0  # Connection reset sayacı
        self.last_error_type = None  # Son hata türü
        self.consecutive_errors = 0  # Ardışık hata sayısı
        self.db_lock = asyncio.Lock()  # Veritabanı yazma işlemleri için kilit
        self._startup_lookback_days: float = 1.0
        self._startup_fill_end_ms: int = 0
        self._gap_start_ms: Dict[str, int] = {}
        self._oi_cache: Dict[str, dict] = {}  # OI in-memory cache

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

        # İndikatör incremental hesaplama durumu (Faz D, 6 Tem): sembol -> TF ->
        # IndicatorState. _merge_closed_bar_and_index'te bootstrap edilip güncellenir;
        # ana event loop thread'inde okunup yazılır (executor thread'leri sadece
        # kendilerine verilen state nesnesini mutasyona uğratır — thread-safe).
        self._indicator_state: Dict[str, Dict[str, IndicatorState]] = {}
        
        self.processing_tasks: set[asyncio.Task] = set()
        self._last_prices: Dict[str, float] = {}
        self._ticker_prices: Dict[str, float] = {}
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
            _INTERVAL_MS_MAP = {
                "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
                "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
            }
            interval_ms = _INTERVAL_MS_MAP.get(self.interval, 60_000)
            desired_bars = 1500
            now_ms = int(time.time() * 1000)
            desired_start_ms = now_ms - (desired_bars * interval_ms)

            oldest_timestamp = await get_oldest_timestamp(symbol, interval=self.interval)
            if oldest_timestamp is None or oldest_timestamp > (desired_start_ms + interval_ms):
                start_time = desired_start_ms
                logger.info(
                    f"[{symbol}] Geçmiş yetersiz (oldest={oldest_timestamp}), {desired_bars} bar çekiliyor..."
                )
            else:
                last_timestamp = await get_last_timestamp(symbol, interval=self.interval)
                start_time = last_timestamp + 1 if last_timestamp else None
                if start_time:
                    logger.info(
                        f"[{symbol}] Son kayıt: {pd.to_datetime(start_time - 1, unit='ms')}. Eksik veriler çekiliyor..."
                    )
                else:
                    logger.info(
                        f"[{symbol}] Veritabanında kayıt bulunamadı. Son {desired_bars} mum çekiliyor..."
                    )
                    start_time = desired_start_ms

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
        """Buffer'daki tüm kline verilerini tek transaction'da yazar."""
        if not self.kline_buffer:
            return

        buffer_copy = self.kline_buffer.copy()
        self.kline_buffer.clear()
        self.last_flush_time = time.time()

        try:
            async with self.db_lock:
                await bulk_insert_price_data_multi(buffer_copy)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Batch flush hatası: %s", e, exc_info=True)
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

    def _handle_websocket_message(self, socket_mgr, msg: str):
        """WebSocket'ten gelen multi-timeframe mesajları işler."""
        self.last_message_time = self.loop.time()  # Her mesajda zamanı güncelle
        self.connection_health_ok = True  # Mesaj geldi, bağlantı sağlıklı
        conn_id = self._socket_mgr_to_conn_id.get(id(socket_mgr))
        if conn_id is not None:
            self._conn_last_message_time[conn_id] = self.loop.time()
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
                    interval = kline["i"]  # 1m-türetme cutover sonrası her zaman "1m"
                    is_closed = kline["x"]

                    logger.debug(f"[{symbol}] {interval} Bar closed (x): {is_closed}")

                    # PnL/watchlist fiyatı — kline buffer throttle'ından bağımsız,
                    # HER mesajda güncellenir (sadece dict yazımı, bedava). Önceden
                    # _handle_tick içindeydi ve 2s throttle'a bağımlıydı, PnL'i
                    # gereksiz yere yavaşlatıyordu.
                    self._last_prices[symbol] = float(kline["c"])

                    if is_closed:
                        logger.info(f"🕯️ [{symbol}] {interval} mum kapandı. Fiyat: {kline['c']}")
                        # WebSocket thread'inden ana event loop'a güvenli coroutine çağrısı.
                        # _update_and_process_symbol_mtf (1m barını buffer'a ekler) ile
                        # _derive_and_dispatch_closing_tfs (o buffer'ı okur) TEK coroutine'de
                        # SIRALI await edilir — ayrı run_coroutine_threadsafe çağrıları
                        # sıralama garantisi vermiyordu (ilki executor'a await ettiği an
                        # event loop'u bırakıyor, ikincisi HENÜZ EKLENMEMİŞ buffer'ı okuyup
                        # sessizce atlıyordu; 10 Tem, çoğu sembolde türetme hiç tetiklenmiyordu).
                        asyncio.run_coroutine_threadsafe(
                            self._process_closed_1m_and_derive(
                                symbol, interval, kline, int(kline["T"]) + 1
                            ), self.loop
                        )
                    else:
                        tick_key = f"{symbol}:{interval}"
                        now = time.time()
                        throttle = _TICK_THROTTLE_SECS.get(interval, 2)
                        if now - self._tick_last_sent.get(tick_key, 0) >= throttle:
                            self._tick_last_sent[tick_key] = now
                            # bkz. yukarıdaki is_closed dalı — aynı sıralama garantisi
                            # forming bar türetmesi için de gerekli.
                            asyncio.run_coroutine_threadsafe(
                                self._process_tick_and_derive(symbol, interval, kline), self.loop
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
            _tbv = float(kline_data["V"])
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
                "taker_buy_base_asset_volume": _tbv,
                "taker_buy_quote_asset_volume": float(kline_data["Q"]),
                "buy_volume": _tbv,
                "sell_volume": float(kline_data["v"]) - _tbv,
            }
            limit = self.mtf_buffer_limits.get(interval, 100)
            loop = asyncio.get_event_loop()
            merged = await loop.run_in_executor(
                _MTF_EXECUTOR, _merge_tick_row, buf, tick_row, limit
            )
            await RedisClient.set_mtf_klines(symbol, interval, merged)
            logger.debug("[%s] %s tick Redis'e yazıldı", symbol, interval)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("[%s] %s tick hatası: %s", symbol, interval, e)

    @staticmethod
    def _new_row_to_kline_dict(new_row: dict) -> dict:
        """1m-türetme projesi (10 Tem 2026): _build_derived_closed_bar/
        _build_derived_forming_bar'ın ürettiği new_row'u, _update_and_process_symbol_mtf
        / _handle_tick'in beklediği Binance-kline-şekilli dict'e (t/o/h/l/c/v/T/q/n/V/Q)
        çevirir — bu iki merkezi fonksiyona HİÇ dokunmadan besleyebilmek için."""
        return {
            "t": new_row["open_time"],
            "T": new_row["close_time"],
            "o": new_row["open"],
            "h": new_row["high"],
            "l": new_row["low"],
            "c": new_row["close"],
            "v": new_row["volume"],
            "q": new_row["quote_asset_volume"],
            "n": new_row["number_of_trades"],
            "V": new_row["taker_buy_base_asset_volume"],
            "Q": new_row["taker_buy_quote_asset_volume"],
        }

    async def _process_closed_1m_and_derive(
        self, symbol: str, interval: str, kline: Dict, next_open_time_ms: int
    ) -> None:
        """1m-türetme: _update_and_process_symbol_mtf (1m barını buffer'a ekler)
        ile _derive_and_dispatch_closing_tfs (o buffer'ı okuyup üst TF türetir)
        SIRALI await edilir — bkz. _handle_websocket_message'daki açıklama."""
        await self._update_and_process_symbol_mtf(symbol, interval, kline)
        await self._derive_and_dispatch_closing_tfs(symbol, next_open_time_ms)

    async def _process_tick_and_derive(self, symbol: str, interval: str, kline: Dict) -> None:
        """1m-türetme: _handle_tick ile _derive_and_dispatch_forming_tfs için aynı
        sıralama garantisi (bkz. _process_closed_1m_and_derive)."""
        await self._handle_tick(symbol, interval, kline)
        await self._derive_and_dispatch_forming_tfs(symbol)

    async def _derive_and_dispatch_closing_tfs(self, symbol: str, next_open_time_ms: int) -> None:
        """1m-türetme projesi (Adım 5-6, 10 Tem 2026): bir 1m barı kapandığında,
        hangi üst TF'lerin de kapandığını (TimeframeAggregator.get_closing_timeframes)
        tespit eder, her biri için 1m buffer'ından kapanan barı türetir
        (_build_derived_closed_bar) ve mevcut _update_and_process_symbol_mtf'e AYNEN
        besler — indikatör hesaplama/sinyal üretimi/do_kirilimi-do_open_streak
        tetikleme kodlarına HİÇ dokunulmadı, sadece verinin kaynağı değişti."""
        derive_tfs = [tf for tf in self.supported_timeframes if tf != "1m"]
        if not derive_tfs:
            return
        closing_tfs = TimeframeAggregator.get_closing_timeframes(next_open_time_ms, derive_tfs)
        if not closing_tfs:
            return
        df_1m = self.mtf_buffers.get(symbol, {}).get("1m")
        if df_1m is None or df_1m.empty:
            return
        # Restart sonrası bazı sembollerin 1m geçmişi henüz kademeli yüklenme
        # sürecinde (batch init) kısa kalabiliyor — TimeframeAggregator'ı (ve onun
        # gürültülü "boundary alignment yok" uyarısını) hiç çağırmadan, şansı
        # olmayan TF'leri baştan ele. Gerçek bir hata değil, geçici ısınma durumu.
        n_bars = len(df_1m)
        closing_tfs = [
            tf for tf in closing_tfs
            if n_bars >= TimeframeAggregator.TIMEFRAME_MINUTES.get(tf, 0)
        ]
        if not closing_tfs:
            return
        loop = asyncio.get_event_loop()
        for tf in closing_tfs:
            new_row = await loop.run_in_executor(_MTF_EXECUTOR, _build_derived_closed_bar, df_1m, tf)
            if new_row is None:
                continue
            logger.info(f"🕯️ [{symbol}] {tf} mum kapandı (1m'den türetildi). Fiyat: {new_row['close']}")
            await self._update_and_process_symbol_mtf(symbol, tf, self._new_row_to_kline_dict(new_row))

    async def _derive_and_dispatch_forming_tfs(self, symbol: str) -> None:
        """1m-türetme projesi: her 1m tick'inde (throttle zaten _handle_websocket_message'ta
        uygulanıyor), üst TF'lerin oluşum halindeki (forming) barını 1m buffer'ından
        türetip _handle_tick'e besler — panel/watchlist canlı gösterimi için (bugünkü
        ayrı-WS-tick davranışının eşdeğeri, bkz. Adım 4 doğrulaması: kapanmış barlarda
        tam eşleşme, forming barlarda sadece ~2-3sn'lik doğal senkron farkı).

        10 Tem 2026 akşam: her TF için AYRI throttle (_TICK_THROTTLE_SECS — 1h=30s,
        4h/6h/8h=60s, 12h/1d=120s) uygulanıyor — önceki sürüm bu sözlüğü YOKSAYIP tüm
        9 TF'yi HER 1m tick'inde (2sn'de bir, 543 sembol için) yeniden hesaplıyordu.
        py-spy ile canlıda doğrulandı: _MTF_EXECUTOR'ın 4 worker'ı da sürekli
        _build_derived_forming_bar ile meşguldü (özellikle 12h/1d gibi büyük TF'lerin
        720-1440 satırlık dilimleri, 6 ayrı fillna+sum çağrısıyla), bu da MTF Batch
        Initialization gibi AYNI executor'ı paylaşan işleri kuyrukta süresiz bekletip
        120s timeout'a düşürüyordu. 1d artık 120sn'de bir hesaplanıyor (2sn yerine) —
        60x daha az çağrı, executor üzerindeki büyük TF yükü ortadan kalkıyor."""
        derive_tfs = [tf for tf in self.supported_timeframes if tf != "1m"]
        if not derive_tfs:
            return
        df_1m = self.mtf_buffers.get(symbol, {}).get("1m")
        if df_1m is None or df_1m.empty:
            return
        now = time.time()
        due_tfs = []
        for tf in derive_tfs:
            throttle = _TICK_THROTTLE_SECS.get(tf, 2)
            tick_key = f"{symbol}:{tf}"
            if now - self._tick_last_sent.get(tick_key, 0) >= throttle:
                self._tick_last_sent[tick_key] = now
                due_tfs.append(tf)
        if not due_tfs:
            return
        # bkz. _derive_and_dispatch_closing_tfs — aynı ısınma-döneminde-atla mantığı.
        # forming bar için tam periyot şartı yok ama en az 1 bar olması yeterli,
        # bu yüzden burada eşik 0 (her zaman geçer) — asıl amaç closing tarafındaki
        # log gürültüsünü önlemekti, forming zaten _build_derived_forming_bar
        # içinde "period_bars boşsa None dön" ile sessizce ele alınıyor.
        loop = asyncio.get_event_loop()
        for tf in due_tfs:
            new_row = await loop.run_in_executor(_MTF_EXECUTOR, _build_derived_forming_bar, df_1m, tf)
            if new_row is None:
                continue
            await self._handle_tick(symbol, tf, self._new_row_to_kline_dict(new_row))

    async def _update_and_process_symbol_mtf(self, symbol: str, interval: str, kline_data: Dict):
        """
        Multi-timeframe version: Updates the DataFrame for specific timeframe and triggers signal processing.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '15m')
            kline_data: Kline data from WebSocket
        """
        record_activity("live_data_manager")
        try:
            # Parse kline data
            _tbv = float(kline_data["V"])
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
                "taker_buy_base_asset_volume": _tbv,
                "taker_buy_quote_asset_volume": float(kline_data["Q"]),
                "buy_volume": _tbv,
                "sell_volume": float(kline_data["v"]) - _tbv,
            }

            # MTF buffer'a ekle (her timeframe için ayrı buffer) + indikatörler
            if self.mtf_enabled and symbol in self.mtf_buffers:
                limit = self.mtf_buffer_limits.get(interval, 100)
                loop = asyncio.get_event_loop()
                state = self._indicator_state.get(symbol, {}).get(interval)

                # ÜRETİM (Faz D, 6 Tem): incremental hesaplama artık gerçek yol —
                # gölge modda (BTCUSDT, saatlerce) doğrulandı, sıfır fark bulundu.
                merged, new_state = await loop.run_in_executor(
                    _MTF_EXECUTOR,
                    _merge_closed_bar_and_index,
                    self.mtf_buffers[symbol][interval],
                    new_row,
                    limit,
                    state,
                    True,  # use_incremental
                )
                self.mtf_buffers[symbol][interval] = merged
                if new_state is not None:
                    self._indicator_state.setdefault(symbol, {})[interval] = new_state
                else:
                    # Hata oldu (tam yeniden hesaplamaya düşüldü) — bir sonraki
                    # çağrıda state yeniden bootstrap edilsin.
                    self._indicator_state.setdefault(symbol, {}).pop(interval, None)

                # Cache to Redis
                await RedisClient.set_mtf_klines(symbol, interval, self.mtf_buffers[symbol][interval])
                logger.debug(f"[{symbol}] {interval} buffer updated and cached")
                # do_kirilimi/do_open_streak tetikleme artık signal_service.py'de
                # (bkz. paper trading ayrıştırması, 10 Tem 2026 cutover).

            # Legacy 1m buffer (kline_data) — sadece DB batch insert için tutuluyor
            if interval == '1m':
                new_df = pd.DataFrame([new_row])
                self.kline_data[symbol] = pd.concat(
                    [self.kline_data[symbol], new_df], ignore_index=True
                ).tail(1000)

            # Batch insert için buffer'a ekle (sadece 1m için - diğer TF'ler opsiyonel)
            if interval == '1m':
                await self._add_to_batch_buffer(symbol, new_row)


            # Sinyal üretimi (her timeframe için)
            if self.mtf_enabled:
                # Get reference data for this timeframe
                ref_df = pd.DataFrame()
                if self.ref_symbol in self.mtf_buffers and interval in self.mtf_buffers[self.ref_symbol]:
                    ref_df = await loop.run_in_executor(
                        _MTF_EXECUTOR, pd.DataFrame.copy, self.mtf_buffers[self.ref_symbol][interval]
                    )

                # Minimum bar requirements per timeframe
                min_bars = {'1m': 200, '5m': 100, '15m': 67, '1h': 24, '4h': 12, '1d': 7}.get(interval, 100)

                if not ref_df.empty and len(self.mtf_buffers[symbol][interval]) >= min_bars:
                    # Cutover sonrası (SIGNAL_SOURCE=yeni) signal_service.py gerçek
                    # yazan taraf — bu gölge hesaplama (dry_run=True) artık saf israf
                    # değil, signal_engine.SignalFilter.check() dry_run'dan habersiz
                    # koşulsuz signal_filter_events'e INSERT yapıyor: iki process aynı
                    # olayı çift yazıyordu. Cutover aktifken bu blok tamamen atlanır,
                    # publish_kline_closed_event (signal_service'i besleyen asıl satır)
                    # dokunulmadan kalır. SIGNAL_SOURCE='eski'ye dönülürse otomatik
                    # eski davranışa döner.
                    if Config.SIGNAL_SOURCE != "yeni":
                        oi_info = self._oi_cache.get(symbol)
                        oi_data_json = json.dumps(oi_info) if oi_info else None
                        df_copy = await loop.run_in_executor(
                            _MTF_EXECUTOR, pd.DataFrame.copy, self.mtf_buffers[symbol][interval]
                        )
                        task = asyncio.create_task(
                            process_and_enrich_signals(
                                symbol=symbol,
                                df=df_copy,
                                ref_df=ref_df,
                                interval=interval,
                                oi_data=oi_data_json,
                                symbol_buffers=self.mtf_buffers.get(symbol, {}),
                                dry_run=False,
                            )
                        )
                        self.processing_tasks.add(task)
                        task.add_done_callback(self.processing_tasks.discard)
                        logger.info(f"🎯 [{symbol}] {interval} sinyal üretimi başlatıldı")
                    await RedisClient.publish_kline_closed_event(symbol, interval, new_row["open_time"])

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

    # =============================================================================
    # MULTI-TIMEFRAME FUNCTIONS (NEW!)
    # =============================================================================

    async def _initialize_mtf_dataframes(self, reload_symbols: set[str] | None = None):
        """
        Hibrit batch initialization: Tarihsel tüm TF'leri batch halinde yükle + sonra WebSocket.

        reload_symbols: None → tüm sembolleri yükle (ilk açılış).
                        set  → sadece bu sembolleri DB'den yükle; diğerlerini Redis'ten hızlı yükle.
        """
        if not self.mtf_enabled:
            return

        # ── Hızlı yükleme: TÜM semboller önce Redis'ten alınır ──────────────────────────────────
        # Hem reload_symbols=None (ilk açılış) hem de reload_symbols=set durumunda çalışır.
        # Redis'te yeterli veri (≥ limit/2) olan semboller chart'ta anında görünür.
        min_bars_ratio = 0.5
        redis_hit: set[str] = set()
        logger.info("[MTF] %d sembol Redis cache hızlı yükleniyor...", len(self.symbols))

        async def _load_from_redis(sym: str) -> tuple[str, bool]:
            all_tf_ok = True
            for tf in self.supported_timeframes:
                limit = self.mtf_buffer_limits.get(tf, 250)
                df = await RedisClient.get_mtf_klines(sym, tf, limit=limit)
                if df is not None and len(df) >= limit * min_bars_ratio:
                    self.mtf_buffers[sym][tf] = df.tail(limit)
                else:
                    all_tf_ok = False
            return sym, all_tf_ok

        redis_results = await asyncio.gather(*[_load_from_redis(s) for s in self.symbols])
        for sym, ok in redis_results:
            if ok:
                redis_hit.add(sym)
        logger.info("[MTF] Redis hızlı yükleme: %d/%d sembol tam yüklendi.", len(redis_hit), len(self.symbols))

        # ── Batch yükleme: Redis'te eksik/yetersiz olanlar + zorla yenilenmesi gerekenler ────────
        force_reload = reload_symbols if reload_symbols is not None else set()
        symbols_to_reload = [s for s in self.symbols if s not in redis_hit or s in force_reload]

        if not symbols_to_reload:
            logger.info("🎉 MTF Batch Initialization tamamlandı! WebSocket canlı mod başlatılabilir.")
            return

        # 10 Tem 2026 akşam: batch_size 10→30. Eskiden burada REST çağrı sayısına göre
        # manuel bir bekleme hesaplanıyordu (WEIGHT_PER_CALL/WEIGHT_BUDGET_PER_MIN) —
        # artık BinanceClientManager.fetch_klines'ın kendisi RedisClient.throttle_external_api
        # ile TÜM process'leri (run_services.py + desktop panel) kapsayan merkezi bir
        # sliding-window limiter'dan geçiyor. Burada AYRICA beklemek çifte throttle
        # olurdu — gerçek hız sınırı zaten REST çağrısının içinde uygulanıyor, batch
        # boyutu sadece eşzamanlı sembol sayısını (executor/ağ paralelliği) belirliyor.
        batch_size = 30

        total_symbols = len(symbols_to_reload)
        total_batches = (total_symbols + batch_size - 1) // batch_size

        logger.info(f"🚀 MTF Batch Initialization başlatılıyor:")
        logger.info(f"   📊 Toplam: {total_symbols} sembol × {len(self.supported_timeframes)} TF")
        logger.info(f"   📦 Batch: {batch_size} sembol/batch, {total_batches} batch")

        # Sembolleri batch'lere böl
        for i in range(0, total_symbols, batch_size):
            batch = symbols_to_reload[i:i + batch_size]
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
            success_count = sum(1 for r in results if r is not None)
            total_rest_calls = sum(r[1] for r in results if r is not None)
            logger.info(f"✅ Batch {batch_num}/{total_batches} tamamlandı ({success_count}/{len(batch)} başarılı, {total_rest_calls} REST çağrısı)")

        logger.info("🎉 MTF Batch Initialization tamamlandı! WebSocket canlı mod başlatılabilir.")

    async def _load_symbol_all_timeframes(self, symbol: str) -> Tuple[bool, int]:
        """
        Bir sembol için tüm timeframe'leri API'den yükle.

        Args:
            symbol: Sembol adı

        Returns:
            (bool, int): (REST kullanılmadıysa True, yapılan REST çağrı sayısı) —
            REST çağrı sayısı, batch gecikmesinin gerçek ağırlığa göre hesaplanması için.
        """
        rest_call_count = 0
        try:
            # Binance'ten çekilecek TF'ler (1m ve aggregate edilebilecekler hariç)
            binance_timeframe_limits = {
                '1h': 250,   # ~10 gün  — aggregate için 1m yetersiz (15k bar lazım)
                '4h': 250,   # ~41 gün
                '1d': 250,   # ~250 gün
            }

            loaded_count = 0
            binance_call_made = False
            loop = asyncio.get_event_loop()

            # ── 1m: DB'den yükle ──────────────────────────────────────────────
            limit_1m = max(1500, int(self._startup_lookback_days * 24 * 60))
            df_1m = await get_recent_klines(symbol, "1m", limit_1m)
            if df_1m.empty:
                logger.warning(f"[{symbol}] 1m DB'de yok, Binance'ten çekiliyor...")
                rest_call_count += 1
                df_1m = await BinanceClientManager.fetch_klines(symbol=symbol, interval="1m", limit=1500)

            if not df_1m.empty:
                df_1m_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1m)
                self.mtf_buffers[symbol]['1m'] = df_1m_ind.tail(self.mtf_buffer_limits.get('1m', 250))
                await RedisClient.set_mtf_klines(symbol, '1m', self.mtf_buffers[symbol]['1m'])
                loaded_count += 1

            # ── 5m/15m/30m/6h/8h/12h: Redis-first (REST sadece ilk kurulumda) ──
            for ws_tf in ['5m', '15m', '30m', '6h', '8h', '12h']:
                limit = self.mtf_buffer_limits.get(ws_tf, 250)
                cached = await RedisClient.get_mtf_klines(symbol, ws_tf, limit=limit)
                check_kline_schema(cached, f"RedisCache.{ws_tf}")
                if cached is not None and len(cached) >= limit // 2:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, cached)
                    self.mtf_buffers[symbol][ws_tf] = df_ind.tail(limit)
                    loaded_count += 1
                else:
                    binance_call_made = True
                    rest_call_count += 1
                    df_ws = await BinanceClientManager.fetch_klines(symbol=symbol, interval=ws_tf, limit=limit)
                    if not df_ws.empty:
                        df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_ws)
                        self.mtf_buffers[symbol][ws_tf] = df_ind.tail(limit)
                        await RedisClient.set_mtf_klines(symbol, ws_tf, self.mtf_buffers[symbol][ws_tf])
                        loaded_count += 1

            # ── 1h / 4h: CA view'larından (boşluksuz, 1m'den otomatik türetilmiş) ──
            for tf in ['1h', '4h']:
                limit = self.mtf_buffer_limits.get(tf, 250)
                ca_df = await get_cagg_klines(symbol, tf, limit)
                if not ca_df.empty:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, ca_df)
                    self.mtf_buffers[symbol][tf] = df_ind.drop_duplicates(subset=["open_time"], keep="last")
                    await RedisClient.set_mtf_klines(symbol, tf, self.mtf_buffers[symbol][tf])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] {tf}: CA'dan yüklendi ({len(ca_df)} bar)")
                else:
                    logger.warning(f"[{symbol}] {tf}: CA boş")

            # ── 1d: Redis cache → yoksa Binance (CA için çok fazla 1m gerekir) ──
            limit_1d = binance_timeframe_limits.get('1d', 250)
            cached_df = await RedisClient.get_mtf_klines(symbol, '1d', limit=limit_1d)
            if cached_df is not None and len(cached_df) >= limit_1d // 2:
                self.mtf_buffers[symbol]['1d'] = cached_df.drop_duplicates(subset=["open_time"], keep="last")
                loaded_count += 1
                logger.debug(f"[{symbol}] 1d: Redis cache'den yüklendi ({len(cached_df)} bar)")
            else:
                binance_call_made = True
                rest_call_count += 1
                df_1d = await BinanceClientManager.fetch_klines(symbol=symbol, interval='1d', limit=limit_1d)
                if not df_1d.empty:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1d)
                    self.mtf_buffers[symbol]['1d'] = df_ind.tail(limit_1d).drop_duplicates(subset=["open_time"], keep="last")
                    await RedisClient.set_mtf_klines(symbol, '1d', self.mtf_buffers[symbol]['1d'])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] 1d: Binance'ten çekildi ({len(df_1d)} bar)")
                else:
                    logger.warning(f"[{symbol}] 1d: Veri boş")

            src = "REST" if binance_call_made else "Redis"
            logger.info(f"✅ [{symbol}] {loaded_count} TF yüklendi (1m=DB, 5m-12h={src}, 1h/4h=CA, 1d=cache)")
            return not binance_call_made, rest_call_count

        except Exception as e:
            logger.error(f"❌ [{symbol}] Yükleme hatası: {e}", exc_info=False)
            return False, rest_call_count
    
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
                    logger.info(
                        f"💚 Keep-Alive Health Check: Bağlantı sağlıklı "
                        f"(son mesaj: {time_since_last_msg:.1f}s önce)"
                    )
                    self.last_ping_time = current_time
                    await beat("ws_ingestion")

                    # Tekil bağlantı ölümü kontrolü: global last_message_time herhangi
                    # bir bağlantıdan mesaj gelince tazelendiği için tek bir bağlantının
                    # sessizce ölmesini maskeleyebilir (3 Tem vakası) — her bağlantıyı
                    # ayrı ayrı kontrol et.
                    stale_conns = [
                        conn_id for conn_id, ts in self._conn_last_message_time.items()
                        if current_time - ts > 60
                    ]
                    if stale_conns:
                        for conn_id in stale_conns:
                            age = current_time - self._conn_last_message_time[conn_id]
                            symbols = self._conn_symbols.get(conn_id, [])
                            symbols_preview = ", ".join(symbols[:5]) + (
                                f" (+{len(symbols) - 5} diğer)" if len(symbols) > 5 else ""
                            )
                            logger.error(
                                f"⚠️ Connection #{conn_id + 1} {age:.0f}s'dir mesaj almıyor "
                                f"(semboller: {symbols_preview})"
                            )
                        await send_telegram_message(
                            f"⚠️ {len(stale_conns)} WS bağlantısı bayat (60s+): "
                            f"#{', #'.join(str(c + 1) for c in stale_conns)} — tam reconnect tetikleniyor"
                        )
                        self.connection_health_ok = False
                        self.is_ws_connected = False
                        continue
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

        # 1m-türetme (10 Tem 2026 cutover): sadece kline_1m'e abone olunur, diğer
        # TF'ler 1m buffer'ından türetilir (bkz. _derive_and_dispatch_closing_tfs).
        stream_tfs = ['1m']

        logger.info(f"🚀 Multi-Timeframe WebSocket başlatılıyor: {len(self.symbols)} sembol × {len(stream_tfs)} TF")

        all_streams = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols]

        total_streams = len(all_streams)
        # Allow override from central config (new tunable)
        self.max_streams_per_connection = getattr(
            Config, 'WS_MAX_STREAMS_PER_CONNECTION', self.max_streams_per_connection
        )
        connections_needed = (
            (total_streams + self.max_streams_per_connection - 1)
            // self.max_streams_per_connection
        )

        logger.info(f"📊 Toplam stream: {total_streams} ({len(self.symbols)} sembol × {len(stream_tfs)} TF)")
        logger.info(f"🔌 Gerekli connection: {connections_needed} (max {self.max_streams_per_connection} stream/connection)")

        try:
            # Eski bağlantıları güvenli şekilde kapat
            await self._safe_close_websocket()

            # Tekil bağlantı takibini sıfırla — yeniden bağlanmada eski eşlemeler kalmasın
            self._socket_mgr_to_conn_id.clear()
            self._conn_last_message_time.clear()
            self._conn_symbols.clear()

            # Asyncio-native taşıma katmanı (utils/asyncio_ws_client.py) — her bağlantı
            # kendi OS thread'i yerine aynı event loop'ta bir task (10 Tem 2026 cutover,
            # thread-tabanlı binance-connector kalıcı olarak kaldırıldı — gölge testlerle
            # doğrulanmıştı, bkz. modül docstring'i).
            base_url = getattr(Config, 'BINANCE_WS_BASE', 'wss://fstream.binance.com/market')
            self._asyncio_ws_manager = AsyncioBinanceStreamManager(
                base_url=base_url,
                on_message=self._handle_websocket_message,
                max_streams_per_connection=self.max_streams_per_connection,
            )
            connections = await self._asyncio_ws_manager.start(all_streams)
            for connection_id, conn in connections.items():
                self.ws_clients[connection_id] = conn
                self._socket_mgr_to_conn_id[id(conn)] = connection_id
                self._conn_last_message_time[connection_id] = self.loop.time()
                self._conn_symbols[connection_id] = sorted({s.split("@")[0].upper() for s in conn.streams})
                logger.info(f"✅ Connection #{connection_id} başarıyla kuruldu ({len(conn.streams)} stream, asyncio)")

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
                              AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - INTERVAL '1 hour'
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

    async def _startup_gap_fill(self) -> set[str]:
        """Startup gap fill: 1m için dinamik lookback ile gap doldurur.
        Gap doldurulan sembollerin setini döndürür (MTF init optimizasyonu için)."""
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

        now_ms = int(time.time() * 1000)
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
                              AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - (:days * INTERVAL '1 day')
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

            # 10 Tem 2026 akşam: sembol başına sıralı (her fetch öncesi 0.5s sleep)
            # çalışıyordu — 538 sembolde bu tek başına ~4.5dk + hata/retry'lerle
            # ~9dk'ya çıkıyordu, arkasındaki replay+MTF init'i geciktiriyordu
            # (bkz. _replay_filter_state_for_gaps'teki aynı 10 Tem notu). Artık
            # _replay_filter_state_for_gaps ile AYNI ağırlık-bütçeli parti deseni:
            # sembolün KENDİ gap'leri hâlâ sırayla dolduruluyor (fetch_start ilerlemesi
            # doğası gereği sıralı), ama FARKLI semboller partiler halinde paralel.
            _WEIGHT_PER_CALL_GAP = 2
            _WEIGHT_BUDGET_PER_MIN = 1200
            _GAP_BATCH_SIZE = 40

            async def _fill_one_symbol(sym: str, gaps: list[tuple[int, int]]) -> tuple[int, int]:
                filled = 0
                rest_calls = 0
                for gap_start_ms, gap_end_ms in gaps:
                    fetch_start = gap_start_ms + _INTERVAL_MS
                    while fetch_start < gap_end_ms:
                        df = None
                        for attempt in range(3):
                            try:
                                rest_calls += 1
                                df = await BinanceClientManager.fetch_klines(
                                    symbol=sym, interval="1m", limit=1000, startTime=fetch_start,
                                )
                                break
                            except Exception as exc:
                                logger.warning("[Startup] %s API hatası (deneme %d/3): %s", sym, attempt + 1, exc)
                                if attempt < 2:
                                    await asyncio.sleep(30)
                        if df is None or df.empty:
                            break
                        df = df[df["open_time"] < gap_end_ms]
                        if df.empty:
                            break
                        async with self.db_lock:
                            await bulk_insert_price_data(sym, df, interval="1m")
                        filled += len(df)
                        last_ts = int(df["open_time"].iloc[-1])
                        if last_ts <= fetch_start or len(df) < 1000:
                            break
                        fetch_start = last_ts + _INTERVAL_MS
                return filled, rest_calls

            total_filled = 0
            items = list(all_gaps.items())
            for i in range(0, len(items), _GAP_BATCH_SIZE):
                batch = items[i:i + _GAP_BATCH_SIZE]
                results = await asyncio.gather(
                    *[_fill_one_symbol(sym, gaps) for sym, gaps in batch],
                    return_exceptions=True,
                )
                total_filled += sum(r[0] for r in results if isinstance(r, tuple))
                total_rest_calls = sum(r[1] for r in results if isinstance(r, tuple))
                if i + _GAP_BATCH_SIZE < len(items):
                    wait = max(0.5, (total_rest_calls * _WEIGHT_PER_CALL_GAP / _WEIGHT_BUDGET_PER_MIN) * 60)
                    await asyncio.sleep(wait)

            logger.info("[Startup] 1m gap fill tamamlandı: %d bar eklendi", total_filled)
        else:
            logger.info("[Startup] 1m: gap yok.")

        self._startup_fill_end_ms = int(time.time() * 1000)
        self._startup_lookback_days = float(lookback_days)
        # SignalFilter replay'i için her sembolün en erken gap başlangıcı saklanır
        # (bkz. _background_startup — sinyal filtresi referans noktalarını bu
        # zamandan itibaren geçmiş fiyat hareketiyle senkronize eder).
        self._gap_start_ms = {sym: min(gs for gs, _ in gaps) for sym, gaps in all_gaps.items()}
        return set(all_gaps.keys())

    async def _replay_filter_state_for_gaps(self, gap_starts: Dict[str, int], source: str = "") -> None:
        """Gap'i olan semboller için SignalFilter referans noktalarını (bkz.
        SignalEngine.replay_filter_state) downtime süresince gerçekleşen fiyat
        hareketiyle senkronize eder. Binance'ten DOĞRUDAN gap aralığını çeker —
        canlı MTF buffer'ının boyut sınırına (mtf_buffer_limits, 5m/15m için
        ~16-17 saat) bağımlı değildir, bu yüzden çok günlük gap'lerde de çalışır.

        gap_starts: sembol -> bu sembolün en erken gap başlangıcı (ms epoch).
        Hem başlangıç gap doldurmasından (_background_startup) hem runtime gap
        iyileştirmesinden (_continuous_gap_heal_loop) ortak çağrılır.

        10 Tem 2026 akşam: (sembol, TF) çiftleri artık ağırlık-bütçeli partiler
        halinde PARALEL işleniyor — önceki sürüm sırayla (538 sembol × 2 TF ≈
        1076 REST çağrısı, tek tek await) çalışıyordu, bu da 15-25 dakika
        sürüyordu ve arkasındaki _initialize_mtf_dataframes'i (Redis'ten gerçek
        5m-12h geçmişini yükleyen adım) o kadar geciktiriyordu (alpha/beta gibi
        referans-sembole bağlı metrikler o pencerede None kalıyordu). Partileme
        deseni _initialize_mtf_dataframes'in batch mantığıyla AYNI (WEIGHT_PER_CALL/
        WEIGHT_BUDGET_PER_MIN, 4 Tem ban dersi) — SignalFilter state'i DB'de
        tutuluyor (bkz. signals/signal_filter.py), farklı (sembol, TF) çiftleri
        için paralel çağrı güvenli.
        """
        from signals.signal_engine import signal_engine as _se
        from signals.signal_processor import _SIGNAL_GENERATION_TFS

        _BAR_MS = {"5m": 300_000, "15m": 900_000}
        _MAX_API_LIMIT = 1500
        _CONTEXT_BARS = 210  # MA200 gibi indikatörlerin warm-up'ı için gap öncesi ek bağlam
        _WEIGHT_PER_CALL = 2
        _WEIGHT_BUDGET_PER_MIN = 1200
        _BATCH_SIZE = 40  # (sembol, TF) çifti / parti
        now_ms = int(time.time() * 1000)
        loop = asyncio.get_event_loop()

        pairs = [
            (sym, tf, gap_start)
            for sym, gap_start in gap_starts.items()
            if gap_start is not None
            for tf in _SIGNAL_GENERATION_TFS
            if tf in _BAR_MS
        ]

        async def _replay_one(sym: str, tf: str, gap_start: int) -> int:
            bar_ms = _BAR_MS[tf]
            total_bars = int((now_ms - gap_start) / bar_ms) + _CONTEXT_BARS
            try:
                if total_bars <= _MAX_API_LIMIT:
                    df = await BinanceClientManager.fetch_klines(
                        symbol=sym, interval=tf, limit=total_bars,
                        startTime=gap_start - _CONTEXT_BARS * bar_ms,
                    )
                else:
                    # Gap API limitinden uzun — en güncel (şu ana en yakın) kısmı önceliklendir.
                    df = await BinanceClientManager.fetch_klines(
                        symbol=sym, interval=tf, limit=_MAX_API_LIMIT,
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("[Replay] %s %s veri çekilemedi: %s", sym, tf, exc)
                return 0
            if df.empty:
                return 0
            df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df)
            return await _se.replay_filter_state(df_ind, sym, tf, gap_start)

        replay_bars = 0
        total_pairs = len(pairs)
        for i in range(0, total_pairs, _BATCH_SIZE):
            batch = pairs[i:i + _BATCH_SIZE]
            results = await asyncio.gather(
                *[_replay_one(sym, tf, gap_start) for sym, tf, gap_start in batch],
                return_exceptions=True,
            )
            replay_bars += sum(r for r in results if isinstance(r, int))
            if i + _BATCH_SIZE < total_pairs:
                wait = max(0.5, (len(batch) * _WEIGHT_PER_CALL / _WEIGHT_BUDGET_PER_MIN) * 60)
                await asyncio.sleep(wait)

        logger.info(
            "[%s] SignalFilter replay: %d sembol, %d bar "
            "(gap sırasında kaçırılan crossover'lar referans noktalarına yansıtıldı)",
            source or "Replay", len(gap_starts), replay_bars,
        )

    async def _continuous_gap_heal_loop(self) -> None:
        """Her 5 dakikada bir son 1 saatin 1m gap'lerini ve kuyruk gap'lerini tarar ve doldurur."""
        import time as _t

        _INTERVAL_MS = 60_000
        _THRESHOLD_MS = _INTERVAL_MS * 2
        _TAIL_THRESHOLD_MS = _INTERVAL_MS * 3  # son bar 3dk'dan eskiyse tail gap

        while True:
            await asyncio.sleep(300)
            try:
                now_ms = int(_t.time() * 1000)
                symbols_list = list(self.symbols)

                # İç gap tespiti (LAG)
                async def _fetch_lag_gaps():
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
                                      AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - INTERVAL '1 hour'
                                ) t
                                WHERE prev_ts IS NOT NULL
                                  AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > :thresh
                                ORDER BY symbol, prev_ts
                            """),
                            {"syms": symbols_list, "thresh": _THRESHOLD_MS},
                        )
                        return result.fetchall()

                rows = await run_with_db_timeout(_fetch_lag_gaps())

                gaps: dict[str, list[tuple[int, int]]] = {}
                for sym, prev_dt, curr_dt in rows:
                    g_start = int(prev_dt.timestamp() * 1000)
                    g_end = int(curr_dt.timestamp() * 1000)
                    gaps.setdefault(sym, []).append((g_start, g_end))

                # Kuyruk gap tespiti: son bar'dan şu ana kadar boşluk var mı?
                async def _fetch_tail_gaps():
                    async with get_session() as session:
                        tail_result = await session.execute(
                            text("""
                                SELECT symbol, MAX(timestamp)
                                FROM price_data
                                WHERE symbol = ANY(:syms) AND interval = '1m'
                                GROUP BY symbol
                            """),
                            {"syms": symbols_list},
                        )
                        return tail_result.fetchall()

                tail_rows = await run_with_db_timeout(_fetch_tail_gaps())

                for sym, last_dt in tail_rows:
                    if last_dt is None:
                        continue
                    tail_ms = int(last_dt.timestamp() * 1000)
                    if (now_ms - tail_ms) > _TAIL_THRESHOLD_MS:
                        existing = gaps.get(sym, [])
                        if not any(gs >= tail_ms for gs, _ in existing):
                            gaps.setdefault(sym, []).append((tail_ms, now_ms - _INTERVAL_MS))

                if not gaps:
                    logger.debug("[GapHeal] Gap yok.")
                    continue

                logger.warning("[GapHeal] %d sembolde gap bulundu, dolduruluyor...", len(gaps))
                total_filled = 0

                for sym, sym_gaps in gaps.items():
                    for gap_start_ms, gap_end_ms in sym_gaps:
                        fetch_start = gap_start_ms + _INTERVAL_MS
                        while fetch_start < gap_end_ms:
                            await asyncio.sleep(0.3)
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

                if total_filled:
                    logger.info("[GapHeal] %d bar dolduruldu.", total_filled)
                    if self.mtf_enabled:
                        for sym in gaps:
                            await self._refresh_mtf_redis(sym)
                        logger.info("[GapHeal] %d sembol MTF Redis yenilendi.", len(gaps))
                    gap_starts = {sym: min(gs for gs, _ in sym_gaps) for sym, sym_gaps in gaps.items()}
                    await self._replay_filter_state_for_gaps(gap_starts, source="GapHeal")

            except Exception as exc:
                logger.error("[GapHeal] Hata: %s", exc)

    async def _refresh_mtf_redis(self, symbol: str) -> None:
        """Tüm TF'leri Binance REST / CA'dan yenileyerek Redis ve buffer'ı günceller."""
        if not self.mtf_enabled or symbol not in self.mtf_buffers:
            return
        try:
            loop = asyncio.get_event_loop()
            limit_1m = max(1500, int(self._startup_lookback_days * 24 * 60))
            df_1m = await get_recent_klines(symbol, "1m", limit_1m)
            if not df_1m.empty:
                df_1m_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1m)
                self.mtf_buffers[symbol]["1m"] = df_1m_ind.tail(self.mtf_buffer_limits.get("1m", 1000))
                await RedisClient.set_mtf_klines(symbol, "1m", self.mtf_buffers[symbol]["1m"])
            for ws_tf in ["5m", "15m", "30m", "6h", "8h", "12h"]:
                limit = self.mtf_buffer_limits.get(ws_tf, 250)
                df_ws = await BinanceClientManager.fetch_klines(symbol=symbol, interval=ws_tf, limit=limit)
                if df_ws.empty:
                    continue
                df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_ws)
                self.mtf_buffers[symbol][ws_tf] = df_ind.tail(limit)
                await RedisClient.set_mtf_klines(symbol, ws_tf, self.mtf_buffers[symbol][ws_tf])
            for ca_tf in ["1h", "4h"]:
                limit = self.mtf_buffer_limits.get(ca_tf, 250)
                ca_df = await get_cagg_klines(symbol, ca_tf, limit)
                if ca_df.empty:
                    continue
                df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, ca_df)
                self.mtf_buffers[symbol][ca_tf] = df_ind.drop_duplicates(subset=["open_time"], keep="last")
                await RedisClient.set_mtf_klines(symbol, ca_tf, self.mtf_buffers[symbol][ca_tf])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[MTF-Refresh] %s hata: %s", symbol, exc)

    async def _post_init_catchup(self) -> None:
        """MTF init sonrası startup penceresindeki 1m gap'leri doldurur ve MTF Redis'i günceller."""
        if not self._startup_fill_end_ms:
            return

        now_ms = int(time.time() * 1000)
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
                              AND timestamp >= NOW() AT TIME ZONE 'Europe/Istanbul' - INTERVAL '2 hours'
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

    async def _ticker_refresh_loop(self) -> None:
        """Her 60 saniyede bir Binance 24h ticker REST API'sini çağırır.
        Tüm USDT sembollerinin fiyat/change%/volume verisini Redis'e yazar (TTL=90s).
        Backend durduğunda keyler otomatik expire olur, stale veri kalmaz."""
        _INTERVAL = 60
        _TTL = 90
        ticker_logger = logging.getLogger("TickerRefresh")
        redis_conn = aioredis.from_url(Config.REDIS_URL, decode_responses=True)

        while True:
            try:
                stats, funding_stats, equity_symbols = await asyncio.gather(
                    BinanceClientManager.get_24hr_ticker_stats(),
                    BinanceClientManager.get_funding_rates(),
                    BinanceClientManager.get_equity_underlying_symbols(),
                )
                funding_map = {f["symbol"]: float(f.get("lastFundingRate", 0)) for f in funding_stats}
                ticker_prices: Dict[str, float] = {}
                pipe = redis_conn.pipeline()
                written = 0
                for t in stats:
                    sym = t.get("symbol", "")
                    if not sym.endswith("USDT") or sym in equity_symbols:
                        continue
                    written += 1
                    last_price = float(t.get("lastPrice", 0))
                    if last_price > 0:
                        ticker_prices[sym] = last_price
                    change_pct = round(float(t.get("priceChangePercent", 0)), 2)
                    pipe.set(
                        f"ticker:{sym}",
                        json.dumps({
                            "price": float(t.get("lastPrice", 0)),
                            "change_pct": change_pct,
                            "volume": float(t.get("quoteVolume", 0)),
                            "high": float(t.get("highPrice", 0)),
                            "low": float(t.get("lowPrice", 0)),
                            "funding_rate": funding_map.get(sym, 0.0),
                        }),
                        ex=_TTL,
                    )
                await pipe.execute()
                self._ticker_prices = ticker_prices
                ticker_logger.info("Ticker güncellendi: %d sembol", written)
            except Exception as exc:
                ticker_logger.warning("Ticker güncelleme hatası: %s", exc)
            await asyncio.sleep(_INTERVAL)

    async def _vpmv_post_loop(self) -> None:
        """Her 10 dakikada bir post_avg boş sinyalleri günceller.
        Sinyal barından sonra POST_BARS bar oluşmuşsa post_avg/post_delta yazılır."""
        _INTERVAL = 600
        _TF_MINUTES = Config.INTERVAL_MINUTES
        await asyncio.sleep(120)

        while True:
            try:
                from database.engine import get_session as _gs
                from database.models import Signal as _Sig
                from sqlalchemy import select as _sel
                from utils.vpmv import compute_post, PRE_BARS, POST_BARS

                async with _gs() as _s:
                    rows = (await _s.execute(_sel(_Sig).where(
                        _Sig.vpmv_post_avg.is_(None),
                        _Sig.vpmv_pre_avg.isnot(None),
                    ))).scalars().all()

                updated = 0
                for sig in rows:
                    tf_min = _TF_MINUTES.get(sig.interval, 5)
                    needed_min = (PRE_BARS + POST_BARS + 1) * tf_min
                    if sig.opened_at is None:
                        continue
                    age_min = (datetime.now() - sig.opened_at).total_seconds() / 60
                    if age_min < needed_min:
                        continue

                    try:
                        raw = await RedisClient.get_mtf_klines(sig.symbol, sig.interval)
                        if raw is None or raw.empty or len(raw) < PRE_BARS + POST_BARS + 2:
                            continue

                        sig_time = sig.opened_at
                        if hasattr(raw.index, 'tz'):
                            raw_times = raw.index
                        else:
                            if "open_time" in raw.columns:
                                raw_times = pd.to_datetime(raw["open_time"], unit="ms", utc=True).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
                            else:
                                raw_times = raw.index

                        diffs = (raw_times - pd.Timestamp(sig_time)).abs()
                        bar_idx = int(diffs.argmin())
                        post_avg, post_delta = compute_post(raw, sig.signal_type, bar_idx, POST_BARS)
                        if post_avg is None:
                            continue

                        async with _gs() as _s2:
                            _row = (await _s2.execute(_sel(_Sig).where(_Sig.id == sig.id))).scalars().first()
                            if _row:
                                _row.vpmv_post_avg   = round(post_avg, 2)
                                _row.vpmv_post_delta = round(post_delta, 2)
                                await _s2.commit()
                                updated += 1
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        logger.debug("[VPMVPost] sinyal %s güncellenemedi: %s", sig.id, exc)

                if updated:
                    logger.info("[VPMVPost] %d sinyal post_avg güncellendi", updated)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("[VPMVPost] Hata: %s", exc)
            await asyncio.sleep(_INTERVAL)

    async def _oi_refresh_loop(self) -> None:
        """Her 5 dakikada bir takip edilen sembollerin Open Interest verisini çeker.
        Redis: oi:{symbol} → {oi, prev_oi, change_pct, ts}  TTL=7 dakika."""
        _INTERVAL = 300
        _TTL = 420
        oi_logger = logging.getLogger("OIRefresh")
        redis_conn = aioredis.from_url(Config.REDIS_URL, decode_responses=True)

        while True:
            try:
                symbols = list(self.symbols)
                if not symbols:
                    await asyncio.sleep(_INTERVAL)
                    continue

                new_oi = await BinanceClientManager.get_open_interest_batch(symbols)
                now_ts = int(time.time())
                pipe = redis_conn.pipeline()

                for sym, oi_val in new_oi.items():
                    key = f"oi:{sym}"
                    prev_raw = await redis_conn.get(key)
                    prev_oi = 0.0
                    if prev_raw:
                        try:
                            prev_oi = json.loads(prev_raw).get("oi", 0.0)
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            logger.debug("[OI] önceki değer parse edilemedi [%s]: %s", sym, exc)

                    change_pct = 0.0
                    if prev_oi and prev_oi != 0:
                        change_pct = round((oi_val - prev_oi) / prev_oi * 100, 2)

                    entry = {"oi": oi_val, "prev_oi": prev_oi, "change_pct": change_pct, "ts": now_ts}
                    self._oi_cache[sym] = entry
                    pipe.set(key, json.dumps(entry), ex=_TTL)

                await pipe.execute()
                oi_logger.info("OI güncellendi: %d sembol", len(new_oi))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                oi_logger.warning("OI güncelleme hatası: %s", exc)
            await asyncio.sleep(_INTERVAL)

    async def _price_publish_loop(self) -> None:
        """Her 1 saniyede canlı fiyatları tek Redis key'ine yazar (panel canlı PnL için).
        Ticker (REST, 628 sembol) taban; WS tick fiyatları daha taze olduğundan üzerine yazar.
        Paylaşımlı havuz kullanılmaz: iptal ortasında zehirlenen havuz bağlantısı
        timeout'suz set()'i sonsuza dek askıda bırakabiliyor (3 Tem vakası)."""
        redis_conn = None
        while True:
            await asyncio.sleep(1)
            prices = {**self._ticker_prices, **self._last_prices}
            if not prices:
                continue
            try:
                if redis_conn is None:
                    redis_conn = aioredis.from_url(
                        Config.REDIS_URL, decode_responses=True,
                        socket_timeout=5, socket_connect_timeout=5,
                    )
                await asyncio.wait_for(
                    redis_conn.set("prices:live", json.dumps(prices), ex=15), timeout=5
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("[PricePublish] prices:live yazılamadı, bağlantı yenilenecek: %s", exc)
                try:
                    if redis_conn is not None:
                        await redis_conn.aclose()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                redis_conn = None

    async def _manual_refresh_loop(self) -> None:
        """UI'dan açılan manuel işlemleri algılayıp manual_manager cache'ini yeniler."""
        redis = RedisClient.get_client()
        while True:
            await asyncio.sleep(10)
            try:
                val = await redis.get("manual_trade:refresh")
                if val:
                    await manual_manager.load_open_symbols()
                    await redis.delete("manual_trade:refresh")
                    logger.info("[ManualTrade] Cache yenilendi: %d açık sembol", len(manual_manager._open_symbols))  # noqa: SLF001
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("[ManualTrade] refresh kontrolü başarısız: %s", exc)

    async def _health_loop(self):
        """Her 15 dakikada price_data tazeliğini kontrol eder; gap tespit ederse doldurur.

        Sembol başına DB sorgusu (get_last_timestamp) birbirinden bağımsız — semaphore'lu
        asyncio.gather ile paralel çalıştırılıyor (Binance REST çağrısı değil, saf DB
        okuma olduğu için rate-limit kısıtı yok; DB pool kapasitesiyle (20+30 overflow,
        database/engine.py) paylaşılan bir üst sınır yeterli)."""
        _CHECK_INTERVAL = 15 * 60
        _MAX_GAP_MS = 10 * 60 * 1000  # 10 dakika (ms cinsinden)
        _GAP_SCAN_CONCURRENCY = 15

        await asyncio.sleep(60)  # Başlangıç stabilizasyonu için bekle
        semaphore = asyncio.Semaphore(_GAP_SCAN_CONCURRENCY)

        async def _check_symbol(symbol: str, now_ms: float) -> Optional[str]:
            async with semaphore:
                try:
                    last_ts = await get_last_timestamp(symbol, interval="1m")
                    if last_ts is None:
                        return None
                    if now_ms - last_ts > _MAX_GAP_MS:
                        gap_min = (now_ms - last_ts) / 60_000
                        logger.warning(
                            "[Health] %s — %.1f dakika gap tespit edildi, dolduruluyor",
                            symbol, gap_min,
                        )
                        return symbol
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug("[Health] %s timestamp kontrolü hatası: %s", symbol, exc)
                return None

        while True:
            await asyncio.sleep(_CHECK_INTERVAL)
            now_ms = time.time() * 1000
            results = await asyncio.gather(*[_check_symbol(s, now_ms) for s in list(self.symbols)])
            stale = [s for s in results if s]

            for symbol in stale:
                asyncio.create_task(self._sync_symbol_data(symbol))

            if stale:
                logger.info("[Health] %d sembol için gap fill başlatıldı", len(stale))
            else:
                logger.debug("[Health] Tüm semboller güncel")

    async def _startup_signal_reconciliation(self) -> None:
        """MTF buffer'lar hazır olduktan sonra aktif sinyalleri kontrol eder.

        PC kapalıyken oluşmuş ters sinyalleri tespit edip ilgili aktif sinyali kapatır.
        """
        from sqlalchemy import select as sa_select

        from backtest.mtf_backfill import MTFBackfillEngine
        from database.engine import get_session
        from database.models import Signal
        from signals.signal_lifecycle_manager import signal_lifecycle_manager

        recon_log = logging.getLogger("SignalRecon")

        try:
            async with get_session() as session:
                result = await session.execute(
                    sa_select(Signal).where(Signal.status == "active")
                )
                active_signals = result.scalars().all()
        except Exception as exc:
            recon_log.error("Aktif sinyal sorgusu başarısız: %s", exc)
            return

        if not active_signals:
            recon_log.info("Aktif sinyal yok, reconciliation atlandı.")
            return

        recon_log.info("%d aktif sinyal reconciliation başlıyor...", len(active_signals))
        engine = MTFBackfillEngine()
        closed = 0

        for sig in active_signals:
            symbol = sig.symbol
            interval = sig.interval

            buf = self.mtf_buffers.get(symbol, {}).get(interval)
            if buf is None or buf.empty:
                recon_log.debug("[%s] %s buffer yok, atlandı", symbol, interval)
                continue

            opened_at_ms = int(sig.opened_at.timestamp() * 1000)
            after_open = buf[buf["open_time"] > opened_at_ms]
            if after_open.empty:
                recon_log.debug("[%s] %s açılıştan sonra bar yok, atlandı", symbol, interval)
                continue

            last_direction = sig.signal_type
            rows = after_open.reset_index(drop=True)
            for i in range(len(rows)):
                row = rows.iloc[i]
                prev_row = rows.iloc[i - 1] if i > 0 else None
                signal_data = engine._check_signal_conditions(
                    row, symbol, interval, prev_row, df_mtf=rows, idx=i
                )
                if signal_data:
                    last_direction = signal_data["signal_type"]

            if last_direction != sig.signal_type:
                close_price = float(rows.iloc[-1]["close"])
                ok = await signal_lifecycle_manager.close_stale(sig.id, close_price)
                if ok:
                    recon_log.info(
                        "[%s] %s kapatıldı — offline reversal: %s → %s",
                        symbol, interval, sig.signal_type, last_direction,
                    )
                    closed += 1

        recon_log.info("Reconciliation tamamlandı: %d sinyal kapatıldı.", closed)

    async def _background_startup(self) -> None:
        """WebSocket başladıktan sonra gap fill → MTF init → post_init sırayla çalışır.

        Sıralı çalışma zorunlu: MTF init gap fill bitmeden DB'den okursa kirli buffer yüklenir.
        WS zaten ayakta olduğu için sıralama chart'a herhangi bir gap yaratmaz.
        """
        try:
            logger.info("[BackgroundStartup] Başladı (WebSocket zaten aktif).")
            filled_symbols = await self._startup_gap_fill()
            if filled_symbols:
                gap_starts = {sym: self._gap_start_ms[sym] for sym in filled_symbols if sym in self._gap_start_ms}
                await self._replay_filter_state_for_gaps(gap_starts, source="BackgroundStartup")
            if self.mtf_enabled:
                await self._initialize_mtf_dataframes(reload_symbols=filled_symbols)
            await self._post_init_catchup()
            await self._startup_signal_reconciliation()
            logger.info("[BackgroundStartup] Tamamlandı.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[BackgroundStartup] Hata: %s", exc, exc_info=True)

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
            asyncio.create_task(self._deferred_internal_gap_check(delay_seconds=240))
            # Python 3.10+ weak ref fix: periyodik loop task'larını sakla
            self._bg_tasks: list[asyncio.Task] = [
                asyncio.create_task(self._health_loop()),
                asyncio.create_task(self._continuous_gap_heal_loop()),
                asyncio.create_task(self._ticker_refresh_loop()),
                asyncio.create_task(self._oi_refresh_loop()),
                asyncio.create_task(self._price_publish_loop()),
                asyncio.create_task(self._vpmv_post_loop()),
                asyncio.create_task(self._manual_refresh_loop()),
            ]

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

        # Asyncio-native bağlantı yöneticisi aktifse onu kapat (async stop())
        if self._asyncio_ws_manager is not None:
            try:
                await asyncio.wait_for(self._asyncio_ws_manager.stop(), timeout=5.0)
                logger.info("Asyncio WS bağlantıları güvenli şekilde kapatıldı.")
            except asyncio.TimeoutError:
                logger.warning("Asyncio WS kapatma işlemi timeout oldu.")
            except Exception as e:
                logger.warning(f"Asyncio WS kapatma sırasında hata (göz ardı edildi): {e}")
            self._asyncio_ws_manager = None

        # Thread-tabanlı WebSocket client'larını kapat (senkron .stop())
        if self.ws_clients:
            for connection_id, ws_client in list(self.ws_clients.items()):
                if ws_client and hasattr(ws_client, "stop") and not asyncio.iscoroutinefunction(ws_client.stop):
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
    # Veritabanını ve tabloları oluştur — DB/pgbouncer başlangıçta henüz hazır
    # olmayabilir (ör. Postgres yeni restart oldu), bu yüzden sembol listesi
    # çekimiyle aynı retry deseni uygulanıyor. Retry'sız hâli 9 Tem'de
    # live_data_manager'ı kalıcı olarak öldürmüştü (bkz. proje hafızası).
    for attempt in range(1, 7):
        try:
            await initialize_database()
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Veritabanı başlatılamadı (deneme {attempt}/6): {e}")
            if attempt < 6:
                await asyncio.sleep(min(5 * attempt, 30))
            else:
                raise

    # Aktif pozisyonları belleğe yükle (QueuePool taşmasını önler)
    await risk_manager.load_active_symbols()
    await paper_trade_manager.load_open_symbols()
    await ha_cross_manager.load_open_symbols()
    await rsi_15m_manager.load_open_symbols()
    await manual_manager.load_open_symbols()
    await do_kirilimi_manager.load_open_symbols()
    await do_open_streak_manager.load_open_symbols()

    logger.info("En yüksek hacimli semboller Binance'ten çekiliyor...")
    symbols_to_track: List[str] = []
    for attempt in range(1, 4):
        try:
            symbols_to_track = await BinanceClientManager.get_top_volume_symbols_async(
                limit=Config.SYMBOL_LIMIT
            )
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Sembol listesi çekilemedi (deneme {attempt}/3): {e}")
            if attempt < 3:
                await asyncio.sleep(5 * attempt)
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
