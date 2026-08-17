import asyncio
import concurrent.futures
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis.asyncio as aioredis
from sqlalchemy import text

from binance_client import BinanceClientManager
from config import Config
from database.crud import (
    bulk_insert_price_data,
    bulk_insert_price_data_multi,
    delete_symbol_data,
    get_cagg_klines,
    get_last_timestamp,
    get_oldest_timestamp,
    get_recent_klines,
    initialize_database,
    refresh_cagg_chain,
)
from database.engine import get_session, run_with_db_timeout
from indicators.core import add_all_indicators, calculate_atr, calculate_rsi, truncate_after_gap
from indicators.incremental import RESYNC_INTERVAL, IndicatorState, bootstrap_state, update_state
from signals.paper_trade_manager import (
    do_kirilimi_manager,
    do_open_streak_manager,
    ha_cross_manager,
    manual_manager,
    paper_trade_manager,
    rsi_15m_manager,
    rsi_cross_live_manager,
)
from signals.risk_manager import risk_manager
from signals.signal_processor import (
    _compute_devisso_score,
    process_and_enrich_signals,
    trim_to_closed_bar,
)
from signals.tf_alignment_gate import _heikin_ashi_bull
from signals.vpm_calculator import VPMCalculator
from utils.asyncio_ws_client import AsyncioBinanceStreamManager
from utils.exceptions import BinanceAPIError, DatabaseError
from utils.heartbeat import beat, record_activity
from utils.kline_schema import check_kline_schema
from utils.logger import get_logger
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient
from utils.telegram_notify import send_telegram_message
from utils.timeframe_aggregator import TimeframeAggregator
from utils.vpmv import compute_components

# MTF init/refresh için ayrı thread pool — default executor'ı (WS sinyalleri) bloklamaz
# 12→4 (10 Tem 2026): incremental indikatör hesaplama (17.8x hızlanma, ~2.9ms/çağrı)
# sonrası 12 thread'e gerçek paralellik ihtiyacı kalmadı, sadece GIL çekişmesi
# yaratıyorlardı — bu da (WS artık aynı event loop'ta olduğu için) ping/pong
# gecikmesine ve DB timeout'larına yol açıyordu. Bkz. memory: project_data_layer_debt.md.
_MTF_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mtf_init")
# Evren-geneli ranking sweep'i (~500 sembol × 4 TF) için AYRI, küçük havuz —
# gerçek zamanlı bar birleştirmeyi kullanan _MTF_EXECUTOR'ı boğmamak için
# (14 Tem 2026 planı: "RankingWorker: Backend Hesaplar, UI Okur"). max_workers=1:
# 2 worker'la bile canlıda (15 Tem sabahı) event loop'un Redis I/O'sunu
# yetiştiremediği, havuz timeout'larına yol açtığı gözlemlendi (GIL çekişmesi) —
# tek thread + batch'ler arası gerçek asyncio.sleep ile daha yumuşak bir profil.
_RANKING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ranking"
)
_TICK_TF_WHITELIST = {"1m", "5m", "15m", "30m", "1h", "4h", "6h", "8h", "12h", "1d"}
_TICK_THROTTLE_SECS = {
    # 1m sinyal üretimi/VPMV/ATR hep KAPANAN bar yolundan tetikleniyor
    # (_update_and_process_symbol_mtf) — forming-tick sadece görüntü içindir,
    # 2sn'den 3sn'ye çekmek sinyal mantığını etkilemez (30 Tem 2026 CPU profili).
    "1m": 3,
    # 30 Tem 2026: 5m/15m/30m forming-bar dispatch'i (buf.copy()+Arrow serialize+
    # Redis set) 548 sembol × 2sn'de bir tek çekirdeği doyuruyordu (bkz. CPU
    # performans araştırması) — 1h/4h/... TF'lerinde zaten uygulanan kademeli
    # throttle deseni bu üçüne de uygulandı, 1m dokunulmadı (en hassas görünüm).
    "5m": 5,
    "15m": 8,
    "30m": 10,
    "1h": 30,
    "4h": 60,
    "6h": 60,
    "8h": 60,
    "12h": 120,
    "1d": 120,
}

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


def _has_gap(existing: pd.DataFrame, new_open_time: int) -> bool:
    """Buffer'ın son barı ile yeni barın open_time'ı arasında tipik bar
    aralığının 1.5 katından büyük boşluk varsa True döner (zorla resync için)."""
    if len(existing) < 3 or "open_time" not in existing.columns:
        return False
    diffs = existing["open_time"].diff().dropna()
    typical = diffs.median()
    if not typical or typical <= 0:
        return False
    return (new_open_time - int(existing["open_time"].iloc[-1])) > typical * 1.5


def _merge_closed_bar_and_index(
    existing: pd.DataFrame,
    new_row: dict,
    limit: int,
    state: Optional[IndicatorState],
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
        merged = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .tail(limit)
        )
        merged = truncate_after_gap(merged)
        return add_all_indicators(merged), None

    try:
        if (
            state is None
            or state.steps_since_bootstrap >= RESYNC_INTERVAL
            or _has_gap(existing, int(new_row["open_time"]))
        ):
            # İlk çağrı VEYA periyodik resync — state'in kendi içinde biriken
            # floating-point farkını ground-truth'tan (tam yeniden hesaplama) sıfırlar.
            existing = truncate_after_gap(existing)
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
            new_indicators["momentum"] = (
                ((new_row["close"] - close_then) / close_then) * 100 if close_then else np.nan
            )
        else:
            new_indicators["momentum"] = np.nan

        new_row_full = {**new_row, **new_indicators}
        merged = (
            pd.concat([existing, pd.DataFrame([new_row_full])], ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .tail(limit)
        )
        return merged, state
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("İncremental indikatör hatası, tam yeniden hesaplamaya dönülüyor: %s", e)
        new_df = pd.DataFrame([new_row])
        merged = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .tail(limit)
        )
        merged = truncate_after_gap(merged)
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
        int(close_time) if close_time is not None and pd.notna(close_time) else period_end - 1
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
        self.mtf_enabled = getattr(Config, "MTF_ENABLED", True)
        self.supported_timeframes = getattr(Config, "MTF_TIMEFRAMES", ["1m", "5m", "15m"])
        self.mtf_buffer_limits = getattr(
            Config,
            "MTF_BUFFER_LIMITS",
            {
                "1m": 1000,  # 16+ hours
                "5m": 200,  # 16+ hours
                "15m": 67,  # 16+ hours
                "1h": 24,  # 24 hours
                "4h": 12,  # 48 hours
                "1d": 7,  # 7 days
            },
        )
        # Multi-WebSocket istemcileri: Her connection için ayrı client
        self.ws_clients: Dict[int, Any] = {}  # connection_id -> ws_client
        # Asyncio-native WS taşıma katmanı yöneticisi (utils/asyncio_ws_client.py) —
        # thread-per-connection yerine aynı event loop'ta task modeli.
        self._asyncio_ws_manager: Optional[AsyncioBinanceStreamManager] = None
        self.is_ws_connected = False
        self.last_message_time: Optional[float] = None  # Son WebSocket mesajının zamanını takip et
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
        # 12 Tem 2026 gap-mekanizması konsolidasyonu: sembol -> "bu zamana kadar
        # internal-gap taraması temiz doğrulandı" (epoch ms). _startup_gap_fill
        # tarama penceresinin başlangıcıyla set eder, _continuous_gap_heal_loop
        # her turda ilerletir (bkz. o metottaki watermark mantığı).
        self._gap_watermark_ms: Dict[str, int] = {}
        self._gap_retry_count: Dict[str, int] = {}  # sembol -> ardışık kapanmamış-gap tur sayısı
        self._startup_complete_event: asyncio.Event = asyncio.Event()
        self._oi_cache: Dict[str, dict] = {}  # OI in-memory cache
        # (symbol, interval) -> [{"id": int, "signal_type": str}, ...] — VPMV canlı
        # yayını hangi sembol+TF'lerin aktif sinyali olduğunu bilsin diye (14 Tem 2026,
        # bkz. plan: "VPMV: Backend Hesaplar, UI Okur"). _active_signal_registry_loop
        # tarafından periyodik tazelenir.
        self._active_signal_registry: Dict[tuple, list] = {}
        # active_signal_registry'nin sadece sembol kümesi — Divergence yayını
        # VPMV'nin aksine sinyalin KENDİ TF'iyle sınırlı değil (kullanıcı panelde
        # farklı bir TF seçip aynı sembolü görebilmeli), bu yüzden ayrı tutuluyor.
        self._active_signal_symbols: set = set()
        # signal_id -> son ~20 VPMV değeri (Redis'e "recent" olarak yayınlanır,
        # panelin grafik çizmesi için — DB'ye gidilmeden).
        self._vpmv_recent: Dict[int, deque] = {}
        # 18 Tem 2026: bar-kapanışı burst kontrolü — _handle_websocket_message tüm
        # sembollerin (aynı dakika sınırında near-simultaneous) kapanış coroutine'lerini
        # run_coroutine_threadsafe ile fire-and-forget başlatıyor; 15 Tem refaktörü
        # sonrası her sembol artık 1 yerine kadar 4 Redis yazımı yapıyor
        # (set_mtf_klines + vpmv/divergence/atr live), bu da dakika başı ~657 sembol ×
        # 4 = binlerce eşzamanlı bağlantı talebiyle pool'u (300) tüketip
        # "No connection available" hatasına yol açıyordu. Semaphore hiçbir sembolü
        # atlamadan sadece eşzamanlılığı sınırlayıp dalgalar halinde işlemeye zorlar.
        self._bar_close_semaphore = asyncio.Semaphore(50)

        # Multi-WebSocket configuration
        self.max_streams_per_connection = 200  # Binance limit

        # Keep-Alive Ping/Pong Tracking
        self.ping_task: Optional[asyncio.Task] = None
        self.last_ping_time: Optional[float] = None
        self.ping_interval = getattr(Config, "WS_PING_INTERVAL", 20)
        self.connection_health_ok = True

        # Legacy single timeframe buffer (backward compatibility)
        self.kline_data: Dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}

        # NEW: Multi-timeframe buffers
        if self.mtf_enabled:
            self.mtf_buffers: Dict[str, Dict[str, pd.DataFrame]] = {}
            for symbol in symbols:
                self.mtf_buffers[symbol] = {}
                for tf in self.supported_timeframes:
                    self.mtf_buffers[symbol][tf] = pd.DataFrame()
            logger.info(
                f"MTF buffers initialized for {len(symbols)} symbols, {len(self.supported_timeframes)} timeframes"
            )

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
        semaphore = asyncio.Semaphore(
            2
        )  # Aynı anda max 2 istek (arka plan görevi, rate limit dostu)
        fill_starts: Dict[str, int] = {}

        async def sync_with_semaphore(symbol):
            async with semaphore:
                try:
                    _, first_fill_ms = await self._sync_symbol_data(symbol)
                    if first_fill_ms is not None:
                        fill_starts[symbol] = first_fill_ms
                    logger.info(f"[{symbol}] Tarihsel veri senkronizasyonu tamamlandı.")
                    return True
                except Exception as e:
                    logger.error(f"[{symbol}] Tarihsel veri senkronizasyonu sırasında hata: {e}")
                    return False

        # Tüm sembolleri paralel olarak işle
        tasks = [sync_with_semaphore(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_count = sum(1 for r in results if r is True)
        failed_count = len(results) - successful_count

        logger.info(
            f"Tarihsel veri senkronizasyonu tamamlandı. Başarılı: {successful_count}, Başarısız: {failed_count}"
        )

        # Düzeltme #4 (12 Tem 2026): BackgroundStartup'ın kendi replay'i henüz
        # olmadıysa (event set değilse) burada replay yapma — startup gap fill'in
        # _replay_filter_state_for_gaps çağrısıyla çakışıp aynı (sembol, TF)
        # çiftlerini gereksiz iki kere çekmeyi önler.
        if fill_starts and self._startup_complete_event.is_set():
            await self._replay_filter_state_for_gaps(fill_starts, source="HistoricalSync")

    async def _sync_symbol_data(self, symbol: str) -> Tuple[bool, Optional[int]]:
        """Helper method to sync historical data for a single symbol.

        Döndürür: (başarı, ilk-eklenen-barın open_time'ı veya None) — çağıran
        (sync_historical_data) bunu SignalFilter replay'i için toplar (düzeltme #4,
        12 Tem 2026)."""
        first_fill_open_time: Optional[int] = None
        try:
            _INTERVAL_MS_MAP = {
                "1m": 60_000,
                "3m": 180_000,
                "5m": 300_000,
                "15m": 900_000,
                "30m": 1_800_000,
                "1h": 3_600_000,
                "4h": 14_400_000,
                "1d": 86_400_000,
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
                        f"[{symbol}] Son kayıt: {datetime.fromtimestamp((start_time - 1) / 1000)}. Eksik veriler çekiliyor..."
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

                if first_fill_open_time is None:
                    first_fill_open_time = int(df_missing["open_time"].iloc[0])

                async with self.db_lock:
                    await bulk_insert_price_data(symbol, df_missing, interval=self.interval)
                total_inserted += len(df_missing)
                logger.info(
                    f"[{symbol}] {len(df_missing)} mum kaydedildi (toplam: {total_inserted})"
                )

                if len(df_missing) < 1500:
                    break

                start_time = int(df_missing["open_time"].iloc[-1]) + 1

            if total_inserted:
                logger.info(f"[{symbol}] Senkronizasyon tamamlandı: {total_inserted} mum eklendi.")
                if self.mtf_enabled:
                    await self._refresh_mtf_redis(symbol)
            elif BinanceClientManager.is_banned():
                logger.warning(
                    f"[{symbol}] Ban cooldown aktifken senkronizasyon denendi, sonuç belirsiz (gap-heal telafi edecek)."
                )
            else:
                logger.info(f"[{symbol}] Yeni veri bulunamadı, sistem güncel.")

            return True, first_fill_open_time

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
                if (
                    symbol != Config.MARKET_REFERENCE_SYMBOL
                    and recent_data["volume"].sum() < Config.MIN_VOLUME_THRESHOLD
                ):
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
            return await BinanceClientManager.fetch_klines(symbol, self.interval, limit=500)
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
                            ),
                            self.loop,
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
            logger.error(f"WebSocket mesaj işleme hatası: {e} | Mesaj: {msg}", exc_info=True)

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
            # already_owned=True: 'merged' _merge_tick_row'un ürettiği taze/özel
            # kopya — başka hiçbir yerde (self.mtf_buffers dahil) tutulmuyor, bu
            # yüzden Arrow'a çevrilirken ikinci bir kopyaya gerek yok (30 Tem 2026
            # CPU profili). Diğer set_mtf_klines çağrıları bunu KULLANMAMALI.
            await RedisClient.set_mtf_klines(symbol, interval, merged, already_owned=True)
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
        SIRALI await edilir — bkz. _handle_websocket_message'daki açıklama.

        _bar_close_semaphore ile sarılı (18 Tem 2026, bkz. __init__ açıklaması):
        tüm semboller dakika sınırında near-simultaneous tetiklendiği için,
        eşzamanlılığı sınırlamadan Redis pool'u tükeniyordu."""
        async with self._bar_close_semaphore:
            await self._update_and_process_symbol_mtf(symbol, interval, kline)
            await self._derive_and_dispatch_closing_tfs(symbol, next_open_time_ms)

    async def _process_tick_and_derive(self, symbol: str, interval: str, kline: Dict) -> None:
        """1m-türetme: _handle_tick ile _derive_and_dispatch_forming_tfs için aynı
        sıralama garantisi (bkz. _process_closed_1m_and_derive). Aynı
        _bar_close_semaphore'u paylaşır — toplam eşzamanlı Redis bağlantı
        talebi (kapanış+forming) tek noktadan sınırlanır."""
        async with self._bar_close_semaphore:
            await self._handle_tick(symbol, interval, kline)
            await self._derive_and_dispatch_forming_tfs(symbol)

    async def _refresh_1d_bar(self, symbol: str) -> None:
        """1d kapanışını 1m buffer'ından türetmek yerine (yapısal olarak imkansız
        — bkz. _derive_and_dispatch_forming_tfs docstring'i) doğrudan Binance
        REST'ten son kapanmış günü çeker; her zaman tam/doğru, UTC-anchor'lı
        (12 Tem 2026)."""
        try:
            df_1d = await BinanceClientManager.fetch_klines(symbol=symbol, interval="1d", limit=2)
            if df_1d.empty:
                return
            new_row = df_1d.iloc[-1].to_dict()
            logger.info(f"🕯️ [{symbol}] 1d mum kapandı (REST'ten). Fiyat: {new_row['close']}")
            await self._update_and_process_symbol_mtf(
                symbol, "1d", self._new_row_to_kline_dict(new_row)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[1d-Refresh] %s hata: %s", symbol, exc)

    async def _derive_and_dispatch_closing_tfs(self, symbol: str, next_open_time_ms: int) -> None:
        """1m-türetme projesi (Adım 5-6, 10 Tem 2026): bir 1m barı kapandığında,
        hangi üst TF'lerin de kapandığını (TimeframeAggregator.get_closing_timeframes)
        tespit eder, her biri için 1m buffer'ından kapanan barı türetir
        (_build_derived_closed_bar) ve mevcut _update_and_process_symbol_mtf'e AYNEN
        besler — indikatör hesaplama/sinyal üretimi/do_kirilimi-do_open_streak
        tetikleme kodlarına HİÇ dokunulmadı, sadece verinin kaynağı değişti.

        1d ayrı ele alınır (12 Tem 2026) — bkz. _refresh_1d_bar."""
        derive_tfs = [tf for tf in self.supported_timeframes if tf != "1m"]
        if not derive_tfs:
            return
        closing_tfs = TimeframeAggregator.get_closing_timeframes(next_open_time_ms, derive_tfs)
        if "1d" in closing_tfs:
            closing_tfs = [tf for tf in closing_tfs if tf != "1d"]
            asyncio.create_task(self._refresh_1d_bar(symbol))
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
            tf for tf in closing_tfs if n_bars >= TimeframeAggregator.TIMEFRAME_MINUTES.get(tf, 0)
        ]
        if not closing_tfs:
            return
        loop = asyncio.get_event_loop()
        for tf in closing_tfs:
            new_row = await loop.run_in_executor(
                _MTF_EXECUTOR, _build_derived_closed_bar, df_1m, tf
            )
            if new_row is None:
                continue
            logger.info(
                f"🕯️ [{symbol}] {tf} mum kapandı (1m'den türetildi). Fiyat: {new_row['close']}"
            )
            await self._update_and_process_symbol_mtf(
                symbol, tf, self._new_row_to_kline_dict(new_row)
            )

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
        60x daha az çağrı, executor üzerindeki büyük TF yükü ortadan kalkıyor.

        1d bu türetmenin DIŞINDA (12 Tem 2026) — 1m buffer'ı en fazla
        mtf_buffer_limits['1m']=1000 satır (~16.7 saat) tutuyor, 1d'nin tam bir
        periyodu (1440 dk) için hiçbir zaman yeterli olamaz. Eksik pencereyle
        üretilen "forming" bar, Redis/REST'ten doğru şekilde dolmuş son 1d
        satırının üzerine yanlış OHLCV yazıp indikatörlerini NaN'a düşürüyordu
        (bkz. _refresh_1d_bar — kapanış tarafı artık REST'ten besleniyor)."""
        derive_tfs = [tf for tf in self.supported_timeframes if tf != "1m" and tf != "1d"]
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
            new_row = await loop.run_in_executor(
                _MTF_EXECUTOR, _build_derived_forming_bar, df_1m, tf
            )
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
                await RedisClient.set_mtf_klines(
                    symbol, interval, self.mtf_buffers[symbol][interval]
                )
                logger.debug(f"[{symbol}] {interval} buffer updated and cached")

                # VPMV canlı yayını — SADECE bu sembol+TF'de aktif sinyali varsa
                # (14 Tem 2026 planı: "Backend Hesaplar, UI Okur"). Masaüstü
                # VpmvWorker'ın her açık sinyal için ham kline çekip compute_series
                # ile sıfırdan hesapladığı tekrarlayan işi ortadan kaldırır —
                # aynı hesaplama BURADA, zaten bellekte olan buffer'la, sadece
                # aktif sinyaller için (657 sembolün tamamı değil) bir kez yapılır.
                active_here = self._active_signal_registry.get((symbol, interval))
                if active_here:
                    await self._publish_vpmv_live(symbol, interval, merged, active_here)
                    # Verimlilik (devisso, ERSI) canlı yayını — VPMV ile AYNI
                    # tetikleyici (27 Tem 2026, kullanıcı isteği).
                    await self._publish_devisso_live(symbol, interval, merged, active_here)
                # Divergence (Z-score) canlı yayını — aynı desen, aynı gerekçe:
                # DivergenceWorker (desktop) da her sembol için ham kline çekip
                # EMA/rolling-std'i sıfırdan hesaplıyordu (14 Tem 2026 gece,
                # VPMV fix'i sonrası hâlâ devam eden bellek patlamasının kaynağı
                # bulundu). VPMV'nin aksine sinyalin KENDİ TF'iyle sınırlı değil —
                # panelde kullanıcı farklı bir TF seçip aynı sembolü görebilmeli
                # (eski kod da böyleydi, ham kline'ı seçilen TF'den çekiyordu),
                # bu yüzden SEMBOL bazında (aktif sinyali olan herhangi bir TF'de
                # varsa) her (symbol, interval) bar kapanışında yayınlanır.
                if symbol in self._active_signal_symbols:
                    await self._publish_divergence_live(symbol, interval, merged)
                # Dinamik ATR trailing yayını + açık trade registry + manuel
                # trade cache yenileme artık signal_service.py'de (17 Ağu 2026,
                # bkz. signals/registry_loops.py — _publish_atr_live process-içi
                # canlı buffer'a bağımlıydı, artık get_fresh_klines'ın zaten
                # hesapladığı df'i kullanıyor).
                # do_kirilimi/do_open_streak tetikleme artık signal_service.py'de
                # (bkz. paper trading ayrıştırması, 10 Tem 2026 cutover).

            # Legacy 1m buffer (kline_data) — sadece DB batch insert için tutuluyor
            if interval == "1m":
                new_df = pd.DataFrame([new_row])
                self.kline_data[symbol] = pd.concat(
                    [self.kline_data[symbol], new_df], ignore_index=True
                ).tail(1000)

            # Batch insert için buffer'a ekle (sadece 1m için - diğer TF'ler opsiyonel)
            if interval == "1m":
                await self._add_to_batch_buffer(symbol, new_row)

            # Sinyal üretimi (her timeframe için)
            if self.mtf_enabled:
                # Get reference data for this timeframe
                ref_df = pd.DataFrame()
                if (
                    self.ref_symbol in self.mtf_buffers
                    and interval in self.mtf_buffers[self.ref_symbol]
                ):
                    ref_df = await loop.run_in_executor(
                        _MTF_EXECUTOR,
                        pd.DataFrame.copy,
                        self.mtf_buffers[self.ref_symbol][interval],
                    )

                # Minimum bar requirements per timeframe
                min_bars = {"1m": 200, "5m": 100, "15m": 67, "1h": 24, "4h": 12, "1d": 7}.get(
                    interval, 100
                )

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
                        df_copy = trim_to_closed_bar(df_copy, new_row["open_time"])
                        ref_df_trimmed = trim_to_closed_bar(ref_df, new_row["open_time"])
                        task = asyncio.create_task(
                            process_and_enrich_signals(
                                symbol=symbol,
                                df=df_copy,
                                ref_df=ref_df_trimmed,
                                interval=interval,
                                oi_data=oi_data_json,
                                symbol_buffers=self.mtf_buffers.get(symbol, {}),
                                dry_run=False,
                            )
                        )
                        self.processing_tasks.add(task)
                        task.add_done_callback(self.processing_tasks.discard)
                        logger.info(f"🎯 [{symbol}] {interval} sinyal üretimi başlatıldı")
                    await RedisClient.publish_kline_closed_event(
                        symbol, interval, new_row["open_time"]
                    )

        except Exception as e:
            logger.error(f"[{symbol}] {interval} veri güncelleme hatası: {e}", exc_info=True)

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
                    # 29 Tem 2026: bar SAYISI yeterli görünse de içeride kapanma/
                    # yeniden başlatma kaynaklı bir boşluk olabilir (bkz. ESPUSDT
                    # 15m/1h vakası) — truncate_after_gap ile boşluk sonrası
                    # temiz kuyruk yeterli değilse bu TF'i "eksik" say, REST/CA'dan
                    # taze çekilsin.
                    clean = truncate_after_gap(df)
                    if len(clean) >= limit * min_bars_ratio:
                        self.mtf_buffers[sym][tf] = clean.tail(limit)
                        continue
                all_tf_ok = False
            return sym, all_tf_ok

        redis_results = await asyncio.gather(*[_load_from_redis(s) for s in self.symbols])
        for sym, ok in redis_results:
            if ok:
                redis_hit.add(sym)
        logger.info(
            "[MTF] Redis hızlı yükleme: %d/%d sembol tam yüklendi.",
            len(redis_hit),
            len(self.symbols),
        )

        # ── Batch yükleme: Redis'te eksik/yetersiz olanlar + zorla yenilenmesi gerekenler ────────
        force_reload = reload_symbols if reload_symbols is not None else set()
        symbols_to_reload = [s for s in self.symbols if s not in redis_hit or s in force_reload]

        if not symbols_to_reload:
            logger.info(
                "🎉 MTF Batch Initialization tamamlandı! WebSocket canlı mod başlatılabilir."
            )
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
        failed_symbols: list[str] = []
        for i in range(0, total_symbols, batch_size):
            batch = symbols_to_reload[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} sembol yükleniyor...")

            # Batch timeout: 120s — takılı semboller iptal edilerek batch ilerler
            batch_tasks = {s: asyncio.create_task(self._load_symbol_all_timeframes(s)) for s in batch}
            _, pending = await asyncio.wait(batch_tasks.values(), timeout=120)
            for p in pending:
                p.cancel()
                try:
                    await p
                except (asyncio.CancelledError, Exception):
                    pass
            if pending:
                logger.warning(f"⚠️ Batch {batch_num}: {len(pending)} sembol timeout ile atlandı.")

            results = {
                s: (t.result() if not t.cancelled() and t.exception() is None else None)
                for s, t in batch_tasks.items()
            }
            for s, r in results.items():
                if r is None or not r[0]:
                    failed_symbols.append(s)

            # Başarı oranını hesapla
            success_count = sum(1 for r in results.values() if r is not None and r[0])
            total_rest_calls = sum(r[1] for r in results.values() if r is not None)
            logger.info(
                f"✅ Batch {batch_num}/{total_batches} tamamlandı ({success_count}/{len(batch)} başarılı, {total_rest_calls} REST çağrısı)"
            )

        # ── Başarısız sembolleri bir kez daha dene (16 Ağu 2026: restart sonrası
        # max_locks_per_transaction/geçici DB baskısı yüzünden 129/550 sembol
        # sessizce başarısız oluyordu, hiç retry/alarm yoktu — Telegram'da hiçbir
        # iz bırakmadan saatlerce eksik buffer'la kalıyorlardı) ──────────────────
        if failed_symbols:
            logger.warning(
                f"🔁 {len(failed_symbols)} sembol ilk denemede başarısız oldu, yeniden deneniyor: "
                f"{', '.join(failed_symbols[:10])}{' ...' if len(failed_symbols) > 10 else ''}"
            )
            await asyncio.sleep(5)
            retry_tasks = {s: asyncio.create_task(self._load_symbol_all_timeframes(s)) for s in failed_symbols}
            _, pending = await asyncio.wait(retry_tasks.values(), timeout=120)
            for p in pending:
                p.cancel()
                try:
                    await p
                except (asyncio.CancelledError, Exception):
                    pass

            still_failed = [
                s
                for s, t in retry_tasks.items()
                if t in pending or t.cancelled() or t.exception() is not None or not t.result()[0]
            ]
            if still_failed:
                logger.error(
                    f"❌ {len(still_failed)} sembol retry sonrası da başarısız: "
                    f"{', '.join(still_failed[:20])}{' ...' if len(still_failed) > 20 else ''}"
                )
                await send_telegram_message(
                    f"⚠️ MTF backfill: {len(still_failed)}/{total_symbols} sembol retry sonrası "
                    f"da yüklenemedi (buffer eksik kalabilir): "
                    f"{', '.join(still_failed[:15])}{' ...' if len(still_failed) > 15 else ''}"
                )
            else:
                logger.info(f"✅ Retry başarılı: {len(failed_symbols)} sembolün tamamı yüklendi.")

        logger.info("🎉 MTF Batch Initialization tamamlandı! WebSocket canlı mod başlatılabilir.")

    async def _load_symbol_all_timeframes(self, symbol: str) -> Tuple[bool, int]:
        """
        Bir sembol için tüm timeframe'leri API'den yükle.

        Args:
            symbol: Sembol adı

        Returns:
            (bool, int): (yükleme başarılıysa True, yapılan REST çağrı sayısı) —
            REST çağrı sayısı, batch gecikmesinin gerçek ağırlığa göre hesaplanması için.
        """
        rest_call_count = 0
        try:
            # Binance'ten çekilecek TF'ler (1m ve aggregate edilebilecekler hariç)
            binance_timeframe_limits = {
                "1h": 250,  # ~10 gün  — aggregate için 1m yetersiz (15k bar lazım)
                "4h": 250,  # ~41 gün
                "1d": 250,  # ~250 gün
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
                df_1m = await BinanceClientManager.fetch_klines(
                    symbol=symbol, interval="1m", limit=1500
                )

            if not df_1m.empty:
                df_1m_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1m)
                self.mtf_buffers[symbol]["1m"] = df_1m_ind.tail(
                    self.mtf_buffer_limits.get("1m", 250)
                )
                await RedisClient.set_mtf_klines(symbol, "1m", self.mtf_buffers[symbol]["1m"])
                loaded_count += 1

            # ── 30m: CA karşılığı yok — Redis-first, REST fallback ──────────────
            for ws_tf in ["30m"]:
                limit = self.mtf_buffer_limits.get(ws_tf, 250)
                cached = await RedisClient.get_mtf_klines(symbol, ws_tf, limit=limit)
                check_kline_schema(cached, f"RedisCache.{ws_tf}")
                if cached is not None and len(cached) >= limit // 2:
                    cached = truncate_after_gap(cached)
                if cached is not None and len(cached) >= limit // 2:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, cached)
                    self.mtf_buffers[symbol][ws_tf] = df_ind.tail(limit)
                    loaded_count += 1
                else:
                    binance_call_made = True
                    rest_call_count += 1
                    df_ws = await BinanceClientManager.fetch_klines(
                        symbol=symbol, interval=ws_tf, limit=limit
                    )
                    if not df_ws.empty:
                        df_ind = await loop.run_in_executor(
                            _MTF_EXECUTOR, add_all_indicators, df_ws
                        )
                        self.mtf_buffers[symbol][ws_tf] = df_ind.tail(limit)
                        await RedisClient.set_mtf_klines(
                            symbol, ws_tf, self.mtf_buffers[symbol][ws_tf]
                        )
                        loaded_count += 1

            # ── 5m/15m/1h/4h/6h/8h/12h: CA view'larından (boşluksuz, 1m'den otomatik
            # türetilmiş, hiyerarşik zincir: 5m←1m, 15m←5m, 1h←15m, 4h←1h, 6h←1h,
            # 8h/12h←4h) — 29 Tem 2026: eskiden 5m/15m/6h/8h/12h ayrı Binance REST
            # çağrısıyla çekiliyordu; restart sonrası TÜM semboller aynı anda reload'a
            # düşünce (bkz. truncate_after_gap fix'i) bu REST hacmi IP ban fırtınasına
            # yol açıyordu. CA'lar zaten var ve sürekli refresh policy'li — REST'e hiç
            # gerek yok, ban riski kaynağında ortadan kalkıyor.
            for tf in ["5m", "15m", "1h", "4h", "6h", "8h", "12h"]:
                limit = self.mtf_buffer_limits.get(tf, 250)
                ca_df = await get_cagg_klines(symbol, tf, limit)
                if not ca_df.empty:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, ca_df)
                    self.mtf_buffers[symbol][tf] = df_ind.drop_duplicates(
                        subset=["open_time"], keep="last"
                    )
                    await RedisClient.set_mtf_klines(symbol, tf, self.mtf_buffers[symbol][tf])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] {tf}: CA'dan yüklendi ({len(ca_df)} bar)")
                else:
                    logger.warning(f"[{symbol}] {tf}: CA boş")

            # ── 1d: Redis cache → yoksa Binance (CA için çok fazla 1m gerekir) ──
            limit_1d = binance_timeframe_limits.get("1d", 250)
            cached_df = await RedisClient.get_mtf_klines(symbol, "1d", limit=limit_1d)
            if cached_df is not None and len(cached_df) >= limit_1d // 2:
                cached_df = truncate_after_gap(cached_df)
            if cached_df is not None and len(cached_df) >= limit_1d // 2:
                self.mtf_buffers[symbol]["1d"] = cached_df.drop_duplicates(
                    subset=["open_time"], keep="last"
                )
                loaded_count += 1
                logger.debug(f"[{symbol}] 1d: Redis cache'den yüklendi ({len(cached_df)} bar)")
            else:
                binance_call_made = True
                rest_call_count += 1
                df_1d = await BinanceClientManager.fetch_klines(
                    symbol=symbol, interval="1d", limit=limit_1d
                )
                if not df_1d.empty:
                    df_ind = await loop.run_in_executor(_MTF_EXECUTOR, add_all_indicators, df_1d)
                    self.mtf_buffers[symbol]["1d"] = df_ind.tail(limit_1d).drop_duplicates(
                        subset=["open_time"], keep="last"
                    )
                    await RedisClient.set_mtf_klines(symbol, "1d", self.mtf_buffers[symbol]["1d"])
                    loaded_count += 1
                    logger.debug(f"[{symbol}] 1d: Binance'ten çekildi ({len(df_1d)} bar)")
                else:
                    logger.warning(f"[{symbol}] 1d: Veri boş")

            src = "REST" if binance_call_made else "Redis"
            logger.info(
                f"✅ [{symbol}] {loaded_count} TF yüklendi (1m=DB, 30m={src}, "
                "5m/15m/1h/4h/6h/8h/12h=CA, 1d=cache)"
            )
            return True, rest_call_count

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
                        conn_id
                        for conn_id, ts in self._conn_last_message_time.items()
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
        stream_tfs = ["1m"]

        logger.info(
            f"🚀 Multi-Timeframe WebSocket başlatılıyor: {len(self.symbols)} sembol × {len(stream_tfs)} TF"
        )

        all_streams = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols]

        total_streams = len(all_streams)
        # Allow override from central config (new tunable)
        self.max_streams_per_connection = getattr(
            Config, "WS_MAX_STREAMS_PER_CONNECTION", self.max_streams_per_connection
        )
        connections_needed = (
            total_streams + self.max_streams_per_connection - 1
        ) // self.max_streams_per_connection

        logger.info(
            f"📊 Toplam stream: {total_streams} ({len(self.symbols)} sembol × {len(stream_tfs)} TF)"
        )
        logger.info(
            f"🔌 Gerekli connection: {connections_needed} (max {self.max_streams_per_connection} stream/connection)"
        )

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
            base_url = getattr(Config, "BINANCE_WS_BASE", "wss://fstream.binance.com/market")
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
                self._conn_symbols[connection_id] = sorted(
                    {s.split("@")[0].upper() for s in conn.streams}
                )
                logger.info(
                    f"✅ Connection #{connection_id} başarıyla kuruldu ({len(conn.streams)} stream, asyncio)"
                )

            # Bağlantılar kurulduktan sonra kısa bir bekleme
            await asyncio.sleep(2)

            self.is_ws_connected = True
            self.reconnect_attempt = 0  # Başarılı bağlantıda backoff'u sıfırla
            self.connection_health_ok = True  # Health durumunu sıfırla
            logger.info(
                f"🎉 Multi-WebSocket başarıyla başlatıldı: {connections_needed} connection, {total_streams} stream"
            )

            # Keep-Alive ping task'ını başlat
            await self._start_ping_task()

        except Exception as e:
            logger.error(f"Multi-WebSocket başlatma hatası: {e}", exc_info=True)
            self.is_ws_connected = False
            raise

    async def _deferred_sync_historical(self, delay_seconds: int = 30):
        """sync_historical_data'yı gecikmeyle arka planda çalıştırır."""
        await asyncio.sleep(delay_seconds)
        logger.info(
            f"🔄 Tarihsel veri senkronizasyonu başlatılıyor (arka plan, {delay_seconds}s sonra)..."
        )
        await self.sync_historical_data()

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
                    text(
                        """
                        SELECT MAX(timestamp)
                        FROM price_data
                        WHERE symbol = ANY(:syms) AND interval = '1m'
                    """
                    ),
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
                    text(
                        """
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
                    """
                    ),
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
                    text(
                        """
                        SELECT symbol, MAX(timestamp)
                        FROM price_data
                        WHERE symbol = ANY(:syms) AND interval = '1m'
                        GROUP BY symbol
                    """
                    ),
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
            logger.info(
                "[Startup] 1m: %d sembolde %d gap bulundu, dolduruluyor...",
                len(all_gaps),
                total_gaps,
            )

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
                                    symbol=sym,
                                    interval="1m",
                                    limit=1000,
                                    startTime=fetch_start,
                                )
                                break
                            except Exception as exc:
                                logger.warning(
                                    "[Startup] %s API hatası (deneme %d/3): %s",
                                    sym,
                                    attempt + 1,
                                    exc,
                                )
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
                batch = items[i : i + _GAP_BATCH_SIZE]
                results = await asyncio.gather(
                    *[_fill_one_symbol(sym, gaps) for sym, gaps in batch],
                    return_exceptions=True,
                )
                total_filled += sum(r[0] for r in results if isinstance(r, tuple))
                total_rest_calls = sum(r[1] for r in results if isinstance(r, tuple))
                if i + _GAP_BATCH_SIZE < len(items):
                    wait = max(
                        0.5, (total_rest_calls * _WEIGHT_PER_CALL_GAP / _WEIGHT_BUDGET_PER_MIN) * 60
                    )
                    await asyncio.sleep(wait)

            logger.info("[Startup] 1m gap fill tamamlandı: %d bar eklendi", total_filled)

            if total_filled:
                # 1m'e geriye dönük bar eklendi — cagg zinciri (5m→15m→1h→4h→
                # 6h/8h/12h) OTOMATİK yenilenmez, elle tetiklemezsek grafik/MTF
                # buffer'ları backfill'i hiç görmez (bkz. memory: 13 Tem CA
                # backfill pilotu Bulgu 4, 29 Tem: cagg_1h'de doğrudan doğrulandı).
                min_gap_start_ms = min(gs for gaps in all_gaps.values() for gs, _ in gaps)
                refresh_start = datetime.fromtimestamp(min_gap_start_ms / 1000)
                refresh_end = datetime.fromtimestamp(now_ms / 1000)
                logger.info(
                    "[Startup] CAGG zinciri yenileniyor: %s → %s", refresh_start, refresh_end
                )
                try:
                    await refresh_cagg_chain(refresh_start, refresh_end)
                    logger.info("[Startup] CAGG zinciri yenilendi.")
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("[Startup] CAGG zinciri yenileme hatası: %s", exc)
        else:
            logger.info("[Startup] 1m: gap yok.")

        self._startup_fill_end_ms = int(time.time() * 1000)
        self._startup_lookback_days = float(lookback_days)
        # SignalFilter replay'i için her sembolün en erken gap başlangıcı saklanır
        # (bkz. _background_startup — sinyal filtresi referans noktalarını bu
        # zamandan itibaren geçmiş fiyat hareketiyle senkronize eder).
        self._gap_start_ms = {sym: min(gs for gs, _ in gaps) for sym, gaps in all_gaps.items()}

        # Düzeltme #1 (12 Tem 2026): watermark = taranan pencerenin BAŞLANGICI (fill
        # bitiş zamanı DEĞİL) — _continuous_gap_heal_loop buradan itibaren ilerleyecek,
        # fill sırasında oluşabilecek yeni gap'ler de bir sonraki turda yakalanır.
        scan_start_ms = now_ms - int(lookback_days * 86_400_000)
        for sym in symbols_list:
            self._gap_watermark_ms[sym] = scan_start_ms

        return set(all_gaps.keys())

    async def _replay_filter_state_for_gaps(
        self, gap_starts: Dict[str, int], source: str = ""
    ) -> None:
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
                        symbol=sym,
                        interval=tf,
                        limit=total_bars,
                        startTime=gap_start - _CONTEXT_BARS * bar_ms,
                    )
                else:
                    # Gap API limitinden uzun — en güncel (şu ana en yakın) kısmı önceliklendir.
                    df = await BinanceClientManager.fetch_klines(
                        symbol=sym,
                        interval=tf,
                        limit=_MAX_API_LIMIT,
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
            batch = pairs[i : i + _BATCH_SIZE]
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
            source or "Replay",
            len(gap_starts),
            replay_bars,
        )

    async def _mtf_resync_loop(self) -> None:
        """MTF buffer'ları (5m/15m/1h/4h/6h/8h/12h) periyodik olarak CA ile
        yeniden senkronize eder (16 Ağu 2026).

        _initialize_mtf_dataframes SADECE process başlangıcında çağrılıyordu.
        Restart anında CA henüz güncel/tam materialize olmamışsa (refresh
        job'ları henüz yetişmemişse) buffer eksik bir anlık görüntüyle
        doluyor ve WS'nin incremental append'i bunu asla telafi edemiyordu
        — canlı kanıt: bir heartbeat-testi restart'ı sonrası BTCUSDT 15m/1h
        buffer'ı 250/250'den 47/13'e düştü, saatlerce öyle kaldı, grafik
        boşluk gösterdi. _continuous_gap_heal_loop bunun price_data/1m
        karşılığını çözüyor ama Redis'teki MTF katmanını hiç kapsamıyor —
        bu döngü o boşluğu kapatır. _initialize_mtf_dataframes zaten kendi
        içinde "Redis'te limit*0.5 üzerinde mi" kontrolü yapıp SADECE
        eksik kalan sembolleri CA'dan yeniden çekiyor — sağlıklı semboller
        için gereksiz Postgres/CA yükü oluşturmaz."""
        _INTERVAL_SEC = 1800  # 30 dakikada bir

        try:
            await asyncio.wait_for(self._startup_complete_event.wait(), timeout=600)
        except asyncio.TimeoutError:
            pass

        while True:
            await asyncio.sleep(_INTERVAL_SEC)
            if not self.mtf_enabled:
                continue
            try:
                await self._initialize_mtf_dataframes(reload_symbols=None)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("[MTFResync] Hata: %s", exc, exc_info=True)

    async def _continuous_gap_heal_loop(self) -> None:
        """Merkezi sürekli gap-doldurma mekanizması (12 Tem 2026 konsolidasyonu:
        eski _deferred_internal_gap_check/_post_init_catchup/_health_loop'un işlevini
        de kapsıyor). Sabit 1 saatlik pencere yerine sembol başına watermark
        (self._gap_watermark_ms — bkz. __init__ ve _startup_gap_fill) kullanılır:
        "bu sembol için buraya kadar internal-gap taraması temiz doğrulandı" noktası.

        Watermark ilerletme kuralı: bir sembolün o turdaki TÜM gap'leri kapandıysa
        (veya hiç gap yoksa) watermark tur başlangıcına ilerler. Ban/hata yüzünden
        kapanmayan gap varsa watermark İLERLEMEZ — ban ne kadar sürerse sürsün, kalkınca
        watermark'tan itibaren tüm boşluk taranır (eski sabit-1-saat kör noktası kapandı).
        Art arda _MAX_RETRY_TURNS tur kapanmazsa (örn. delisted sembol), watermark yine de
        zorla ilerletilir — aksi halde tek sorunlu sembol min-watermark floor'unu sonsuza
        dek geriye çekip her turun sorgusunu gereksiz genişletir."""
        import time as _t

        _INTERVAL_MS = 60_000
        _THRESHOLD_MS = _INTERVAL_MS * 2
        _TAIL_THRESHOLD_MS = _INTERVAL_MS * 3  # son bar 3dk'dan eskiyse tail gap
        _MAX_RETRY_TURNS = 12  # ~1 saat (5dk tur) ardışık başarısızlıktan sonra pes et

        # Startup'ta oluşan gap'ler _startup_gap_fill/_background_startup tarafından
        # zaten ele alınıyor — ilk tur, startup bitiş sinyalini bekleyip hemen başlar
        # (eski sabit 300s bekleme, startup-bitişi ile ilk tur arasında kör pencere
        # bırakıyordu; bu pencereyi eskiden _post_init_catchup kapatıyordu).
        try:
            await asyncio.wait_for(self._startup_complete_event.wait(), timeout=600)
        except asyncio.TimeoutError:
            logger.warning(
                "[GapHeal] Startup tamamlanma sinyali 600s'de gelmedi, yine de başlanıyor."
            )

        while True:
            scan_start_ms = int(_t.time() * 1000)
            try:
                symbols_list = list(self.symbols)

                def _watermark_for(sym: str) -> int:
                    if sym in self._gap_watermark_ms:
                        return self._gap_watermark_ms[sym]
                    if self._startup_fill_end_ms:
                        return scan_start_ms - int(self._startup_lookback_days * 86_400_000)
                    return scan_start_ms - 3_600_000  # mevcut eski davranış: ~1 saat

                floor_ms = min(
                    (_watermark_for(s) for s in symbols_list), default=scan_start_ms - 3_600_000
                )
                floor_dt = datetime.fromtimestamp(floor_ms / 1000)

                # İç gap tespiti (LAG) — tek batched sorgu (en eski watermark'tan itibaren),
                # per-symbol watermark filtresi Python tarafında uygulanır.
                async def _fetch_lag_gaps():
                    async with get_session() as session:
                        result = await session.execute(
                            text(
                                """
                                SELECT symbol, prev_ts, curr_ts
                                FROM (
                                    SELECT symbol,
                                           timestamp AS curr_ts,
                                           LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                                    FROM price_data
                                    WHERE symbol = ANY(:syms) AND interval = '1m'
                                      AND timestamp >= :floor
                                ) t
                                WHERE prev_ts IS NOT NULL
                                  AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > :thresh
                                ORDER BY symbol, prev_ts
                            """
                            ),
                            {"syms": symbols_list, "floor": floor_dt, "thresh": _THRESHOLD_MS},
                        )
                        return result.fetchall()

                rows = await run_with_db_timeout(_fetch_lag_gaps())

                gaps: dict[str, list[tuple[int, int]]] = {}
                for sym, prev_dt, curr_dt in rows:
                    g_start = int(prev_dt.timestamp() * 1000)
                    g_end = int(curr_dt.timestamp() * 1000)
                    if g_start < _watermark_for(sym):
                        continue  # bu sembol için zaten temiz doğrulanmış aralık
                    gaps.setdefault(sym, []).append((g_start, g_end))

                # Kuyruk gap tespiti: son bar'dan şu ana kadar boşluk var mı?
                async def _fetch_tail_gaps():
                    async with get_session() as session:
                        tail_result = await session.execute(
                            text(
                                """
                                SELECT symbol, MAX(timestamp)
                                FROM price_data
                                WHERE symbol = ANY(:syms) AND interval = '1m'
                                GROUP BY symbol
                            """
                            ),
                            {"syms": symbols_list},
                        )
                        return tail_result.fetchall()

                tail_rows = await run_with_db_timeout(_fetch_tail_gaps())

                for sym, last_dt in tail_rows:
                    if last_dt is None:
                        continue
                    tail_ms = int(last_dt.timestamp() * 1000)
                    if (scan_start_ms - tail_ms) > _TAIL_THRESHOLD_MS:
                        existing = gaps.get(sym, [])
                        if not any(gs >= tail_ms for gs, _ in existing):
                            gaps.setdefault(sym, []).append((tail_ms, scan_start_ms - _INTERVAL_MS))

                async def _fill_gap(
                    sym: str, gap_start_ms: int, gap_end_ms: int
                ) -> tuple[int, bool]:
                    """Gap'i doldurur, (eklenen bar sayısı, tamamen kapandı mı) döner.
                    "Kapandı" = son eklenen bar gap_end_ms'e ulaştı — ban/boş-yanıt/
                    hata yüzünden erken kesilirse False (watermark ilerlemez)."""
                    filled = 0
                    fetch_start = gap_start_ms + _INTERVAL_MS
                    closed = fetch_start >= gap_end_ms
                    while fetch_start < gap_end_ms:
                        await asyncio.sleep(0.3)
                        try:
                            df = await BinanceClientManager.fetch_klines(
                                symbol=sym,
                                interval="1m",
                                limit=1000,
                                startTime=fetch_start,
                            )
                        except Exception:
                            closed = False
                            break
                        if df is None or df.empty:
                            closed = False
                            break
                        df = df[df["open_time"] < gap_end_ms]
                        if df.empty:
                            closed = False
                            break
                        async with self.db_lock:
                            await bulk_insert_price_data(sym, df, interval="1m")
                        filled += len(df)
                        last_ts = int(df["open_time"].iloc[-1])
                        closed = (last_ts + _INTERVAL_MS) >= gap_end_ms
                        if last_ts <= fetch_start or len(df) < 1000:
                            break
                        fetch_start = last_ts + _INTERVAL_MS
                    return filled, closed

                if gaps:
                    logger.warning("[GapHeal] %d sembolde gap bulundu, dolduruluyor...", len(gaps))

                total_filled = 0
                gap_starts: Dict[str, int] = {}
                mtf_refresh_syms: list[str] = []

                for sym, sym_gaps in gaps.items():
                    sym_filled = 0
                    sym_all_closed = True
                    for gap_start_ms, gap_end_ms in sym_gaps:
                        filled, closed = await _fill_gap(sym, gap_start_ms, gap_end_ms)
                        sym_filled += filled
                        sym_all_closed = sym_all_closed and closed
                    total_filled += sym_filled
                    if sym_filled:
                        gap_starts[sym] = min(gs for gs, _ in sym_gaps)
                        mtf_refresh_syms.append(sym)

                    if sym_all_closed:
                        self._gap_watermark_ms[sym] = scan_start_ms
                        self._gap_retry_count[sym] = 0
                    else:
                        retry = self._gap_retry_count.get(sym, 0) + 1
                        self._gap_retry_count[sym] = retry
                        if retry >= _MAX_RETRY_TURNS:
                            logger.warning(
                                "[GapHeal] %s %d turdur gap kapanmadı, watermark zorla ilerletiliyor (pes edildi).",
                                sym,
                                retry,
                            )
                            self._gap_watermark_ms[sym] = scan_start_ms
                            self._gap_retry_count[sym] = 0

                # Gap'i olmayan (bu turda temiz taranmış) semboller için watermark ilerler.
                for sym in symbols_list:
                    if sym not in gaps:
                        self._gap_watermark_ms[sym] = scan_start_ms
                        self._gap_retry_count[sym] = 0

                if total_filled:
                    logger.info("[GapHeal] %d bar dolduruldu.", total_filled)
                    if gap_starts:
                        refresh_start = datetime.fromtimestamp(min(gap_starts.values()) / 1000)
                        refresh_end = datetime.fromtimestamp(scan_start_ms / 1000)
                        try:
                            await refresh_cagg_chain(refresh_start, refresh_end)
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            logger.warning("[GapHeal] CAGG zinciri yenileme hatası: %s", exc)
                    if self.mtf_enabled:
                        for sym in mtf_refresh_syms:
                            await self._refresh_mtf_redis(sym)
                        logger.info(
                            "[GapHeal] %d sembol MTF Redis yenilendi.", len(mtf_refresh_syms)
                        )
                    await self._replay_filter_state_for_gaps(gap_starts, source="GapHeal")
                else:
                    logger.debug("[GapHeal] Gap yok.")

            except Exception as exc:
                logger.error("[GapHeal] Hata: %s", exc)

            await asyncio.sleep(300)

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
                self.mtf_buffers[symbol]["1m"] = df_1m_ind.tail(
                    self.mtf_buffer_limits.get("1m", 1000)
                )
                await RedisClient.set_mtf_klines(symbol, "1m", self.mtf_buffers[symbol]["1m"])
            for ws_tf in ["5m", "15m", "30m", "6h", "8h", "12h", "1d"]:
                limit = self.mtf_buffer_limits.get(ws_tf, 250)
                df_ws = await BinanceClientManager.fetch_klines(
                    symbol=symbol, interval=ws_tf, limit=limit
                )
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
                self.mtf_buffers[symbol][ca_tf] = df_ind.drop_duplicates(
                    subset=["open_time"], keep="last"
                )
                await RedisClient.set_mtf_klines(symbol, ca_tf, self.mtf_buffers[symbol][ca_tf])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[MTF-Refresh] %s hata: %s", symbol, exc)

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
                funding_map = {
                    f["symbol"]: float(f.get("lastFundingRate", 0)) for f in funding_stats
                }
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
                        json.dumps(
                            {
                                "price": float(t.get("lastPrice", 0)),
                                "change_pct": change_pct,
                                "volume": float(t.get("quoteVolume", 0)),
                                "high": float(t.get("highPrice", 0)),
                                "low": float(t.get("lowPrice", 0)),
                                "funding_rate": funding_map.get(sym, 0.0),
                            }
                        ),
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
                from sqlalchemy import select as _sel

                from database.engine import get_session as _gs
                from database.models import PaperTrade as _PT
                from database.models import Signal as _Sig
                from utils.vpmv import POST_BARS, PRE_BARS, compute_post

                async with _gs() as _s:
                    rows = (
                        (
                            await _s.execute(
                                _sel(_Sig).where(
                                    _Sig.vpmv_post_avg.is_(None),
                                    _Sig.vpmv_pre_avg.isnot(None),
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )

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
                        if hasattr(raw.index, "tz"):
                            raw_times = raw.index
                        else:
                            if "open_time" in raw.columns:
                                raw_times = (
                                    pd.to_datetime(raw["open_time"], unit="ms", utc=True)
                                    .dt.tz_convert("Europe/Istanbul")
                                    .dt.tz_localize(None)
                                )
                            else:
                                raw_times = raw.index

                        diffs = (raw_times - pd.Timestamp(sig_time)).abs()
                        bar_idx = int(diffs.argmin())
                        post_avg, post_delta = compute_post(
                            raw, sig.signal_type, bar_idx, POST_BARS
                        )
                        if post_avg is None:
                            continue

                        async with _gs() as _s2:
                            _row = (
                                (await _s2.execute(_sel(_Sig).where(_Sig.id == sig.id)))
                                .scalars()
                                .first()
                            )
                            if _row:
                                _row.vpmv_post_avg = round(post_avg, 2)
                                _row.vpmv_post_delta = round(post_delta, 2)
                                # 2 Ağu 2026 (migration 029): aynı değer daha önce
                                # SADECE Signal'e yazılıyordu, PaperTrade.vpmv_post_avg/
                                # delta hep NULL kalıyordu — bu sinyalden doğmuş bir
                                # paper trade varsa o da güncelleniyor.
                                _pt_row = (
                                    (await _s2.execute(_sel(_PT).where(_PT.signal_id == sig.id)))
                                    .scalars()
                                    .first()
                                )
                                if _pt_row:
                                    _pt_row.vpmv_post_avg = round(post_avg, 2)
                                    _pt_row.vpmv_post_delta = round(post_delta, 2)
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

                    entry = {
                        "oi": oi_val,
                        "prev_oi": prev_oi,
                        "change_pct": change_pct,
                        "ts": now_ts,
                    }
                    self._oi_cache[sym] = entry
                    pipe.set(key, json.dumps(entry), ex=_TTL)

                await pipe.execute()
                oi_logger.info("OI güncellendi: %d sembol", len(new_oi))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                oi_logger.warning("OI güncelleme hatası: %s", exc)
            await asyncio.sleep(_INTERVAL)

    async def _active_signal_registry_loop(self) -> None:
        """Her 60 saniyede bir aktif sinyalleri DB'den çekip
        `self._active_signal_registry`'yi tazeler — VPMV canlı yayınının
        (bkz. plan: "VPMV: Backend Hesaplar, UI Okur") hangi sembol+TF'lerde
        çalışacağını bilmesi için. signal_service.py AYRI bir süreç olduğundan
        (in-process callback yok), bu periyodik DB okuması gerekli."""
        from sqlalchemy import select as _sel

        from database.models import Signal as _Sig

        _INTERVAL = 60
        reg_logger = logging.getLogger("ActiveSignalRegistry")

        while True:
            try:
                async with get_session() as session:
                    rows = (
                        await run_with_db_timeout(
                            session.execute(
                                _sel(_Sig.id, _Sig.symbol, _Sig.interval, _Sig.signal_type).where(
                                    _Sig.status == "active"
                                )
                            )
                        )
                    ).all()
                new_registry: Dict[tuple, list] = {}
                for sig_id, symbol, interval, signal_type in rows:
                    new_registry.setdefault((symbol, interval), []).append(
                        {"id": sig_id, "signal_type": signal_type}
                    )
                self._active_signal_registry = new_registry
                self._active_signal_symbols = {sym for sym, _ in new_registry}
                reg_logger.debug(
                    "%d aktif sinyal, %d (sembol,TF) çifti", len(rows), len(new_registry)
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                reg_logger.warning("Yenileme hatası: %s", exc)
            await asyncio.sleep(_INTERVAL)

    _RANKING_TF_WEIGHTS: Dict[str, float] = {"5m": 0.35, "15m": 0.30, "1h": 0.20, "4h": 0.15}
    _RANKING_Z_LOOKBACK = 100
    _RANKING_R_PERIOD = 14
    _RANKING_MIN_BARS = 50
    _RANKING_INTERVAL = 90

    @staticmethod
    def _ranking_vpmv(df: pd.DataFrame) -> Tuple[Optional[float], int, Optional[float]]:
        """desktop/workers/ranking_worker.py::_vpmv ile BİREBİR aynı formül —
        utils/vpmv.py::compute_components'ten FARKLI, yönsüz/evren-geneli 3.
        bir VPMV varyantı (14 Tem 2026 planı: "RankingWorker: Backend Hesaplar,
        UI Okur"). Kasıtlı olarak DEĞİŞTİRİLMEDİ, sadece taşındı."""
        try:
            rsi_series = calculate_rsi(df, period=14)
            rsi_centered = rsi_series - 50
            atr_series = calculate_atr(df, period=Config.ATR_PERIOD)
            price_pct = df["close"].pct_change().fillna(0.0) * 100.0

            vpmv_series = (
                normalize_volume_0_100(df["volume"]) * 0.35
                + normalize_momentum_0_100(rsi_centered) * 0.35
                + normalize_volatility_0_100(atr_series) * 0.20
                + normalize_price_0_100(price_pct) * 0.10
            )

            current = float(vpmv_series.iloc[-1])
            direction = 1 if float(rsi_series.iloc[-1]) >= 50 else -1

            lookback = vpmv_series.iloc[-LiveDataManager._RANKING_Z_LOOKBACK - 1 : -1]
            if len(lookback) >= 20:
                mean = float(lookback.mean())
                std = float(lookback.std())
                z = round((current - mean) / std, 2) if std > 0 else 0.0
            else:
                z = None

            return round(current, 1), direction, z
        except Exception:  # pylint: disable=broad-exception-caught
            return None, 0, None

    @staticmethod
    def _ranking_rsi_cross_score(df: pd.DataFrame) -> Optional[float]:
        """RSI9-RSI24 farkinin (Config.RSI_FAST_WINDOW/SLOW_WINDOW —
        RSI_Cross(9,24) sinyalinin kendi mantigi) normalize_momentum_0_100 ile
        0-100'e normalize edilmis hali. 18 Tem 2026: research/pattern_lab/
        rsi_cross_combined_score_bt.py ile 4 kapili dogrulandi (gercek korelasyon
        Long rho=+0.196/Short +0.215, gercek $ dogrulamasi, placebo, split-period)
        — mevcut _ranking_vpmv'nin RSI(14)>=50 esikli direction'indan (rho≈+0.02,
        gercek $'da etkisiz) cok daha guclu. Ayri, ek bir sutun olarak eklendi —
        mevcut Birlesik/TF Uyum degistirilmedi."""
        try:
            rsi_fast = calculate_rsi(df, period=Config.RSI_FAST_WINDOW)
            rsi_slow = calculate_rsi(df, period=Config.RSI_SLOW_WINDOW)
            spread = rsi_fast - rsi_slow
            score_series = normalize_momentum_0_100(spread)
            return round(float(score_series.iloc[-1]), 1)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _ranking_r_score(df: pd.DataFrame) -> Optional[float]:
        """desktop/workers/ranking_worker.py::_r_score ile BİREBİR aynı formül
        (Sharpe/Sortino/Calmar/Omega blend), sadece taşındı."""
        try:
            closes = df["close"].astype(float).values
            r_period = LiveDataManager._RANKING_R_PERIOD
            returns = np.diff(np.log(closes + 1e-12))[-r_period:]
            if len(returns) < r_period // 2:
                return None

            # Oranların paydası (std/drawdown/kayıp toplamı) gerçekten sıfıra
            # yakınsa (ör. pencerede neredeyse hiç kayıp/dalgalanma yoksa)
            # 1e-12'lik epsilon blow-up'ı ENGELLEMİYOR — pay/1e-12 hâlâ
            # milyarlarca çıkabiliyor (20 Tem 2026, ZESTUSDT R-Score=+4.58
            # milyar, masaüstü panelinde donmaya/bellek patlamasına eşlik
            # etti). Her bileşen makul bir aralığa (±50) kırpılıyor — gerçek
            # sinyal kalitesi skorları zaten bu aralıkta, blow-up olursa
            # sonucu sınırlıyor.
            _CLIP = 50.0

            avg = returns.mean()
            std = returns.std() + 1e-12
            sharpe = float(np.clip(avg / std, -_CLIP, _CLIP))

            neg = returns[returns < 0]
            neg_std = neg.std() + 1e-12 if len(neg) > 1 else 1e-12
            sortino = float(np.clip(avg / neg_std, -_CLIP, _CLIP))

            price_window = closes[-r_period - 1 :]
            max_dd = (price_window.max() - price_window.min()) / (price_window.max() + 1e-12)
            calmar = float(np.clip(avg / (max_dd + 1e-12), -_CLIP, _CLIP))

            gains = returns[returns >= 0].sum()
            losses = abs(returns[returns < 0].sum()) + 1e-12
            omega = float(np.clip(gains / losses, -_CLIP, _CLIP))

            r = sortino * 0.40 + omega * 0.30 + calmar * 0.20 + sharpe * 0.10
            return round(float(r), 3)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _ranking_compute_symbol(self, symbol: str) -> Optional[dict]:
        """Tek sembol için 4 TF'lik VPMV parçalarını + r_score'u zaten bellekteki
        `self.mtf_buffers`'tan hesaplar — Redis GET/Arrow decode YOK. Ayrı bir
        thread executor'da (`_RANKING_EXECUTOR`) çalıştırılmak üzere tasarlandı,
        pandas/numpy işlemleri senkron."""
        tf_scores: Dict[str, float] = {}
        tf_dirs: Dict[str, int] = {}
        tf_zscores: Dict[str, float] = {}
        tf_rsi_cross: Dict[str, float] = {}

        buffers = self.mtf_buffers.get(symbol) or {}
        for tf in self._RANKING_TF_WEIGHTS:
            df = buffers.get(tf)
            if df is None or df.empty or len(df) < self._RANKING_MIN_BARS:
                continue
            vpmv, direction, z = self._ranking_vpmv(df)
            if vpmv is not None:
                tf_scores[tf] = vpmv
                tf_dirs[tf] = direction
                if z is not None:
                    tf_zscores[tf] = z
            rsi_cross = self._ranking_rsi_cross_score(df)
            if rsi_cross is not None:
                tf_rsi_cross[tf] = rsi_cross

        if not tf_scores:
            return None

        total_w = sum(self._RANKING_TF_WEIGHTS[tf] for tf in tf_scores)
        combined = sum(tf_scores[tf] * self._RANKING_TF_WEIGHTS[tf] for tf in tf_scores) / total_w

        if tf_rsi_cross:
            total_rw = sum(self._RANKING_TF_WEIGHTS[tf] for tf in tf_rsi_cross)
            rsi_cross_combined = round(
                sum(tf_rsi_cross[tf] * self._RANKING_TF_WEIGHTS[tf] for tf in tf_rsi_cross)
                / total_rw,
                1,
            )
        else:
            rsi_cross_combined = None

        if tf_zscores:
            total_zw = sum(self._RANKING_TF_WEIGHTS[tf] for tf in tf_zscores)
            z_confluence = round(
                sum(tf_zscores[tf] * self._RANKING_TF_WEIGHTS[tf] for tf in tf_zscores) / total_zw,
                2,
            )
        else:
            z_confluence = None

        dirs = list(tf_dirs.values())
        n_bull = sum(1 for d in dirs if d > 0)
        n_bear = len(dirs) - n_bull
        aligned = max(n_bull, n_bear) == len(dirs)

        df_1h = buffers.get("1h")
        r_score = (
            self._ranking_r_score(df_1h)
            if df_1h is not None and not df_1h.empty and len(df_1h) >= self._RANKING_R_PERIOD
            else None
        )

        return {
            "symbol": symbol,
            "score_5m": tf_scores.get("5m"),
            "score_15m": tf_scores.get("15m"),
            "score_1h": tf_scores.get("1h"),
            "score_4h": tf_scores.get("4h"),
            "combined": round(combined, 1),
            "rsi_cross_combined": rsi_cross_combined,
            "z_confluence": z_confluence,
            "r_score": r_score,
            "aligned": aligned,
            "alignment_count": max(n_bull, n_bear),
            "tf_count": len(dirs),
            "direction": "long" if n_bull >= n_bear else "short",
        }

    async def _ranking_publish_loop(self) -> None:
        """Her ~90 saniyede bir TÜM izlenen evreni (self.mtf_buffers'ta bulunan
        semboller) VPMV skoruna göre sıralayıp `ranking:snapshot` Redis key'ine
        yazar (14 Tem 2026 planı: "RankingWorker: Backend Hesaplar, UI Okur").

        ÖNCEDEN masaüstündeki RankingWorker bunu 3 dakikada bir, 500+ sembol ×
        4 TF için Redis'ten ham kline çekip Arrow decode ederek yapıyordu —
        periyodik 30+GB'lık bellek patlamalarının (footprint ile doğrulandı)
        kaynağıydı. Burada aynı hesap `self.mtf_buffers`'tan (zaten bellekte,
        Redis round-trip yok) yapılıyor. Paylaşımlı `_MTF_EXECUTOR`'ı (gerçek
        zamanlı bar birleştirme için kullanılıyor, max_workers=4) bu ~2000+
        çağrılık sweep'le boğmamak için AYRI, küçük bir executor kullanılıyor.

        Bonus: `ranking:snapshot` ÖNCEDEN sadece masaüstü açıkken doluyordu —
        `signals/paper_trade_manager.py`/`signals/signal_lifecycle_manager.py`
        bu key'i okuyup rank_at_entry/rank_score/vs_btc yazıyor, artık backend
        her zaman taze tutuyor (UI kapalıyken de)."""
        rank_logger = logging.getLogger("RankingPublish")
        loop = asyncio.get_running_loop()

        while True:
            try:
                symbols = list(self.mtf_buffers.keys())
                scores: Dict[str, dict] = {}
                for i in range(0, len(symbols), 20):
                    batch = symbols[i : i + 20]
                    results = await asyncio.gather(
                        *(
                            loop.run_in_executor(
                                _RANKING_EXECUTOR, self._ranking_compute_symbol, sym
                            )
                            for sym in batch
                        )
                    )
                    for sym, data in zip(batch, results):
                        if data is not None:
                            scores[sym] = data
                    # asyncio.sleep(0) SADECE bir sonraki hazır coroutine'e geçiş
                    # yapar, gerçek zaman kaybettirmez — 2 executor thread'i pandas/
                    # numpy ile GIL'i sürekli tuttuğunda event loop'un Redis I/O'sunu
                    # yetiştirememesine (havuz timeout'u, "No connection available",
                    # 15 Tem sabahı canlıda yakalandı) yetmiyordu. Gerçek bir bekleme
                    # event loop'a GIL'i geri verip birikmiş I/O'yu tamamlama şansı
                    # veriyor.
                    await asyncio.sleep(0.2)

                if scores:
                    all_combined = [v["combined"] for v in scores.values()]
                    n = len(all_combined)
                    for data in scores.values():
                        data["rank_score"] = round(
                            sum(1 for v in all_combined if v < data["combined"]) / n * 100, 1
                        )

                    btc_combined = scores.get(Config.MARKET_REFERENCE_SYMBOL, {}).get("combined")
                    for data in scores.values():
                        data["vs_btc"] = (
                            round(data["combined"] - btc_combined, 1)
                            if btc_combined is not None
                            else None
                        )

                    result = sorted(scores.values(), key=lambda x: x["rank_score"], reverse=True)
                    for idx, item in enumerate(result):
                        item["rank"] = idx + 1

                    redis = RedisClient.get_client()
                    await asyncio.wait_for(
                        redis.set("ranking:snapshot", json.dumps(result), ex=600),
                        timeout=SAFE_EXTERNAL_TIMEOUT,
                    )
                    rank_logger.info("Sıralama güncellendi: %d sembol", len(result))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                rank_logger.warning("Sıralama güncelleme hatası: %s", exc, exc_info=True)
            await asyncio.sleep(self._RANKING_INTERVAL)

    _HA_ALIGNMENT_TFS = ("4h", "6h", "8h", "12h")
    _HA_ALIGNMENT_INTERVAL = 90  # ranking ile aynı kademe
    _HA_ALIGNMENT_BARS = 60  # _get_ha_alignment (tf_alignment_gate.py) ile aynı pencere
    # DB pool (database/engine.py: pool_size=20 + max_overflow=30 = 50, TÜM
    # loop'lar arasında paylaşımlı) tükenmesine karşı — 15 Tem 2026'daki
    # "sınırsız eşzamanlılık patlaması" olayının (bkz. memory:
    # project_redis_pool_exhaustion_18tem) tekrarını önlemek için bilinçli
    # olarak DÜŞÜK tutuluyor (ranking'in kullandığı ayrı thread-executor'dan
    # farklı olarak bu loop doğrudan DB I/O yapıyor).
    _HA_ALIGNMENT_SEMAPHORE = asyncio.Semaphore(10)

    async def _ha_alignment_fetch_one(
        self, symbol: str, tf: str
    ) -> Tuple[str, str, Optional[bool]]:
        async with self._HA_ALIGNMENT_SEMAPHORE:
            try:
                df = await get_cagg_klines(symbol, tf, self._HA_ALIGNMENT_BARS, closed_only=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("[HAAlignment] %s %s çekilemedi: %s", symbol, tf, exc)
                return symbol, tf, None
        return symbol, tf, _heikin_ashi_bull(df)

    async def _ha_alignment_publish_loop(self) -> None:
        """Her ~90 saniyede TÜM izlenen sembol evreni (self.mtf_buffers'taki
        semboller, ranking:snapshot ile aynı kaynak) için 4h/6h/8h/12h
        Heikin-Ashi rengini hesaplayıp `ha_alignment:snapshot` Redis
        key'ine yazar (27 Tem 2026, kullanıcı isteği — HTF hizalanan
        coinlerde sinyal açma planının izleme altyapısı).

        `_heikin_ashi_bull` (signals/tf_alignment_gate.py) BİREBİR aynı
        fonksiyon — closed_only=True disiplinli, 27 Tem'de bulunan
        look-ahead hatasının (research/pattern_lab/rsi_cross_ta_percentile_bt.py
        ::searchsorted(...,'right')-1, kapanmamış barı kapanmış sayma) AYNI
        deseni burada TEKRARLANMASIN diye ayrı bir hesap YAZILMADI, mevcut
        (doğru) fonksiyon içe aktarıldı."""
        ha_logger = logging.getLogger("HAAlignmentPublish")

        while True:
            try:
                symbols = list(self.mtf_buffers.keys())
                tasks = [
                    self._ha_alignment_fetch_one(sym, tf)
                    for sym in symbols
                    for tf in self._HA_ALIGNMENT_TFS
                ]
                results = await asyncio.gather(*tasks)

                per_symbol: Dict[str, Dict[str, Optional[bool]]] = {}
                for sym, tf, bull in results:
                    per_symbol.setdefault(sym, {})[tf] = bull

                snapshot = []
                for sym, bulls in per_symbol.items():
                    if any(bulls.get(tf) is None for tf in self._HA_ALIGNMENT_TFS):
                        continue
                    bull_count = sum(1 for tf in self._HA_ALIGNMENT_TFS if bulls[tf])
                    snapshot.append(
                        {
                            "symbol": sym,
                            "ha_4h": bulls["4h"],
                            "ha_6h": bulls["6h"],
                            "ha_8h": bulls["8h"],
                            "ha_12h": bulls["12h"],
                            "bull_count": bull_count,
                            "aligned_bull": bull_count == len(self._HA_ALIGNMENT_TFS),
                            "aligned_bear": bull_count == 0,
                        }
                    )

                if snapshot:
                    redis = RedisClient.get_client()
                    await asyncio.wait_for(
                        redis.set("ha_alignment:snapshot", json.dumps(snapshot), ex=600),
                        timeout=SAFE_EXTERNAL_TIMEOUT,
                    )
                    ha_logger.info("HA hizalanma güncellendi: %d sembol", len(snapshot))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                ha_logger.warning("HA hizalanma güncelleme hatası: %s", exc, exc_info=True)
            await asyncio.sleep(self._HA_ALIGNMENT_INTERVAL)

    _VPMV_TTL_BY_INTERVAL = {"1m": 180, "5m": 900, "15m": 2700, "1h": 10800, "4h": 43200}

    async def _publish_vpmv_live(
        self, symbol: str, interval: str, df: pd.DataFrame, active_signals: list
    ) -> None:
        """(symbol, interval) için aktif sinyal(ler)in VPMV'sini zaten bellekte
        olan buffer'dan hesaplayıp Redis'e yazar — masaüstü VpmvWorker'ın aynı
        hesabı ham kline'ları yeniden çekip tekrar yapmasını gereksiz kılar
        (14 Tem 2026 planı: "VPMV: Backend Hesaplar, UI Okur"). Gölge-mod
        doğrulaması tamamlandı, vpmv_worker.py artık SADECE bu değeri okuyor,
        eski ağır hesap yolu silindi."""
        vpm_weights = Config.VPM.get("WEIGHTS")
        ttl = self._VPMV_TTL_BY_INTERVAL.get(interval, 900)
        redis = RedisClient.get_client()

        for sig in active_signals:
            sig_id = sig["id"]
            sig_type = sig["signal_type"]
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(df, sig_type)
                vpmv = VPMCalculator.calculate(
                    vol_score=vol_s,
                    momentum_score=mom_s,
                    vlt_score=vlt_s,
                    price_score=prc_s,
                    weights=vpm_weights,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("[VPMVLive] %s %s hesaplanamadı: %s", symbol, interval, exc)
                continue

            recent = self._vpmv_recent.setdefault(sig_id, deque(maxlen=20))
            recent.append(round(float(vpmv), 2))

            payload = {
                "vpmv": round(float(vpmv), 2),
                "vol": round(float(vol_s), 2),
                "mom": round(float(mom_s), 2),
                "vlt": round(float(vlt_s), 2),
                "price": round(float(prc_s), 2),
                "ts": int(time.time()),
                "recent": list(recent),
            }
            try:
                await asyncio.wait_for(
                    redis.set(f"vpmv_live:{sig_id}", json.dumps(payload), ex=ttl),
                    timeout=SAFE_EXTERNAL_TIMEOUT,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "[VPMVLive] %s (id=%s) Redis yazımı başarısız: %s", symbol, sig_id, exc
                )

    _DEVISSO_TTL_BY_INTERVAL = {"1m": 180, "5m": 900, "15m": 2700, "1h": 10800, "4h": 43200}

    async def _publish_devisso_live(
        self, symbol: str, interval: str, df: pd.DataFrame, active_signals: list
    ) -> None:
        """(symbol, interval) için aktif sinyal(ler)in Verimlilik'ini (ERSI,
        signal_processor.py::_compute_devisso_score ile BİREBİR aynı fonksiyon)
        zaten bellekteki buffer'dan hesaplayıp `devisso_live:{signal_id}`
        Redis key'ine yazar — _publish_vpmv_live ile AYNI desen (27 Tem 2026,
        kullanıcı isteği). Yön'den bağımsız bir metrik (RSI/fiyat verimliliği),
        bu yüzden VPMV'nin aksine sig_type'a göre değişmiyor — (symbol,
        interval) başına bir kez hesaplanıp aynı (symbol, interval)'daki tüm
        aktif sinyallere yayınlanıyor.

        Panel tarafı "sinyalden beri" deltasını (bu değer − DB'deki
        devisso_score, sinyal açılış anı) kendisi hesaplayacak — VPMV'de
        de aynı disiplin (vpmv_pre_avg zaten sinyal-öncesi statik snapshot,
        canlı delta client-side türetiliyor)."""
        ttl = self._DEVISSO_TTL_BY_INTERVAL.get(interval, 900)
        redis = RedisClient.get_client()

        try:
            live_score = _compute_devisso_score(df)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "[DevissoLive] %s %s hesaplanamadı: %s", symbol, interval, exc, exc_info=True
            )
            return
        if live_score is None:
            return

        payload = {"devisso": live_score, "ts": int(time.time())}
        for sig in active_signals:
            sig_id = sig["id"]
            try:
                await asyncio.wait_for(
                    redis.set(f"devisso_live:{sig_id}", json.dumps(payload), ex=ttl),
                    timeout=SAFE_EXTERNAL_TIMEOUT,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "[DevissoLive] %s (id=%s) Redis yazımı başarısız: %s", symbol, sig_id, exc
                )

    _DIVERGENCE_EMA_PERIOD = 200  # desktop/workers/divergence_worker.py::_EMA_PERIOD ile aynı
    _DIVERGENCE_SERIES_LEN = 100  # desktop/workers/divergence_worker.py::_SERIES_LEN ile aynı
    _DIVERGENCE_TTL_BY_INTERVAL = {"1m": 180, "5m": 900, "15m": 2700, "1h": 10800, "4h": 43200}

    async def _publish_divergence_live(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """(symbol, interval) için HAM (offset uygulanmamış) Z-score serisini
        zaten bellekteki buffer'dan hesaplayıp Redis'e yazar —
        `desktop/workers/divergence_worker.py::_zscore_series` ile BİREBİR aynı
        formül (EMA/rolling-std). Ters-sinyal offset'i (`_offsets`) DESKTOP
        tarafında kalıyor — bu, o mantığın ihtiyaç duymadığı, gerçekten pahalı
        olan kısmı (ham kline çekme + hesaplama) ortadan kaldırıyor."""
        ttl = self._DIVERGENCE_TTL_BY_INTERVAL.get(interval, 900)
        redis = RedisClient.get_client()
        try:
            close = df["close"].astype(float).reset_index(drop=True)
            n = min(self._DIVERGENCE_EMA_PERIOD, len(close))
            ema = close.ewm(span=n, adjust=False).mean()
            std = close.rolling(n, min_periods=5).std().bfill()
            z = (close - ema) / (std + 1e-8)
            z_tail = z.tail(self._DIVERGENCE_SERIES_LEN)

            if "open_time" in df.columns:
                ts_tail = (df["open_time"].tail(len(z_tail)) / 1000.0).tolist()
            else:
                ts_tail = []
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[DivergenceLive] %s %s hesaplanamadı: %s", symbol, interval, exc)
            return

        payload = {
            "z_now": round(float(z_tail.iloc[-1]), 4),
            "recent": [round(float(v), 4) for v in z_tail.tolist()],
            "ts_recent": ts_tail,
            "ts": int(time.time()),
        }
        try:
            await asyncio.wait_for(
                redis.set(f"divergence_live:{symbol}:{interval}", json.dumps(payload), ex=ttl),
                timeout=SAFE_EXTERNAL_TIMEOUT,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[DivergenceLive] %s %s Redis yazımı başarısız: %s", symbol, interval, exc)

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
                        Config.REDIS_URL,
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                    )
                await asyncio.wait_for(
                    redis_conn.set("prices:live", json.dumps(prices), ex=15), timeout=5
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "[PricePublish] prices:live yazılamadı, bağlantı yenilenecek: %s", exc
                )
                try:
                    if redis_conn is not None:
                        await redis_conn.aclose()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                redis_conn = None

    _TRADE_XRAY_INTERVAL = 15  # sn — trade_xray_panel.py'nin eski DB-poll periyoduyla aynı

    async def _trade_xray_publish_loop(self) -> None:
        """desktop/panels/trade_xray_panel.py'nin İşlem Listesi tablosu için
        (14 Ağu 2026 mimari denetimi): panel önceden HER 15 saniyede bir kendi
        DB bağlantısıyla doğrudan sorgu atıyordu (backend-hesaplar-UI-okur
        mimarisine uymuyordu — diğer tüm worker'lar zaten bu deseni kullanıyor,
        bkz. _ranking_publish_loop/_ha_alignment_publish_loop). Aynı sorgu
        burada TEK yerden çalışıp `trade_xray:trades` Redis key'ine yazılıyor,
        panel artık sadece okuyor. Tıklanan işlemin detay grafiği (trade_snapshots
        zaman serisi) hâlâ panelin kendi DB sorgusunda kalıyor — o talebe bağlı/
        tek-trade'e sınırlı, periyodik-geniş-tarama sınıfına girmiyor."""
        xray_logger = logging.getLogger("TradeXRayPublish")
        while True:
            try:

                async def _fetch_trades() -> list:
                    async with get_session() as session:
                        result = await session.execute(
                            text(
                                """
                                SELECT p.id, p.symbol, p.strategy, p.signal_type, p.status,
                                       p.opened_at,
                                       COALESCE(p.pnl_pct, latest.price_since_entry_pct) AS pnl_pct,
                                       p.entry_features
                                FROM paper_trades p
                                LEFT JOIN LATERAL (
                                    SELECT price_since_entry_pct FROM trade_snapshots ts
                                    WHERE ts.trade_id = p.id ORDER BY taken_at DESC LIMIT 1
                                ) latest ON true
                                ORDER BY p.opened_at DESC
                                LIMIT 500
                                """
                            )
                        )
                        return result.all()

                rows = await run_with_db_timeout(_fetch_trades())
                trades = []
                for r in rows:
                    features = r.entry_features
                    if features is not None and not isinstance(features, dict):
                        try:
                            features = json.loads(features)
                        except (TypeError, ValueError):
                            features = None
                    trades.append(
                        {
                            "id": r.id,
                            "symbol": r.symbol,
                            "strategy": r.strategy,
                            "signal_type": r.signal_type,
                            "status": r.status,
                            "opened_at_str": (
                                r.opened_at.strftime("%d/%m %H:%M") if r.opened_at else None
                            ),
                            "pnl_pct": float(r.pnl_pct) if r.pnl_pct is not None else None,
                            "entry_features": features,
                        }
                    )

                redis_conn = RedisClient.get_client()
                await asyncio.wait_for(
                    redis_conn.set(
                        "trade_xray:trades",
                        json.dumps(trades),
                        ex=self._TRADE_XRAY_INTERVAL * 3,
                    ),
                    timeout=SAFE_EXTERNAL_TIMEOUT,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                xray_logger.warning("Yayın hatası: %s", exc, exc_info=True)
            await asyncio.sleep(self._TRADE_XRAY_INTERVAL)

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
                result = await session.execute(sa_select(Signal).where(Signal.status == "active"))
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
                        symbol,
                        interval,
                        sig.signal_type,
                        last_direction,
                    )
                    closed += 1

        recon_log.info("Reconciliation tamamlandı: %d sinyal kapatıldı.", closed)

    async def _background_startup(self) -> None:
        """WebSocket başladıktan sonra gap fill → MTF init → sinyal reconciliation sırayla çalışır.

        Sıralı çalışma zorunlu: MTF init gap fill bitmeden DB'den okursa kirli buffer yüklenir.
        WS zaten ayakta olduğu için sıralama chart'a herhangi bir gap yaratmaz.

        Bitişte (başarılı/başarısız fark etmeksizin) _startup_complete_event set edilir —
        _continuous_gap_heal_loop bu sinyali bekleyip ilk turunu hemen ardından başlatır
        (eski _post_init_catchup'ın kapsadığı startup-sonrası pencereyi artık bu kapsıyor,
        bkz. o metodun docstring'i, 12 Tem 2026)."""
        try:
            logger.info("[BackgroundStartup] Başladı (WebSocket zaten aktif).")
            filled_symbols = await self._startup_gap_fill()
            if filled_symbols:
                gap_starts = {
                    sym: self._gap_start_ms[sym]
                    for sym in filled_symbols
                    if sym in self._gap_start_ms
                }
                await self._replay_filter_state_for_gaps(gap_starts, source="BackgroundStartup")
            if self.mtf_enabled:
                await self._initialize_mtf_dataframes(reload_symbols=filled_symbols)
            await self._startup_signal_reconciliation()
            logger.info("[BackgroundStartup] Tamamlandı.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[BackgroundStartup] Hata: %s", exc, exc_info=True)
        finally:
            self._startup_complete_event.set()

    async def run(self):
        """Ana çalıştırma döngüsü."""
        try:
            # 1. WebSocket önce başlat — canlı veri hemen akar, gap oluşmaz
            await self.start_streams()
            # 2. MTF init + gap fill arka planda (WS'yi bloklamaz)
            asyncio.create_task(self._background_startup())
            # 3. 30s sonra kuyruk gap fill (son timestamp → şu an)
            asyncio.create_task(self._deferred_sync_historical(delay_seconds=30))
            # Python 3.10+ weak ref fix: periyodik loop task'larını sakla
            # (12 Tem 2026 konsolidasyonu: _deferred_internal_gap_check/_post_init_catchup/
            # _health_loop kaldırıldı — hepsi _continuous_gap_heal_loop'a birleşti, bkz. o
            # metodun docstring'i)
            self._bg_tasks: list[asyncio.Task] = [
                asyncio.create_task(self._continuous_gap_heal_loop()),
                asyncio.create_task(self._mtf_resync_loop()),
                asyncio.create_task(self._ticker_refresh_loop()),
                asyncio.create_task(self._oi_refresh_loop()),
                asyncio.create_task(self._price_publish_loop()),
                asyncio.create_task(self._vpmv_post_loop()),
                asyncio.create_task(self._active_signal_registry_loop()),
                asyncio.create_task(self._ranking_publish_loop()),
                asyncio.create_task(self._ha_alignment_publish_loop()),
                asyncio.create_task(self._trade_xray_publish_loop()),
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
                    and (self.loop.time() - self.last_message_time) > Config.WEBSOCKET_TIMEOUT
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
                    delay = min(max_delay, base_delay * (2 ** min(self.reconnect_attempt, 6)))
                    # Basit jitter: +/- 20%
                    jitter = max(0.5, delay * 0.2)
                    import random

                    sleep_for = max(1.0, delay + random.uniform(-jitter, jitter))
                    logger.warning(
                        f"{reconnect_reason} {sleep_for:.1f} saniye içinde yeniden bağlanma denenecek... (attempt={self.reconnect_attempt})"
                    )
                    self.is_ws_connected = False  # Yeniden bağlanma sürecini başlatmak için
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

                            sleep_for = max(1.0, delay + random.uniform(-jitter, jitter))
                            logger.info(f"{sleep_for:.1f} saniye sonra tekrar denenecek.")
                            await asyncio.sleep(sleep_for)
                else:
                    # Bağlantı sağlamsa, döngüyü tıkamadan bekle
                    # Ping/Pong watchdog: belirli aralıkla heartbeat kontrolü
                    await asyncio.sleep(getattr(Config, "WS_HEARTBEAT_CHECK_INTERVAL", 5))
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
                if (
                    ws_client
                    and hasattr(ws_client, "stop")
                    and not asyncio.iscoroutinefunction(ws_client.stop)
                ):
                    try:
                        # Timeout ile güvenli kapatma
                        await asyncio.wait_for(asyncio.to_thread(ws_client.stop), timeout=5.0)
                        logger.debug(
                            f"WebSocket connection #{connection_id} güvenli şekilde kapatıldı."
                        )
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
    await rsi_cross_live_manager.load_open_symbols()

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
