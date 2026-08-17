import asyncio
import functools
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import redis.asyncio as redis
import redis.exceptions as redis_exceptions

from config import Config

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
# 8→4 (10 Tem 2026): _MTF_EXECUTOR ile aynı gerekçe — dedup+serialize artık hafif,
# 8 thread gereksiz GIL çekişmesi yaratıyordu (bkz. live_data_manager.py yorumu).
_ARROW_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arrow")

# get_fresh_klines'ın add_all_indicators çağrısı için AYRI havuz (17 Ağu 2026)
# — _ARROW_EXECUTOR'ı (yukarıdaki not: bilerek küçük tutuluyor, Arrow
# serialize hafif bir iş) paylaşmak canlıda kanıtlandı ki YANLIŞ: bar-kapanışı
# burst'ünde (550 sembol aynı anda) add_all_indicators (~18ms, gerçekten
# CPU-yoğun) bu 4 worker'lı küçük havuzu doldurup kuyruk oluşturuyor, bazı
# sinyal kontrolleri 1-2 saniyeye kadar gecikiyordu.
_INDICATOR_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mtf_indicator")

# Ana pool'un (BlockingConnectionPool) bağlantı-edinme timeout'u. Dıştan bu pool'a
# karşı asyncio.wait_for ile sarılan HER çağrı, SAFE_EXTERNAL_TIMEOUT'tan kısa bir
# süre kullanmamalı — aksi halde pool kendi temiz ConnectionError'ını fırlatamadan
# dıştan iptal edilir, bu da bağlantının yarım kalmış edinme anında sızmasına yol
# açabilir (7 Tem heartbeat donması kök neden analizinde tespit edilen risk).
POOL_ACQUIRE_TIMEOUT = 10
SAFE_EXTERNAL_TIMEOUT = POOL_ACQUIRE_TIMEOUT + 1


class RedisClient:
    """Asenkron Redis istemcisi için merkezi bir yönetici sınıfı.

    Her process/event loop için ayrı bir bağlantı havuzu yönetir.
    set_mtf_klines çağrıları pending dict'e birikir; _batch_flush_loop
    her 500ms'de tek pipeline ile Redis'e iter — burst sorunu ortadan kalkar.
    """

    _pools: Dict[int, redis.ConnectionPool] = {}
    _binary_pools: Dict[int, redis.ConnectionPool] = {}
    _write_semaphores: Dict[int, asyncio.Semaphore] = {}
    _read_semaphores: Dict[int, asyncio.Semaphore] = {}

    # Batch flush state — asyncio single-threaded, race condition yok
    _pending_klines: Dict[str, Tuple[bytes, int]] = {}  # key → (arrow_bytes, ttl)
    _flush_immediately: bool = False  # test dikişi: True ise set anında flush eder
    _pending_publishes: Set[str] = set()  # "symbol:tf"
    _pending_kline_closed: List[Dict[str, Any]] = []  # XADD kline_closed için birikmiş event'ler
    _flusher_task: Optional[asyncio.Task] = None

    # get_mtf_klines kendi kendini onarma state'i (16 Ağu 2026) — bkz. o metodun
    # docstring'i: bayat/eksik bir buffer görülünce arka planda CA'dan tazeler.
    _mtf_refresh_inflight: Set[str] = set()
    _mtf_refresh_last_attempt: Dict[str, float] = {}
    _MTF_REFRESH_COOLDOWN_SEC = 60

    # get_fresh_klines'ın doğrudan Postgres/CA'dan okuyabildiği TF'ler (17 Ağu
    # 2026) — bkz. o metodun docstring'i. 1m/30m/1d bu kümede yok: 1m/30m CA
    # karşılığı yok veya WS zaten sürekli taze tutuyor, 1d için de CA yok.
    _CAGG_TIMEFRAMES = {"5m", "15m", "1h", "4h", "6h", "8h", "12h"}

    @classmethod
    def _get_write_semaphore(cls) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        loop_id = id(loop)
        if loop_id not in cls._write_semaphores:
            cls._write_semaphores[loop_id] = asyncio.Semaphore(30)
        return cls._write_semaphores[loop_id]

    @classmethod
    def _get_read_semaphore(cls) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        loop_id = id(loop)
        if loop_id not in cls._read_semaphores:
            cls._read_semaphores[loop_id] = asyncio.Semaphore(50)
        return cls._read_semaphores[loop_id]

    @classmethod
    def _get_pool_for_current_loop(cls) -> redis.ConnectionPool:
        """Mevcut asyncio event loop için bir bağlantı havuzu oluşturur veya döndürür."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        loop_id = id(loop)

        if loop_id not in cls._pools:
            logger.info(f"Yeni Redis bağlantı havuzu oluşturuluyor (Loop ID: {loop_id})")
            cls._pools[loop_id] = redis.BlockingConnectionPool.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                max_connections=300,
                timeout=POOL_ACQUIRE_TIMEOUT,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10,  # komut cevabı için üst sınır — yoksa donmuş
                # bağlantı havuza sonsuza dek geri dönmez
                health_check_interval=30,  # boşta bağlantıları periyodik PING ile denetler
                retry_on_timeout=True,
            )
        return cls._pools[loop_id]

    @classmethod
    def _get_binary_pool_for_current_loop(cls) -> redis.ConnectionPool:
        """Binary (bytes) veri için decode_responses=False havuzu."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        loop_id = id(loop)

        if loop_id not in cls._binary_pools:
            cls._binary_pools[loop_id] = redis.BlockingConnectionPool.from_url(
                Config.REDIS_URL,
                decode_responses=False,
                max_connections=300,
                timeout=30,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10,  # ana pool ile aynı gerekçe — donmuş bağlantı
                # sonsuza dek havuzu kilitlemesin
                health_check_interval=30,
                retry_on_timeout=True,  # ana poolda vardı, burada eksikti — tutarlılık
            )
        return cls._binary_pools[loop_id]

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Mevcut event loop'a uygun bağlantı havuzundan bir istemci döndürür."""
        pool = cls._get_pool_for_current_loop()
        return redis.Redis(connection_pool=pool)

    # 10 Tem 2026 akşam: process-bağımsız paylaşımlı rate limiter. Kök neden:
    # run_services.py, signal_service.py VE masaüstü panel (desktop.main —
    # market_worker/divergence_worker/ranking_worker/vpmv_worker) AYNI IP'den
    # bağımsız olarak Binance'e istek atıyor; her process kendi içinde ne kadar
    # dikkatli throttle uygularsa uygulasın, TOPLAM istek hızı yine gerçek IP
    # limitini (2400/dk) aşabiliyordu (canlıda 429 alındı, iki kez). Tek process
    # içi bir sayaç (liste/lock) bunu çözemez — sayaç her process'te ayrı hafızada.
    # Redis zaten TÜM process'ler tarafından paylaşılan tek durak, bu yüzden
    # sliding-window sayacı burada, ZSET + Lua (atomik, tek round-trip) ile tutuluyor.
    _RATE_LIMIT_LUA = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local member = ARGV[4]
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        local count = redis.call('ZCARD', key)
        if count < limit then
            redis.call('ZADD', key, now, member)
            redis.call('PEXPIRE', key, window + 1000)
            return 0
        else
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            return (tonumber(oldest[2]) + window) - now
        end
    """
    _rate_limit_sha: Dict[str, str] = {}

    @classmethod
    async def throttle_external_api(cls, bucket: str, max_per_min: int) -> None:
        """Verilen bucket (ör. 'binance') için process-bağımsız sliding-window
        rate limit uygular — limit dolmuşsa gereken süre kadar asyncio.sleep ile
        bekler, sonra tekrar dener. Her çağıran (hangi process'te olursa olsun)
        AYNI Redis anahtarını paylaştığı için toplam istek hızı asla aşılmaz."""
        key = f"rate_limit:{bucket}"
        window_ms = 60_000
        client = cls.get_client()
        while True:
            now_ms = int(time.time() * 1000)
            member = f"{now_ms}-{uuid.uuid4().hex[:8]}"
            try:
                sha = cls._rate_limit_sha.get("script")
                if sha is None:
                    sha = await client.script_load(cls._RATE_LIMIT_LUA)
                    cls._rate_limit_sha["script"] = sha
                try:
                    wait_ms = await client.evalsha(
                        sha, 1, key, now_ms, window_ms, max_per_min, member
                    )
                except redis_exceptions.NoScriptError:
                    sha = await client.script_load(cls._RATE_LIMIT_LUA)
                    cls._rate_limit_sha["script"] = sha
                    wait_ms = await client.evalsha(
                        sha, 1, key, now_ms, window_ms, max_per_min, member
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Redis'e ulaşılamıyorsa güvenli tarafta kal ama tüm sistemi kilitleme —
                # kısa bir bekleme ile devam et (rate limit koruması geçici devre dışı).
                logger.warning("[RateLimit] %s throttle hatası, atlanıyor: %s", bucket, exc)
                return
            if wait_ms <= 0:
                return
            await asyncio.sleep(min(wait_ms / 1000 + 0.02, 5))

    @classmethod
    def _get_binary_client(cls) -> redis.Redis:
        """Binary veri için bağlantı havuzundan istemci döndürür."""
        pool = cls._get_binary_pool_for_current_loop()
        return redis.Redis(connection_pool=pool)

    @staticmethod
    def _dedupe_and_to_arrow_bytes(df: pd.DataFrame, already_owned: bool = False) -> bytes:
        """already_owned=True: çağıran bu df'nin TEK sahibi olduğunu (başka hiçbir
        yerde tutulmadığını/eşzamanlı mutasyona uğramayacağını) garanti ediyorsa
        ikinci kopyayı atlar (30 Tem 2026 CPU profili — forming-tick yolu, tek
        güvenli çağıran: live_data_manager.py::_handle_tick, _merge_tick_row'un
        ZATEN kopyaladığı bir sonucu iletiyor). Varsayılan False — set_mtf_klines'ın
        diğer TÜM çağıranları self.mtf_buffers[...]'ı DOĞRUDAN iletiyor (paylaşılan,
        canlı buffer) — onlarda kopya ZORUNLU, aksi halde executor thread'i Arrow'a
        çevirirken event loop aynı objeyi eşzamanlı mutasyona uğratabilir (yarış durumu)."""
        if "open_time" in df.columns and df["open_time"].duplicated().any():
            df = df.drop_duplicates(subset=["open_time"], keep="last")
        elif not already_owned:
            df = df.copy()
        return RedisClient._to_arrow_bytes_raw(df)

    @staticmethod
    def _to_arrow_bytes_raw(df: pd.DataFrame) -> bytes:
        """df ÇAĞIRANA AİT (tek-sahipli) olmalı — yerinde mutasyon yapar, kopyalamaz."""
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
        table = pa.Table.from_pandas(df, preserve_index=False)
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()
        return _ARROW_MAGIC + sink.getvalue().to_pybytes()

    @staticmethod
    def _df_to_arrow_bytes(df: pd.DataFrame) -> bytes:
        return RedisClient._to_arrow_bytes_raw(df.copy())

    @staticmethod
    def _arrow_bytes_to_df(data: bytes) -> pd.DataFrame:
        reader = pa.ipc.open_stream(data[len(_ARROW_MAGIC) :])
        return reader.read_pandas()

    @classmethod
    async def set_df(cls, key: str, df: pd.DataFrame, ex: int = 60 * 60 * 24) -> None:
        """Bir Pandas DataFrame'i Arrow IPC formatında Redis'e yazar."""
        try:
            loop = asyncio.get_running_loop()
            arrow_bytes = await loop.run_in_executor(_ARROW_EXECUTOR, cls._df_to_arrow_bytes, df)
            sem = cls._get_write_semaphore()
            async with sem:
                r = cls._get_binary_client()
                await r.set(key, arrow_bytes, ex=ex)
                logger.debug("DataFrame Redis'e yazıldı (Arrow). Anahtar: %s", key)
        except Exception as e:
            logger.error("Redis'e DataFrame yazma hatası (Anahtar: %s): %s", key, e)

    @classmethod
    async def get_hot_klines(cls, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Hot cache'den son N kline'ı getirir."""
        cache_key = f"hot_klines:{symbol}"
        df = await cls.get_df(cache_key)

        if df is not None and not df.empty:
            # Son N satırı döndür
            return df.tail(limit)
        return None

    @classmethod
    async def set_hot_klines(cls, symbol: str, df: pd.DataFrame) -> None:
        """Hot cache'e kline verilerini yazar (son 1000 satır)."""
        cache_key = f"hot_klines:{symbol}"
        # Sadece son 1000 satırı cache'le (memory tasarrufu)
        hot_df = df.tail(1000) if len(df) > 1000 else df
        await cls.set_df(cache_key, hot_df, ex=3600)  # 1 saat

    @classmethod
    async def get_df(cls, key: str) -> Optional[pd.DataFrame]:
        """Redis'ten bir DataFrame'i okur (Arrow IPC veya JSON fallback)."""
        sem = cls._get_read_semaphore()
        async with sem:
            return await cls._get_df_inner(key)

    @classmethod
    async def _get_df_inner(cls, key: str) -> Optional[pd.DataFrame]:
        r = cls._get_binary_client()
        try:
            data = await r.get(key)
            if not data:
                return None
            logger.debug(f"DataFrame Redis'ten okundu. Anahtar: {key}")
            loop = asyncio.get_running_loop()
            if isinstance(data, bytes) and data.startswith(_ARROW_MAGIC):
                df = await loop.run_in_executor(_ARROW_EXECUTOR, cls._arrow_bytes_to_df, data)
            else:
                text = data.decode("utf-8") if isinstance(data, bytes) else data
                df = await loop.run_in_executor(
                    _ARROW_EXECUTOR, functools.partial(pd.read_json, StringIO(text), orient="split")
                )
                if "open_time" in df.columns and pd.api.types.is_integer_dtype(df["open_time"]):
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Redis'ten DataFrame okuma hatası (Anahtar: {key}): {e}")
            return None

    @classmethod
    async def set_json(cls, key: str, data: Any, ex: Optional[int] = None) -> bool:
        """JSON verisini Redis'e yaz."""
        try:
            client = cls.get_client()
            json_data = json.dumps(data, ensure_ascii=False, default=str)
            await client.set(key, json_data, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis'e JSON yazma hatası (Anahtar: {key}): {e}")
            return False

    @classmethod
    async def get_json(cls, key: str) -> Optional[Any]:
        """Redis'ten JSON verisini okur ve Python nesnesine çevirir."""
        r = cls.get_client()
        try:
            data = await r.get(key)
            if data:
                logger.debug(f"JSON veri Redis'ten okundu. Anahtar: {key}")
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis'ten JSON okuma hatası (Anahtar: {key}): {e}")
            return None
        # Pool otomatik yönetir, close() gerekmez

    # =============================================================================
    # MULTI-TIMEFRAME CACHE FUNCTIONS
    # =============================================================================

    @classmethod
    def _get_mtf_key(cls, symbol: str, timeframe: str, data_type: str = "klines") -> str:
        """
        Multi-timeframe cache key oluşturur.

        Args:
            symbol: Sembol (örn: 'BTCUSDT')
            timeframe: Zaman dilimi (örn: '5m', '15m')
            data_type: Veri türü ('klines', 'indicators', 'signals')

        Returns:
            str: Redis cache key
        """
        return f"{data_type}:{symbol}:{timeframe}"

    @classmethod
    def _get_ttl_for_timeframe(cls, timeframe: str) -> int:
        """
        Timeframe'e göre TTL (Time To Live) değeri döndürür.

        Args:
            timeframe: Zaman dilimi

        Returns:
            int: TTL saniye cinsinden
        """
        ttl_map = {
            "1m": 86400,  # 24 saat
            "5m": 259200,  # 3 gün
            "15m": 604800,  # 7 gün
            "1h": 604800,  # 7 gün
            "4h": 1209600,  # 14 gün
            "1d": 2592000,  # 30 gün
        }
        return ttl_map.get(timeframe, 3600)  # Default: 1 saat

    @classmethod
    async def set_mtf_klines(
        cls, symbol: str, timeframe: str, df: pd.DataFrame, already_owned: bool = False
    ) -> bool:
        """
        MTF kline'ı pending dict'e ekler — batch flusher Redis'e iter.
        Çağrı başına bağlantı yok, semaphore yok, burst sorunu yok.

        already_owned=True SADECE df'nin bu çağrıdan önce taze üretilmiş
        (self.mtf_buffers[...] gibi paylaşılan bir buffer'a takılı OLMAYAN) özel
        bir kopya olduğu kesinse kullanılmalı — bkz. _dedupe_and_to_arrow_bytes.
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "live_kline_data")
            ttl = cls._get_ttl_for_timeframe(timeframe)
            loop = asyncio.get_running_loop()
            arrow_bytes = await loop.run_in_executor(
                _ARROW_EXECUTOR, cls._dedupe_and_to_arrow_bytes, df, already_owned
            )
            cls._pending_klines[key] = (arrow_bytes, ttl)
            cls._pending_publishes.add(f"{symbol}:{timeframe}")
            if cls._flush_immediately:
                await cls.flush_pending_klines()
            else:
                cls._ensure_flusher()
            return True
        except Exception as e:
            logger.error("MTF klines batch hatası [%s:%s]: %s", symbol, timeframe, e)
            return False

    @classmethod
    def _ensure_flusher(cls) -> None:
        """Batch flush goroutine çalışmıyorsa başlatır."""
        if cls._flusher_task is None or cls._flusher_task.done():
            cls._flusher_task = asyncio.create_task(
                cls._batch_flush_loop(), name="redis_batch_flusher"
            )

    @classmethod
    async def flush_pending_klines(cls) -> int:
        """Pending kline'ları tek pipeline ile Redis'e iter. Yazılan SET sayısını döndürür."""
        if not cls._pending_klines and not cls._pending_kline_closed:
            return 0
        # Swap: yeni yazmaları boş dict/list'e yönlendir, eskiyi flush et
        pending = cls._pending_klines
        publishes = cls._pending_publishes
        closed_events = cls._pending_kline_closed
        cls._pending_klines = {}
        cls._pending_publishes = set()
        cls._pending_kline_closed = []
        try:
            r = cls._get_binary_client()
            async with r.pipeline(transaction=False) as pipe:
                for k, (data, ttl) in pending.items():
                    pipe.set(k, data, ex=ttl)
                if publishes:
                    pipe.publish("kline_updated", ",".join(publishes))
                for event in closed_events:
                    pipe.xadd("kline_closed", event, maxlen=100_000, approximate=True)
                await pipe.execute()
            logger.debug(
                "Batch flush: %d SET, %d PUBLISH, %d XADD",
                len(pending),
                len(publishes),
                len(closed_events),
            )
            return len(pending)
        except Exception as e:
            logger.error("Batch flush hatası: %s", e)
            return 0

    @classmethod
    async def _batch_flush_loop(cls) -> None:
        """Her 500ms'de pending kline'ları Redis'e iter."""
        from utils.heartbeat import beat  # döngüsel import'u önlemek için burada

        logger.info("Redis batch flusher başlatıldı (500ms)")
        while True:
            try:
                await asyncio.sleep(0.5)
                await cls.flush_pending_klines()
                await beat("redis_batch_flush")
            except asyncio.CancelledError:
                logger.warning("Redis batch flusher iptal edildi")
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Redis batch flusher döngü hatası: %s", e, exc_info=True)

    @classmethod
    async def publish_kline_update(cls, symbol: str, timeframe: str) -> None:
        """Kline güncellendi bildirimini pub/sub kanalına gönderir."""
        try:
            client = cls.get_client()
            await client.publish("kline_updated", f"{symbol}:{timeframe}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("kline_updated publish hatası: %s", e)

    @classmethod
    async def publish_kline_closed_event(cls, symbol: str, timeframe: str, open_time: int) -> None:
        """
        Bar kapanışını kline_closed stream'ine yazmak üzere pending listeye ekler —
        Redis'e gerçek yazım _batch_flush_loop'ta tek pipeline ile olur (burst sorunu
        yaşamamak için set_mtf_klines ile aynı desen). Faz 1: henüz tüketici yok.
        """
        cls._pending_kline_closed.append(
            {"v": 1, "symbol": symbol, "interval": timeframe, "open_time": open_time}
        )

    @classmethod
    async def get_mtf_klines(
        cls, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Multi-timeframe kline verilerini cache'den okur.
        Önce in-memory _pending_klines'a bakar (Redis round-trip yok).

        16 Ağu 2026: dönen buffer istenen limit'in yarısından azsa (eskiden
        restart anında bir kez dolup bir daha hiç tazelenmeyen, saatlerce
        eksik kalabilen buffer'lar — bkz. proje hafızası, aynı gün) arka
        planda (bloklamadan) CA'dan tazeleme tetiklenir. Bu fonksiyonu
        çağıran HER yer (tüm stratejiler, sinyal üretimi, masaüstü panel)
        otomatik kendi kendini onarır — tek tek strateji taşımaya gerek
        kalmaz."""
        key = cls._get_mtf_key(symbol, timeframe, "live_kline_data")
        df: Optional[pd.DataFrame] = None

        cached = cls._pending_klines.get(key)
        if cached is not None:
            arrow_bytes, _ = cached
            try:
                loop = asyncio.get_running_loop()
                raw_df = await loop.run_in_executor(
                    _ARROW_EXECUTOR, cls._arrow_bytes_to_df, arrow_bytes
                )
                if raw_df is not None and not raw_df.empty:
                    df = raw_df.tail(limit) if limit else raw_df
            except Exception:
                pass

        if df is None:
            try:
                raw_df = await cls.get_df(key)
                if raw_df is not None and not raw_df.empty:
                    df = raw_df.tail(limit) if limit else raw_df
            except Exception as e:
                logger.error("MTF klines okuma hatası [%s:%s]: %s", symbol, timeframe, e)

        # 16 Ağu 2026 düzeltmesi: gerçek çağıranların HİÇBİRİ limit vermiyor
        # (signal_service.py, signal_processor.py — hepsi limit=None ile
        # çağırıyor), yani yukarıdaki kontrol limit=None ile hiç tetiklenmezdi.
        # limit verilmemişse config'teki standart hedefi kullan.
        check_limit = limit or Config.MTF_BUFFER_LIMITS.get(timeframe)
        if check_limit and (df is None or len(df) < check_limit * 0.5):
            cls._maybe_trigger_mtf_refresh(symbol, timeframe, check_limit)

        return df

    @classmethod
    async def get_fresh_klines(
        cls, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Kritik karar noktaları (sinyal üretimi) için — CA'da olan TF'ler
        (bkz. _CAGG_TIMEFRAMES) HER ZAMAN doğrudan Postgres'ten okunur, asla
        bayat olamaz (~25ms: CA sorgusu + gösterge hesabı, thread pool'da —
        ana event loop'u bloke etmez). Diğer TF'ler (1m/30m/1d — CA
        karşılığı yok veya WS zaten sürekli taze tutuyor, DB'ye her turda
        gitmek gereksiz yük+gecikme olurdu) get_mtf_klines'ın kendi kendini
        onaran Redis okumasına düşer.

        17 Ağu 2026: signal_service.py/signal_processor.py'nin gerçek karar
        anlarında bayat önbelleğe bağımlı kalmaması için eklendi — bkz.
        proje hafızası (RedisClient'in "kritik yol Redis'e hiç bağımlı
        olmamalı" ilkesi)."""
        if timeframe in cls._CAGG_TIMEFRAMES:
            target_limit = limit or Config.MTF_BUFFER_LIMITS.get(timeframe, 250)
            try:
                from database.crud import (  # pylint: disable=import-outside-toplevel
                    get_cagg_klines,
                )
                from indicators.core import (  # pylint: disable=import-outside-toplevel
                    add_all_indicators,
                )

                raw = await get_cagg_klines(symbol, timeframe, target_limit)
                if raw is not None and not raw.empty:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(_INDICATOR_EXECUTOR, add_all_indicators, raw)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "get_fresh_klines CA okuma başarısız [%s:%s]: %s", symbol, timeframe, e
                )
        return await cls.get_mtf_klines(symbol, timeframe, limit)

    @classmethod
    def _maybe_trigger_mtf_refresh(cls, symbol: str, timeframe: str, limit: int) -> None:
        """Bayat/eksik bir get_mtf_klines sonucunu arka planda CA'dan tazeler
        — aynı anda tekrar tetiklenmesin (inflight) ve CA'nın kendisi de
        gerçekten eksikse (ör. sembol top-550'den düşmüş) her çağrıda
        boşuna Postgres'e gitmesin (cooldown) diye korumalı."""
        refresh_key = f"{symbol}:{timeframe}"
        now = time.time()
        if refresh_key in cls._mtf_refresh_inflight:
            return
        if now - cls._mtf_refresh_last_attempt.get(refresh_key, 0) < cls._MTF_REFRESH_COOLDOWN_SEC:
            return
        cls._mtf_refresh_last_attempt[refresh_key] = now
        cls._mtf_refresh_inflight.add(refresh_key)
        asyncio.create_task(cls._background_mtf_refresh(symbol, timeframe, limit, refresh_key))

    @classmethod
    async def _background_mtf_refresh(
        cls, symbol: str, timeframe: str, limit: int, refresh_key: str
    ) -> None:
        """16 Ağu 2026 düzeltmesi: get_cagg_klines HAM OHLCV döndürür
        (st_direction/ha_open/rsi vb. gösterge kolonları YOK) — bunları
        add_all_indicators ile hesaplamadan Redis'e yazmak, o sembolün
        Supertrend/HA_Cross tabanlı sinyallerini SESSİZCE kırardı (kolon
        yok → None/continue, hata bile vermez). add_all_indicators CPU
        işidir, ana event loop'u bloke etmemesi için thread pool'da
        çalıştırılır (live_data_manager.py::_load_symbol_all_timeframes
        ile aynı desen)."""
        try:
            from database.crud import (  # pylint: disable=import-outside-toplevel
                get_cagg_klines,
            )
            from indicators.core import (  # pylint: disable=import-outside-toplevel
                add_all_indicators,
            )

            fresh = await get_cagg_klines(symbol, timeframe, limit)
            if fresh is not None and not fresh.empty:
                loop = asyncio.get_running_loop()
                enriched = await loop.run_in_executor(
                    _INDICATOR_EXECUTOR, add_all_indicators, fresh
                )
                await cls.set_mtf_klines(symbol, timeframe, enriched, already_owned=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("MTF arka plan tazeleme başarısız [%s:%s]: %s", symbol, timeframe, e)
        finally:
            cls._mtf_refresh_inflight.discard(refresh_key)

    @classmethod
    async def set_mtf_indicators(
        cls, symbol: str, timeframe: str, indicators: Dict[str, Any]
    ) -> bool:
        """
        Multi-timeframe indikatör verilerini cache'e yazar.

        Args:
            symbol: Sembol
            timeframe: Zaman dilimi
            indicators: İndikatör verileri dictionary

        Returns:
            bool: Başarılı ise True
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "indicators")
            ttl = min(1800, cls._get_ttl_for_timeframe(timeframe))  # Max 30 dakika

            await cls.set_json(key, indicators, ex=ttl)
            logger.debug(f"MTF indicators cache'lendi: {key}")
            return True

        except Exception as e:
            logger.error(f"MTF indicators cache hatası [{symbol}:{timeframe}]: {e}")
            return False

    @classmethod
    async def get_mtf_indicators(cls, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Multi-timeframe indikatör verilerini cache'den okur.

        Args:
            symbol: Sembol
            timeframe: Zaman dilimi

        Returns:
            Dict: İndikatör verileri veya None
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "indicators")
            indicators = await cls.get_json(key)

            if indicators:
                logger.debug(f"MTF indicators cache'den okundu: {key}")
                return indicators

            return None

        except Exception as e:
            logger.error(f"MTF indicators okuma hatası [{symbol}:{timeframe}]: {e}")
            return None

    @classmethod
    async def set_mtf_signals(
        cls, symbol: str, timeframe: str, signals: List[Dict[str, Any]]
    ) -> bool:
        """
        Multi-timeframe sinyal verilerini cache'e yazar.

        Args:
            symbol: Sembol
            timeframe: Zaman dilimi
            signals: Sinyal listesi

        Returns:
            bool: Başarılı ise True
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "signals")
            ttl = 900  # 15 dakika (sinyaller daha kısa süre cache'lenir)

            await cls.set_json(key, signals, ex=ttl)
            logger.debug(f"MTF signals cache'lendi: {key} ({len(signals)} signals)")
            return True

        except Exception as e:
            logger.error(f"MTF signals cache hatası [{symbol}:{timeframe}]: {e}")
            return False

    @classmethod
    async def get_mtf_signals(cls, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """
        Multi-timeframe sinyal verilerini cache'den okur.

        Args:
            symbol: Sembol
            timeframe: Zaman dilimi

        Returns:
            List: Sinyal listesi veya None
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "signals")
            signals = await cls.get_json(key)

            if signals:
                logger.debug(f"MTF signals cache'den okundu: {key} ({len(signals)} signals)")
                return signals

            return None

        except Exception as e:
            logger.error(f"MTF signals okuma hatası [{symbol}:{timeframe}]: {e}")
            return None

    @classmethod
    async def flush_mtf_cache(cls, symbol: str, timeframes: Optional[List[str]] = None) -> int:
        """
        Belirtilen sembol için MTF cache'i temizler.

        Args:
            symbol: Sembol
            timeframes: Temizlenecek timeframe listesi (None ise tümü)

        Returns:
            int: Temizlenen key sayısı
        """
        try:
            r = cls.get_client()
            deleted_count = 0

            if timeframes is None:
                timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

            data_types = ["live_kline_data", "indicators", "signals"]

            for tf in timeframes:
                for data_type in data_types:
                    key = cls._get_mtf_key(symbol, tf, data_type)
                    if await r.delete(key):
                        deleted_count += 1
                        logger.debug(f"MTF cache temizlendi: {key}")

            logger.info(f"MTF cache flush tamamlandı [{symbol}]: {deleted_count} keys deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"MTF cache flush hatası [{symbol}]: {e}")
            return 0
        # Pool otomatik yönetir, close() gerekmez

    @classmethod
    async def get_mtf_cache_stats(cls, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        MTF cache istatistiklerini döndürür.

        Args:
            symbol: Sembol

        Returns:
            Dict: Cache istatistikleri
        """
        try:
            r = cls.get_client()
            stats = {}

            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            data_types = ["live_kline_data", "indicators", "signals"]

            for tf in timeframes:
                tf_stats = {}
                for data_type in data_types:
                    key = cls._get_mtf_key(symbol, tf, data_type)
                    exists = await r.exists(key)
                    ttl = await r.ttl(key) if exists else -1

                    tf_stats[data_type] = {"exists": bool(exists), "ttl": ttl, "key": key}

                stats[tf] = tf_stats

            return stats

        except Exception as e:
            logger.error(f"MTF cache stats hatası [{symbol}]: {e}")
            return {}
        # Pool otomatik yönetir, close() gerekmez

    @classmethod
    async def warm_mtf_cache(
        cls, symbol: str, timeframes: List[str], klines_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, bool]:
        """
        MTF cache'i önceden doldurur (cache warming).

        Args:
            symbol: Sembol
            timeframes: Timeframe listesi
            klines_data: Timeframe -> DataFrame mapping

        Returns:
            Dict: Timeframe -> success mapping
        """
        results = {}

        for tf in timeframes:
            if tf in klines_data and not klines_data[tf].empty:
                success = await cls.set_mtf_klines(symbol, tf, klines_data[tf])
                results[tf] = success

                if success:
                    logger.info(f"MTF cache warmed: {symbol}:{tf} ({len(klines_data[tf])} bars)")
            else:
                results[tf] = False
                logger.warning(f"MTF cache warm failed: {symbol}:{tf} - No data")

        return results


# Kolay erişim için bir instance oluştur
redis_client = RedisClient()
