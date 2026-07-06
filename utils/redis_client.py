import asyncio
import functools
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pyarrow as pa
import redis.asyncio as redis

from config import Config

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_ARROW_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="arrow")


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
    _pending_publishes: Set[str] = set()                # "symbol:tf"
    _pending_kline_closed: List[Dict[str, Any]] = []     # XADD kline_closed için birikmiş event'ler
    _flusher_task: Optional[asyncio.Task] = None

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
                timeout=3,
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
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
            )
        return cls._binary_pools[loop_id]

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Mevcut event loop'a uygun bağlantı havuzundan bir istemci döndürür."""
        pool = cls._get_pool_for_current_loop()
        return redis.Redis(connection_pool=pool)

    @classmethod
    def _get_binary_client(cls) -> redis.Redis:
        """Binary veri için bağlantı havuzundan istemci döndürür."""
        pool = cls._get_binary_pool_for_current_loop()
        return redis.Redis(connection_pool=pool)

    @staticmethod
    def _dedupe_and_to_arrow_bytes(df: pd.DataFrame) -> bytes:
        if "open_time" in df.columns and df["open_time"].duplicated().any():
            df = df.drop_duplicates(subset=["open_time"], keep="last")
        return RedisClient._df_to_arrow_bytes(df)

    @staticmethod
    def _df_to_arrow_bytes(df: pd.DataFrame) -> bytes:
        df = df.copy()
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
    def _arrow_bytes_to_df(data: bytes) -> pd.DataFrame:
        reader = pa.ipc.open_stream(data[len(_ARROW_MAGIC):])
        return reader.read_pandas()

    @classmethod
    async def set_df(
        cls, key: str, df: pd.DataFrame, ex: int = 60 * 60 * 24
    ) -> None:
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
                    _ARROW_EXECUTOR,
                    functools.partial(pd.read_json, StringIO(text), orient="split")
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
            '1m': 86400,     # 24 saat
            '5m': 259200,    # 3 gün
            '15m': 604800,   # 7 gün
            '1h': 604800,    # 7 gün
            '4h': 1209600,   # 14 gün
            '1d': 2592000,   # 30 gün
        }
        return ttl_map.get(timeframe, 3600)  # Default: 1 saat
    
    @classmethod
    async def set_mtf_klines(cls, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """
        MTF kline'ı pending dict'e ekler — batch flusher Redis'e iter.
        Çağrı başına bağlantı yok, semaphore yok, burst sorunu yok.
        """
        try:
            key = cls._get_mtf_key(symbol, timeframe, "live_kline_data")
            ttl = cls._get_ttl_for_timeframe(timeframe)
            loop = asyncio.get_running_loop()
            arrow_bytes = await loop.run_in_executor(_ARROW_EXECUTOR, cls._dedupe_and_to_arrow_bytes, df)
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
                for msg in publishes:
                    pipe.publish("kline_updated", msg)
                for event in closed_events:
                    pipe.xadd("kline_closed", event, maxlen=100_000, approximate=True)
                await pipe.execute()
            logger.debug(
                "Batch flush: %d SET, %d PUBLISH, %d XADD",
                len(pending), len(publishes), len(closed_events),
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
            await asyncio.sleep(0.5)
            await cls.flush_pending_klines()
            await beat("redis_batch_flush")

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
    async def get_mtf_klines(cls, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Multi-timeframe kline verilerini cache'den okur.
        Önce in-memory _pending_klines'a bakar (Redis round-trip yok).
        """
        key = cls._get_mtf_key(symbol, timeframe, "live_kline_data")

        cached = cls._pending_klines.get(key)
        if cached is not None:
            arrow_bytes, _ = cached
            try:
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(_ARROW_EXECUTOR, cls._arrow_bytes_to_df, arrow_bytes)
                if df is not None and not df.empty:
                    return df.tail(limit) if limit else df
            except Exception:
                pass

        try:
            df = await cls.get_df(key)
            if df is not None and not df.empty:
                if limit:
                    df = df.tail(limit)
                return df
            return None
        except Exception as e:
            logger.error("MTF klines okuma hatası [%s:%s]: %s", symbol, timeframe, e)
            return None
    
    @classmethod
    async def set_mtf_indicators(cls, symbol: str, timeframe: str, indicators: Dict[str, Any]) -> bool:
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
    async def set_mtf_signals(cls, symbol: str, timeframe: str, signals: List[Dict[str, Any]]) -> bool:
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
                timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            
            data_types = ['live_kline_data', 'indicators', 'signals']
            
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
            
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            data_types = ['live_kline_data', 'indicators', 'signals']
            
            for tf in timeframes:
                tf_stats = {}
                for data_type in data_types:
                    key = cls._get_mtf_key(symbol, tf, data_type)
                    exists = await r.exists(key)
                    ttl = await r.ttl(key) if exists else -1
                    
                    tf_stats[data_type] = {
                        'exists': bool(exists),
                        'ttl': ttl,
                        'key': key
                    }
                
                stats[tf] = tf_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"MTF cache stats hatası [{symbol}]: {e}")
            return {}
        # Pool otomatik yönetir, close() gerekmez
    
    @classmethod
    async def warm_mtf_cache(cls, symbol: str, timeframes: List[str], 
                           klines_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
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
