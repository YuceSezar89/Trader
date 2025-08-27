import asyncio
import json
import logging
from typing import Any, Dict, Optional

import pandas as pd
import redis.asyncio as redis

from config import Config

logger = logging.getLogger(__name__)


class RedisClient:
    """Asenkron Redis istemcisi için merkezi bir yönetici sınıfı.

    Her process/event loop için ayrı bir bağlantı havuzu yönetir.
    """
    _pools: Dict[int, redis.ConnectionPool] = {}

    @classmethod
    def _get_pool_for_current_loop(cls) -> redis.ConnectionPool:
        """Mevcut asyncio event loop için bir bağlantı havuzu oluşturur veya döndürür."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Eğer çalışan bir loop yoksa, bu genellikle senkron bir bağlamdır.
            # Yeni bir loop oluşturup onu kullanabiliriz, ancak bu genellikle
            # Streamlit gibi framework'lerin kendi loop yönetimini bekleriz.
            # Bu senaryoda, loop-suz bir anahtar kullanalım.
            loop = None

        loop_id = id(loop)

        if loop_id not in cls._pools:
            logger.info(f"Yeni Redis bağlantı havuzu oluşturuluyor (Loop ID: {loop_id})")
            cls._pools[loop_id] = redis.ConnectionPool.from_url(
                Config.REDIS_URL, decode_responses=True
            )
        return cls._pools[loop_id]

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Mevcut event loop'a uygun bağlantı havuzundan bir istemci döndürür."""
        pool = cls._get_pool_for_current_loop()
        return redis.Redis(connection_pool=pool)

    @classmethod
    async def set_df(
        cls, key: str, df: pd.DataFrame, ex: int = 60 * 60 * 24
    ) -> None:  # 24 saat
        """Bir Pandas DataFrame'i JSON formatında Redis'e yazar."""
        r = cls.get_client()
        try:
            # DataFrame'i 'split' formatında JSON'a çevirerek sakla
            await r.set(key, df.to_json(orient="split"), ex=ex)
            logger.debug(f"DataFrame Redis'e yazıldı. Anahtar: {key}")
        except Exception as e:
            logger.error(f"Redis'e DataFrame yazma hatası (Anahtar: {key}): {e}")
        finally:
            await r.close()

    @classmethod
    async def get_df(cls, key: str) -> Optional[pd.DataFrame]:
        """Redis'ten bir DataFrame'i okur."""
        r = cls.get_client()
        try:
            data = await r.get(key)
            if data:
                logger.debug(f"DataFrame Redis'ten okundu. Anahtar: {key}")
                # 'split' formatında kaydedilen JSON'u DataFrame'e geri çevir
                df = pd.read_json(data, orient="split")
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                return df
            return None
        except Exception as e:
            logger.error(f"Redis'ten DataFrame okuma hatası (Anahtar: {key}): {e}")
            return None
        finally:
            await r.close()

    @classmethod
    async def set_json(cls, key: str, data: Any, ex: int = 3600) -> None:
        """Herhangi bir Python nesnesini JSON olarak Redis'e yazar."""
        r = cls.get_client()
        try:
            await r.set(key, json.dumps(data), ex=ex)
            logger.debug(f"JSON veri Redis'e yazıldı. Anahtar: {key}")
        except Exception as e:
            logger.error(f"Redis'e JSON yazma hatası (Anahtar: {key}): {e}")
        finally:
            await r.close()

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
        finally:
            await r.close()


# Kolay erişim için bir instance oluştur
redis_client = RedisClient()
