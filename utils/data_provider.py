import pandas as pd
from typing import Optional, List, Dict, Any

from utils.logger import get_logger
from utils.redis_client import RedisClient
from binance_client import BinanceClientManager
from config import Config

logger = get_logger(__name__)

REDIS_KEY_PREFIX = getattr(Config, 'REDIS_KEY_PREFIX', 'live_kline_data')


def _redis_key(symbol: str, interval: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{symbol}:{interval}"


def _legacy_redis_key(symbol: str) -> str:
    # Geriye uyumluluk: eski anahtar yapısı
    return f"{REDIS_KEY_PREFIX}:{symbol}"


async def _normalize_df(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Veri kaynağı ne olursa olsun standart kolon/tiplere normalize et."""
    if df is None or df.empty:
        return pd.DataFrame()

    cols = list(df.columns)
    # Beklenen kolonlar varsa kesit al
    expected = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    keep = [c for c in expected if c in cols]
    if keep:
        df = df[keep].copy()
    else:
        # Kademeli eşleştirme
        mapping = {}
        for c in cols:
            lc = c.lower()
            if 'open' == lc:
                mapping[c] = 'open'
            elif 'high' == lc:
                mapping[c] = 'high'
            elif 'low' == lc:
                mapping[c] = 'low'
            elif 'close' == lc:
                mapping[c] = 'close'
            elif 'volume' in lc:
                mapping[c] = 'volume'
            elif 'time' in lc:
                mapping[c] = 'open_time'
        if mapping:
            df = df.rename(columns=mapping)
            df = df[[k for k in ['open_time','open','high','low','close','volume'] if k in df.columns]].copy()

    # Tipler
    if 'open_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna()
    if symbol is not None and 'symbol' not in df.columns:
        df['symbol'] = symbol
    return df


async def fetch_ohlcv(
    symbol: str,
    interval: str,
    limit: int = 300,
    source: str = 'auto',
    cache_write: bool = True,
) -> pd.DataFrame:
    """
    Tek giriş noktası: OHLCV verisini getirir.
    Öncelik: Redis -> REST. (Gelecekte WS ingestor yalnızca Redis'e yazar.)
    """
    symbol = symbol.upper()

    async def from_redis() -> Optional[pd.DataFrame]:
        # Yeni anahtar
        key = _redis_key(symbol, interval)
        df = await RedisClient.get_df(key)
        if df is not None and not df.empty:
            return await _normalize_df(df, symbol)
        # Eski anahtar (geriye uyum)
        legacy = await RedisClient.get_df(_legacy_redis_key(symbol))
        if legacy is not None and not legacy.empty:
            # Eski anahtar alt TF olabilir; yine de normalize edip döndür.
            return await _normalize_df(legacy, symbol)
        return None

    async def from_rest() -> pd.DataFrame:
        try:
            df = await BinanceClientManager.fetch_klines(symbol, interval, limit=limit)
            df = await _normalize_df(df, symbol)
            if cache_write and not df.empty:
                try:
                    await RedisClient.set_df(_redis_key(symbol, interval), df)
                except Exception as e:
                    logger.warning(f"Redis yazımı başarısız: {e}")
            return df
        except Exception as e:
            logger.warning(f"REST veri çekimi hatası [{symbol} {interval}]: {e}")
            return pd.DataFrame()

    # Kaynak seçimi
    if source == 'redis':
        return await from_redis() or pd.DataFrame()
    if source == 'rest':
        return await from_rest()

    # auto
    df = await from_redis()
    if df is not None and not df.empty:
        return df.tail(limit)
    return await from_rest()


async def get_latest_bar(symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    df = await fetch_ohlcv(symbol, interval, limit=2, source='auto')
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    return {
        'open_time': last['open_time'],
        'open': float(last['open']),
        'high': float(last['high']),
        'low': float(last['low']),
        'close': float(last['close']),
        'volume': float(last['volume']),
        'symbol': symbol.upper(),
        'interval': interval,
    }


async def warm_cache(symbols: List[str], interval: str, limit: int = 300) -> None:
    for sym in symbols:
        try:
            df = await fetch_ohlcv(sym, interval, limit=limit, source='rest', cache_write=True)
            if df is not None and not df.empty:
                logger.info(f"Cache warm: {sym} {interval} ({len(df)})")
        except Exception as e:
            logger.warning(f"Cache warm başarısız: {sym} {interval} -> {e}")
