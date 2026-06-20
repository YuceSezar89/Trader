"""
Redis'te yetersiz bar bulunan sembolleri DB'den yeniden yükler.
1m DB verisini çekip 5m/15m aggregate eder, indikatörler ekler, Redis'e yazar.

Kullanım:
    python scripts/reload_redis_klines.py              # otomatik tespit
    python scripts/reload_redis_klines.py MASKUSDT REDUSDT  # belirli semboller
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import redis as redis_sync

from database.crud import get_recent_klines
from indicators.core import add_all_indicators
from utils.redis_client import RedisClient
from utils.timeframe_aggregator import TimeframeAggregator
from utils.logger import get_logger

logger = get_logger(__name__)

_ARROW_MAGIC = b"ARDF"
_BUFFER_LIMITS = {"1m": 1000, "5m": 300, "15m": 100}
_MIN_BARS = {"1m": 300, "5m": 100, "15m": 50}


def _current_bar_count(symbol: str, tf: str) -> int:
    r = redis_sync.Redis()
    raw = r.get(f"live_kline_data:{symbol}:{tf}")
    if not raw:
        return 0
    try:
        data = raw[len(_ARROW_MAGIC):]
        reader = pa.ipc.open_stream(data)
        return len(reader.read_pandas())
    except Exception:
        return 0


async def _reload_symbol(symbol: str) -> bool:
    logger.info("[%s] DB'den 1m yükleniyor...", symbol)
    df_1m = await get_recent_klines(symbol, "1m", 1500)
    if df_1m.empty:
        logger.warning("[%s] DB'de 1m verisi yok, atlanıyor", symbol)
        return False

    loop = asyncio.get_event_loop()

    df_1m_ind = await loop.run_in_executor(None, add_all_indicators, df_1m)
    df_1m_trimmed = df_1m_ind.tail(_BUFFER_LIMITS["1m"])
    await RedisClient.set_mtf_klines(symbol, "1m", df_1m_trimmed)
    logger.info("[%s] 1m: %d bar → Redis", symbol, len(df_1m_trimmed))

    for agg_tf in ["5m", "15m"]:
        if not TimeframeAggregator.can_aggregate("1m", agg_tf):
            continue
        agg_df = TimeframeAggregator.aggregate_ohlcv(df_1m, "1m", agg_tf)
        if agg_df.empty:
            logger.warning("[%s] %s aggregate başarısız", symbol, agg_tf)
            continue
        limit = _BUFFER_LIMITS[agg_tf]
        agg_ind = await loop.run_in_executor(None, add_all_indicators, agg_df.tail(limit))
        await RedisClient.set_mtf_klines(symbol, agg_tf, agg_ind)
        logger.info("[%s] %s: %d bar → Redis", symbol, agg_tf, len(agg_ind))

    return True


async def main():
    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
    else:
        r = redis_sync.Redis()
        all_keys = r.keys("live_kline_data:*:1m")
        symbols = []
        for key in all_keys:
            sym = key.decode().split(":")[1]
            count = _current_bar_count(sym, "1m")
            if count < _MIN_BARS["1m"]:
                symbols.append(sym)
        if not symbols:
            print("Yetersiz bar bulunan sembol yok.")
            return
        print(f"Yetersiz bar bulunan {len(symbols)} sembol: {symbols}")

    ok, fail = 0, 0
    for sym in symbols:
        before_1m = _current_bar_count(sym, "1m")
        success = await _reload_symbol(sym)
        if success:
            after_1m = _current_bar_count(sym, "1m")
            print(f"✅ {sym}: 1m {before_1m} → {after_1m} bar")
            ok += 1
        else:
            print(f"❌ {sym}: yüklenemedi")
            fail += 1

    print(f"\nTamamlandı: {ok} başarılı, {fail} başarısız")


if __name__ == "__main__":
    asyncio.run(main())
