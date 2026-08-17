"""Açık pozisyon kayıt (registry) döngüleri — bkz. paper_trade_manager.py.

17 Ağu 2026: live_data_manager.py'den (run_services.py, best-effort görevlerle
aynı event loop) buraya (signal_service.py, kritik-iş process'i) taşındı.

`open_trade_registry_loop`: hangi (symbol, interval) çiftlerinin açık paper
trade'i olduğunu izler — `publish_atr_live`'ın hangi semboller için güncel
ATR yayınlaması gerektiğini bilmesi için (paper_trade_manager.py::_trail_distance
bunu trailing-stop mesafesi için okur).

`publish_atr_live`: eskiden live_data_manager'ın process-içi, WS'den gelen
canlı buffer'ını (`df["atr"]`, incremental hesap) kullanıyordu — bu veri
SADECE o process'te vardı, taşınamazdı. Artık signal_service.py'nin zaten
her bar kapanışında çağırdığı `RedisClient.get_fresh_klines` (CA'dan taze
okuyup add_all_indicators ile atr kolonunu hesaplayan fonksiyon) sonucunu
kullanıyor — `_process_event` içinde ZATEN elde edilmiş df, ekstra sorgu
gerekmez.

`manual_refresh_loop`: UI'dan açılan manuel işlemleri algılayıp
manual_manager cache'ini yeniler — hiçbir process-içi bağımlılığı yoktu,
doğrudan taşındı.
"""

import asyncio
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import select as _sel

from database.engine import get_session, run_with_db_timeout
from database.models import PaperTrade as _PT
from signals.paper_trade_manager import manual_manager
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient

logger = logging.getLogger("OpenTradeRegistry")
manual_logger = logging.getLogger("ManualTrade")

_REGISTRY_INTERVAL = 60
_MANUAL_INTERVAL = 10
_ATR_LIVE_TTL_BY_INTERVAL = {"1m": 180, "5m": 900, "15m": 2700, "1h": 10800, "4h": 43200}

open_trade_symbols: set[tuple[str, str]] = set()


async def open_trade_registry_loop() -> None:
    """Her 60 saniyede bir açık paper trade'leri (TÜM stratejiler, TÜM
    durumlar) DB'den çekip open_trade_symbols'ı tazeler. Signal.status='active'
    kapsamından KASITLI olarak ayrı: bir paper trade, kendi Signal'i
    kapansa/supersede olsa bile SL/TP/trailing mantığıyla açık kalabiliyor."""
    global open_trade_symbols  # pylint: disable=global-statement
    while True:
        try:
            async with get_session() as session:
                rows = (
                    await run_with_db_timeout(
                        session.execute(_sel(_PT.symbol, _PT.interval).where(_PT.status == "open"))
                    )
                ).all()
            open_trade_symbols = {(symbol, interval) for symbol, interval in rows}
            logger.debug(
                "%d açık trade, %d (sembol,TF) çifti", len(rows), len(open_trade_symbols)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Yenileme hatası: %s", exc)
        await asyncio.sleep(_REGISTRY_INTERVAL)


async def publish_atr_live(symbol: str, interval: str, df: Optional[pd.DataFrame]) -> None:
    """(symbol, interval) için güncel ATR'yi (get_fresh_klines'ın zaten
    hesapladığı df["atr"]) Redis'e yazar — trailing-stop mesafesinin
    pozisyon açılışındaki SABİT ATR yerine GÜNCEL volatiliteye göre
    ayarlanabilmesi için. Sadece açık trade'i olan (symbol,interval) için,
    _process_event zaten çağrıldığı için ekstra sorgu gerekmez."""
    if df is None or df.empty or "atr" not in df.columns:
        return
    try:
        atr_val = float(df["atr"].iloc[-1])
    except (ValueError, TypeError, IndexError):
        return
    if atr_val <= 0:
        return
    ttl = _ATR_LIVE_TTL_BY_INTERVAL.get(interval, 900)
    redis = RedisClient.get_client()
    try:
        await asyncio.wait_for(
            redis.set(f"atr_live:{symbol}:{interval}", atr_val, ex=ttl),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[AtrLive] %s %s Redis yazımı başarısız: %s", symbol, interval, exc)


async def manual_refresh_loop() -> None:
    """UI'dan açılan manuel işlemleri algılayıp manual_manager cache'ini yeniler."""
    redis = RedisClient.get_client()
    while True:
        await asyncio.sleep(_MANUAL_INTERVAL)
        try:
            val = await redis.get("manual_trade:refresh")
            if val:
                await manual_manager.load_open_symbols()
                await redis.delete("manual_trade:refresh")
                manual_logger.info(
                    "Cache yenilendi: %d açık sembol",
                    len(manual_manager._open_symbols),  # pylint: disable=protected-access
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            manual_logger.debug("refresh kontrolü başarısız: %s", exc)
