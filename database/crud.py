from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import func, select, delete, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from .engine import get_session, init_db
from .models import PriceData, Signal
from datetime import datetime
from utils.kline_schema import check_kline_schema
from utils.logger import get_logger

logger = get_logger(__name__)

_CAGG_MAP = {
    "5m":  "cagg_5m",
    "15m": "cagg_15m",
    "1h":  "cagg_1h",
    "4h":  "cagg_4h",
}


async def get_cagg_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """CA view'larından (cagg_5m/15m/1h/4h) son N bar'ı Binance formatında döndürür."""
    view = _CAGG_MAP[interval]
    async with get_session() as session:
        result = await session.execute(
            text(f"""
                SELECT bucket, open, high, low, close, volume
                FROM {view}
                WHERE symbol = :sym
                ORDER BY bucket DESC
                LIMIT :lim
            """),
            {"sym": symbol, "lim": limit},
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    rows = sorted(rows, key=lambda r: r[0])
    df = pd.DataFrame({
        "open_time": [int(r[0].timestamp() * 1000) for r in rows],
        "open":      [r[1] for r in rows],
        "high":      [r[2] for r in rows],
        "low":       [r[3] for r in rows],
        "close":     [r[4] for r in rows],
        "volume":    [r[5] for r in rows],
    })
    # CA view'ları yönlü hacim taşımıyor (bilinen sınır — 1h/4h'yi WS doldurur)
    return check_kline_schema(df, "CA.get_cagg_klines", expect_directional=False)

async def initialize_database():
    """Veritabanını ve tabloları oluşturur."""
    await init_db()
    logger.info("Veritabanı başarıyla başlatıldı ve tablolar oluşturuldu.")

async def get_last_timestamp(symbol: str, interval: Optional[str] = None) -> Optional[int]:
    """Bir sembol (+opsiyonel interval) için veritabanındaki son `timestamp` değerini milisaniye cinsinden getirir."""
    async with get_session() as session:
        stmt = select(func.max(PriceData.timestamp)).where(PriceData.symbol == symbol)
        if interval:
            stmt = stmt.where(PriceData.interval == interval)
        result = await session.execute(stmt)
        last_timestamp_str = result.scalar_one_or_none()

        if last_timestamp_str:
            # DateTime nesnesini milisaniye cinsinden tamsayıya dönüştür
            if isinstance(last_timestamp_str, str):
                dt_obj = datetime.strptime(last_timestamp_str, '%Y-%m-%d %H:%M:%S')
                return int(dt_obj.timestamp() * 1000)
            else:
                # Zaten datetime nesnesi ise direkt dönüştür
                return int(last_timestamp_str.timestamp() * 1000)
            
        return None

async def get_oldest_timestamp(symbol: str, interval: Optional[str] = None) -> Optional[int]:
    """Bir sembol için veritabanındaki en eski timestamp'i milisaniye cinsinden döndürür."""
    async with get_session() as session:
        stmt = select(func.min(PriceData.timestamp)).where(PriceData.symbol == symbol)
        if interval:
            stmt = stmt.where(PriceData.interval == interval)
        result = await session.execute(stmt)
        oldest = result.scalar_one_or_none()
        if oldest:
            if isinstance(oldest, str):
                dt_obj = datetime.strptime(oldest, '%Y-%m-%d %H:%M:%S')
                return int(dt_obj.timestamp() * 1000)
            return int(oldest.timestamp() * 1000)
        return None

async def get_recent_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """DB'den son N kline'ı Binance formatında (open_time ms) döndürür.
    5m/15m/1h/4h → CA view'larından çekilir (boşluksuz, otomatik hesaplanan).
    """
    if interval in _CAGG_MAP:
        return await get_cagg_klines(symbol, interval, limit)

    async with get_session() as session:
        stmt = (
            select(PriceData)
            .where(PriceData.symbol == symbol)
            .where(PriceData.interval == interval)
            .order_by(PriceData.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    rows = sorted(rows, key=lambda r: r.timestamp)
    df = pd.DataFrame({
        "open_time":   [int(r.timestamp.timestamp() * 1000) for r in rows],
        "open":        [r.open        for r in rows],
        "high":        [r.high        for r in rows],
        "low":         [r.low         for r in rows],
        "close":       [r.close       for r in rows],
        "volume":      [r.volume      for r in rows],
        "buy_volume":  [r.buy_volume  for r in rows],
        "sell_volume": [r.sell_volume for r in rows],
    })
    return check_kline_schema(df, "DB.get_recent_klines")


async def bulk_insert_price_data(symbol: str, df: pd.DataFrame, interval: Optional[str] = None):
    """DataFrame'deki fiyat verisini toplu olarak kaydeder."""
    if df.empty:
        return

    df_copy = df.copy()
    df_copy['symbol'] = symbol
    if interval is not None:
        df_copy['interval'] = interval
    df_copy['timestamp'] = pd.to_datetime(df_copy['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)

    if 'taker_buy_base_asset_volume' in df_copy.columns:
        df_copy['buy_volume']  = df_copy['taker_buy_base_asset_volume']
        df_copy['sell_volume'] = df_copy['volume'] - df_copy['taker_buy_base_asset_volume']

    # Modele uygun sütunları seç ve 'open_time'ı hariç tut
    model_columns = [c.name for c in PriceData.__table__.columns]
    columns_to_keep = [col for col in df_copy.columns if col in model_columns]
    
    # DataFrame'i sadece modele uygun sütunlarla sınırla
    df_filtered = df_copy[columns_to_keep]
    
    records = df_filtered.to_dict('records')

    if not records:
        logger.warning(f"[{symbol}] Kaydedilecek geçerli sütun bulunamadı.")
        return

    # Büyük veri setlerini parçalara böl (PostgreSQL parametre limiti için)
    batch_size = 1000
    total_inserted = 0
    
    async with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            stmt = postgresql_insert(PriceData).values(batch)
            update_dict = {c.name: getattr(stmt.excluded, c.name) for c in PriceData.__table__.columns if not c.primary_key}
            stmt = stmt.on_conflict_do_update(
                constraint='price_data_symbol_interval_timestamp_key',
                set_=update_dict
            )
            try:
                await session.execute(stmt)
                total_inserted += len(batch)
            except Exception:
                await session.rollback()
                for record in batch:
                    try:
                        s = postgresql_insert(PriceData).values([record])
                        s = s.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                        await session.execute(s)
                        await session.commit()
                        total_inserted += 1
                    except Exception as row_exc:
                        await session.rollback()
                        logger.debug("[%s] Satır atlandı: %s", symbol, row_exc)

        await session.commit()
        logger.info(f"[{symbol}] {total_inserted} adet fiyat verisi kaydedildi.")

async def bulk_insert_price_data_multi(records: List[Dict[str, Any]]) -> int:
    """Birden fazla sembolün kline satırlarını tek transaction'da yazar.

    Her dict'te 'symbol', 'interval', 'open_time' ve OHLCV alanları olmalı.
    """
    if not records:
        return 0

    model_columns = {c.name for c in PriceData.__table__.columns}

    df = pd.DataFrame(records)
    df["timestamp"] = (
        pd.to_datetime(df["open_time"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
    )
    if "taker_buy_base_asset_volume" in df.columns:
        df["buy_volume"]  = df["taker_buy_base_asset_volume"]
        df["sell_volume"] = df["volume"] - df["taker_buy_base_asset_volume"]

    cols = [c for c in df.columns if c in model_columns]
    rows = df[cols].to_dict("records")

    batch_size = 1000
    total = 0
    async with get_session() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = postgresql_insert(PriceData).values(batch)
            update_dict = {
                c.name: getattr(stmt.excluded, c.name)
                for c in PriceData.__table__.columns
                if not c.primary_key
            }
            stmt = stmt.on_conflict_do_update(
                constraint="price_data_symbol_interval_timestamp_key",
                set_=update_dict,
            )
            try:
                await session.execute(stmt)
                total += len(batch)
            except Exception:  # pylint: disable=broad-exception-caught
                await session.rollback()
                for row in batch:
                    try:
                        s = postgresql_insert(PriceData).values([row])
                        s = s.on_conflict_do_nothing(index_elements=["symbol", "timestamp"])
                        await session.execute(s)
                        await session.commit()
                        total += 1
                    except Exception:  # pylint: disable=broad-exception-caught
                        await session.rollback()
        await session.commit()
    logger.info("bulk_insert_multi: %d satır yazıldı (%d sembol)", total, len({r.get('symbol') for r in records}))
    return total


async def save_price_data_batch(data_map: Dict[str, pd.DataFrame], interval: Optional[str] = None):
    """
    DataFrame haritasındaki fiyat verilerini toplu olarak kaydeder.
    Var olan kayıtları günceller (UPSERT).
    """
    all_records = []
    for symbol, df in data_map.items():
        if df.empty:
            continue

        df_copy = df.copy()
        df_copy['symbol'] = symbol
        if interval is not None:
            df_copy['interval'] = interval
        df_copy['timestamp'] = pd.to_datetime(df_copy['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)

        if 'taker_buy_base_asset_volume' in df_copy.columns:
            df_copy['buy_volume']  = df_copy['taker_buy_base_asset_volume']
            df_copy['sell_volume'] = df_copy['volume'] - df_copy['taker_buy_base_asset_volume']

        # Sadece modelde olan sütunları tut
        model_columns = [c.name for c in PriceData.__table__.columns]
        columns_to_keep = [col for col in df_copy.columns if col in model_columns]
        df_filtered = df_copy[columns_to_keep]

        records = df_filtered.to_dict('records')
        if records:
            all_records.extend(records)

    if not all_records:
        logger.warning("Toplu kaydetme için işlenecek veri bulunamadı.")
        return

    # Büyük veri setlerini parçalara böl (PostgreSQL parametre limiti için)
    batch_size = 1000
    total_inserted = 0
    
    async with get_session() as session:
        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i + batch_size]
            # PostgreSQL için 'upsert' ifadesi
            stmt = postgresql_insert(PriceData).values(batch)
            
            # ON CONFLICT...DO UPDATE ifadesini oluştur
            # Primary key olmayan sütunları güncelle
            update_dict = {
                c.name: getattr(stmt.excluded, c.name) 
                for c in PriceData.__table__.columns 
                if not c.primary_key
            }
            
            stmt = stmt.on_conflict_do_update(
                constraint='price_data_symbol_interval_timestamp_key',
                set_=update_dict
            )
            
            await session.execute(stmt)
            total_inserted += len(batch)
        
        await session.commit()
        logger.info(f"Toplu kaydetme tamamlandı. {total_inserted} adet kayıt işlendi.")

async def delete_symbol_data(symbol: str):
    """Bir sembole ait tüm verileri (fiyat ve sinyal) veritabanından siler."""
    async with get_session() as session:
        try:
            # PriceData tablosundan sil
            delete_price_stmt = delete(PriceData).where(PriceData.symbol == symbol)
            await session.execute(delete_price_stmt)

            # Signal tablosundan sil
            delete_signal_stmt = delete(Signal).where(Signal.symbol == symbol)
            await session.execute(delete_signal_stmt)

            logger.info(f"[{symbol}] için tüm veritabanı kayıtları başarıyla silindi.")
        except Exception as e:
            logger.error(f"[{symbol}] verileri silinirken hata oluştu: {e}", exc_info=True)
            # Hata durumunda get_session context manager'ı rollback yapacaktır.
            raise

async def get_all_price_data_with_indicators() -> List[Dict[str, Any]]:
    """Tüm semboller için en son fiyat ve indikatör verilerini döndürür."""
    async with get_session() as session:
        # Her sembol için en son kaydı almak üzere bir alt sorgu
        subq = (
            select(PriceData.symbol, func.max(PriceData.timestamp).label('max_ts'))
            .group_by(PriceData.symbol)
            .alias('latest')
        )

        # Ana sorgu: Alt sorgu ile birleştirerek tam satırları al
        stmt = (
            select(PriceData)
            .join(subq, (PriceData.symbol == subq.c.symbol) & (PriceData.timestamp == subq.c.max_ts))
        )

        result = await session.execute(stmt)
        price_data_list = result.scalars().all()
        return [p.to_dict() for p in price_data_list if hasattr(p, 'to_dict')]

