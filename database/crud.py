from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import func, select, update, insert, delete
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import OperationalError
import asyncio

from .engine import get_session, init_db
from .models import PriceData, Signal
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)

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

async def bulk_insert_price_data(symbol: str, df: pd.DataFrame, interval: Optional[str] = None):
    """DataFrame'deki fiyat verisini toplu olarak kaydeder."""
    if df.empty:
        return

    df_copy = df.copy()
    df_copy['symbol'] = symbol
    if interval is not None:
        df_copy['interval'] = interval
    df_copy['timestamp'] = pd.to_datetime(df_copy['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)
    
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
            # ON CONFLICT...DO UPDATE
            update_dict = {c.name: getattr(stmt.excluded, c.name) for c in PriceData.__table__.columns if not c.primary_key}
            stmt = stmt.on_conflict_do_update(
                constraint='price_data_symbol_interval_timestamp_key',
                set_=update_dict
            )
            await session.execute(stmt)
            total_inserted += len(batch)
        
        await session.commit()
        logger.info(f"[{symbol}] {total_inserted} adet fiyat verisi kaydedildi.")

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
        # open_time'dan Europe/Istanbul zaman dilimine göre formatlanmış string oluştur
        df_copy['timestamp'] = pd.to_datetime(df_copy['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)

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

async def create_signal(signal_data: Dict[str, Any]):
    """Tek bir sinyal kaydeder. Var olan kaydı günceller."""
    async with get_session() as session:
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        
        stmt = postgresql_insert(Signal).values(signal_data)
        # Sadece signal_data içinde gönderilen ve primary key olmayan alanları güncelle
        update_dict = {
            key: getattr(stmt.excluded, key)
            for key in signal_data.keys()
            if key in Signal.__table__.columns and not Signal.__table__.columns[key].primary_key
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol', 'timestamp', 'signal_type', 'interval'],
            set_=update_dict
        )

        # Exponential backoff ile kilit hatalarına karşı tekrar dene
        max_attempts = 5
        delay = 0.2
        for attempt in range(1, max_attempts + 1):
            try:
                await session.execute(stmt)
                logger.info(f"Sinyal kaydedildi/güncellendi: {signal_data.get('symbol')} - {signal_data.get('signal_type')}")
                break
            except OperationalError as e:
                if 'database is locked' in str(e).lower() and attempt < max_attempts:
                    logger.warning(f"DB kilitli, tekrar denenecek ({attempt}/{max_attempts})...")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise

async def get_recent_signals(hours: int = 24) -> List[Dict[str, Any]]:
    """Son 'hours' saat içindeki sinyalleri veritabanından çeker."""
    async with get_session() as session:
        time_threshold = datetime.now() - timedelta(hours=hours)

        stmt = (
            select(Signal)
            .where(Signal.timestamp >= time_threshold)
            .order_by(Signal.timestamp.desc())
        )
        result = await session.execute(stmt)
        signals = result.scalars().all()
        return [s.to_dict() for s in signals if hasattr(s, 'to_dict')]

async def get_signal_stats() -> Dict[str, Any]:
    """Veritabanındaki sinyal istatistiklerini hesaplar."""
    async with get_session() as session:
        total_signals_stmt = select(func.count()).select_from(Signal)
        long_signals_stmt = select(func.count()).where(Signal.signal_type == 'Long')
        short_signals_stmt = select(func.count()).where(Signal.signal_type == 'Short')
        top_symbols_stmt = (
            select(Signal.symbol, func.count(Signal.symbol).label('count'))
            .group_by(Signal.symbol)
            .order_by(func.count(Signal.symbol).desc())
            .limit(5)
        )

        total_signals = (await session.execute(total_signals_stmt)).scalar_one() or 0
        long_signals = (await session.execute(long_signals_stmt)).scalar_one() or 0
        short_signals = (await session.execute(short_signals_stmt)).scalar_one() or 0
        top_symbols_result = (await session.execute(top_symbols_stmt)).all()
        top_symbols = [{'symbol': row.symbol, 'count': row.count} for row in top_symbols_result]

        return {
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'top_symbols': top_symbols
        }

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

# async def clear_all_signals() -> int:
#     """Signals tablosundaki TÜM kayıtları siler. Geriye silinen satır sayısını döndürür.
#     DİKKAT: Bu işlem geri alınamaz; sadece 'signals' tablosunu etkiler, 'price_data' korunur.
#     Bu fonksiyon olası kazara kullanım riskine karşı geçici olarak devre dışı bırakılmıştır.
#     """
#     async with get_session() as session:
#         stmt = delete(Signal)
#         result = await session.execute(stmt)
#         # SQLAlchemy 2.0'da result.rowcount bazı sürücülerde -1 dönebilir; yine de geriye int döndürmeye çalışalım
#         deleted = getattr(result, "rowcount", -1) or -1
#         logger.info(f"Signals tablosu temizlendi. Silinen kayıt sayısı: {deleted}")
#         return int(deleted) if isinstance(deleted, int) else -1
