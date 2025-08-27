from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text

from .models import Base

DATABASE_URL = "sqlite+aiosqlite:///signals.db"

# Asenkron veritabanı motoru
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # True yaparsanız tüm SQL sorgularını loglar
    connect_args={"timeout": 60},  # Yoğun eşzamanlı yazım için timeout artırıldı
    future=True,
)

# Asenkron oturum yöneticisi
async_session_maker = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Veritabanı tablolarını (modelleri) oluşturur."""
    async with async_engine.begin() as conn:
        # SQLite eşzamanlılık için önerilen PRAGMA ayarları
        # WAL modu daha iyi concurrent read/write sağlar
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        await conn.exec_driver_sql("PRAGMA busy_timeout=5000;")
        # await conn.run_sync(Base.metadata.drop_all) # Gerekirse tabloları siler
        await conn.run_sync(Base.metadata.create_all)
        # --- Lightweight migration: ensure new columns/indexes exist (SQLite) ---
        try:
            # PriceData: interval kolonu ve unique index
            res_pd = await conn.exec_driver_sql("PRAGMA table_info('price_data');")
            pd_cols = [row[1] for row in res_pd.fetchall()]
            if 'interval' not in pd_cols:
                await conn.exec_driver_sql("ALTER TABLE price_data ADD COLUMN interval TEXT;")
            # Unique index (symbol, interval, timestamp)
            await conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_price_data_symbol_interval_timestamp "
                "ON price_data (symbol, interval, timestamp);"
            )

            # Signals tablosunda eksik kolonları tespit et
            res = await conn.exec_driver_sql("PRAGMA table_info('signals');")
            cols = [row[1] for row in res.fetchall()]  # (cid, name, type, ...)
            alter_statements = []
            if 'vpms_score' not in cols:
                alter_statements.append("ALTER TABLE signals ADD COLUMN vpms_score REAL;")
            if 'vpm_confirmed' not in cols:
                # BOOLEAN SQLite'da affinity olarak INTEGER kabul edilir
                alter_statements.append("ALTER TABLE signals ADD COLUMN vpm_confirmed BOOLEAN;")
            if 'mtf_score' not in cols:
                alter_statements.append("ALTER TABLE signals ADD COLUMN mtf_score REAL;")
            if 'vpms_mtf_score' not in cols:
                alter_statements.append("ALTER TABLE signals ADD COLUMN vpms_mtf_score REAL;")
            for stmt in alter_statements:
                await conn.exec_driver_sql(stmt)
        except Exception:
            # Migration hatası kritik değil; sonraki açılışta tekrar denenecek
            pass

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Veritabanı oturumu sağlayan bir context manager."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
