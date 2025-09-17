from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text
import os

from .models import Base

# PostgreSQL connection URL - PgBouncer connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://yusuf@localhost:6432/trader_panel"
)

# Asenkron veritabanı motoru
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # True yaparsanız tüm SQL sorgularını loglar
    pool_size=10,  # PgBouncer ile daha az pool
    max_overflow=5,  # PgBouncer ile daha az overflow
    pool_pre_ping=False,
    future=True,
    pool_timeout=30,  # Timeout artırıldı
    pool_recycle=1800,  # 30 dakika sonra connection'ları yenile
    connect_args={
        "server_settings": {
            "application_name": "trader_panel",
        },
        "command_timeout": 30,
    },
    # Event loop cleanup için ek ayarlar
    pool_reset_on_return='commit'
)

# Asenkron oturum yöneticisi
async_session_maker = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Veritabanı tablolarını (modelleri) oluşturur ve TimescaleDB hypertables kurar."""
    async with async_engine.begin() as conn:
        # PostgreSQL tabloları oluştur
        await conn.run_sync(Base.metadata.create_all)
        
        # TimescaleDB hypertables oluştur
        try:
            # price_data tablosunu hypertable'a çevir
            await conn.execute(text("""
                SELECT create_hypertable('price_data', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """))
            
            # signals tablosunu hypertable'a çevir
            await conn.execute(text("""
                SELECT create_hypertable('signals', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """))
            
            # Performans için indexler oluştur
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_price_data_symbol_interval 
                ON price_data (symbol, interval, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_signal_type 
                ON signals (symbol, signal_type, timestamp DESC);
            """))
            
        except Exception as e:
            # Hypertable zaten varsa veya başka bir hata varsa devam et
            print(f"Hypertable creation warning: {e}")
            pass

from contextlib import asynccontextmanager

@asynccontextmanager
async def get_session():
    """Async database session context manager."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
