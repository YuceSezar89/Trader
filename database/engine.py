from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text
import asyncio
import os

from .models import Base

# Hot-path DB işlemlerini (SignalFilter.check, sinyal yazma/kapama vb.) sarmalamak
# için ortak zaman aşımı — 8 Tem: gerçek bir ağ kesintisi sırasında timeout'suz bir
# DB session'ı saatlerce (4+ saat) sessizce askıda kalmış, hem live_data_manager.py
# hem signal_service.py'nin işlem döngüsünü donuk hâle getirmişti (heartbeat bile
# bunu yakalayamadı — bkz. memory: project_data_layer_debt.md). Bu, Redis çağrıları
# için zaten var olan SAFE_EXTERNAL_TIMEOUT (utils/redis_client.py) deseninin DB
# tarafındaki eşleniği.
DB_CALL_TIMEOUT = 5.0


async def run_with_db_timeout(coro):
    """Bir DB coroutine'ini DB_CALL_TIMEOUT içinde tamamlanmaya zorlar — aksi
    halde TimeoutError fırlatır. Çağıran taraf bunu yakalayıp fail-closed/fail-safe
    davranmalı (ör. sinyali reddetmek, denemeyi loglayıp bir sonrakine geçmek)."""
    try:
        return await asyncio.wait_for(coro, timeout=DB_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        raise TimeoutError(f"DB çağrısı {DB_CALL_TIMEOUT}s içinde tamamlanmadı") from None

# PostgreSQL connection URL - Direct PostgreSQL connection
_db_host = os.getenv("DB_HOST", "localhost")
_db_port = os.getenv("DB_PORT", "6432")
_db_name = os.getenv("DB_NAME", "trader_panel")
_db_user = os.getenv("DB_USER", "yusuf")
_db_pass = os.getenv("DB_PASSWORD", "")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{_db_user}:{_db_pass}@{_db_host}:{_db_port}/{_db_name}"
)

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={
        "server_settings": {
            "application_name": "trader_panel",
            "timezone": "UTC",
        },
        "command_timeout": 30,
        "statement_cache_size": 0,
    },
    pool_reset_on_return="rollback",
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
            await session.execute(text("SET timezone = 'UTC'"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
