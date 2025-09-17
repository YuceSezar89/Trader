"""
Streamlit için güvenli database connection manager.
Event loop çakışmalarını önler.
"""
import asyncio
import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import threading
import time

logger = logging.getLogger(__name__)

class StreamlitSafeConnectionManager:
    """Streamlit için thread-safe database connection manager."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._local = threading.local()
        self._connection_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def _get_thread_id(self) -> str:
        """Mevcut thread ID'sini al."""
        return f"thread_{threading.get_ident()}"
    
    async def _create_connection(self) -> asyncpg.Connection:
        """Yeni connection oluştur."""
        try:
            # PgBouncer URL'sini asyncpg formatına çevir
            url = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(url, command_timeout=30)
            logger.info(f"Yeni connection oluşturuldu: {self._get_thread_id()}")
            return conn
        except Exception as e:
            logger.error(f"Connection oluşturma hatası: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Thread-safe connection context manager."""
        thread_id = self._get_thread_id()
        conn = None
        
        try:
            # Thread için connection al veya oluştur
            with self._lock:
                if thread_id not in self._connection_cache:
                    logger.info(f"Thread {thread_id} için yeni connection oluşturuluyor")
                
            # Her zaman yeni connection oluştur (Streamlit için güvenli)
            conn = await self._create_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"Connection hatası: {e}")
            raise
        finally:
            # Connection'ı güvenli şekilde kapat
            if conn:
                try:
                    await conn.close()
                    logger.debug(f"Connection kapatıldı: {thread_id}")
                except Exception as e:
                    logger.warning(f"Connection kapatma hatası: {e}")
    
    async def execute_query(self, query: str, *args) -> list:
        """Güvenli query execution."""
        async with self.get_connection() as conn:
            try:
                result = await conn.fetch(query, *args)
                return [dict(row) for row in result]
            except Exception as e:
                logger.error(f"Query hatası: {query[:100]}... - {e}")
                raise
    
    async def execute_one(self, query: str, *args) -> Optional[Dict[Any, Any]]:
        """Tek satır döndüren query."""
        async with self.get_connection() as conn:
            try:
                result = await conn.fetchrow(query, *args)
                return dict(result) if result else None
            except Exception as e:
                logger.error(f"Query hatası: {query[:100]}... - {e}")
                raise
    
    def cleanup(self):
        """Tüm connection'ları temizle."""
        with self._lock:
            self._connection_cache.clear()
            logger.info("Connection cache temizlendi")

# Global instance
_connection_manager: Optional[StreamlitSafeConnectionManager] = None

def get_connection_manager() -> StreamlitSafeConnectionManager:
    """Global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        from database.engine import DATABASE_URL
        _connection_manager = StreamlitSafeConnectionManager(DATABASE_URL)
    return _connection_manager

async def safe_query(query: str, *args) -> list:
    """Streamlit-safe query execution."""
    manager = get_connection_manager()
    return await manager.execute_query(query, *args)

async def safe_query_one(query: str, *args) -> Optional[Dict[Any, Any]]:
    """Streamlit-safe single row query."""
    manager = get_connection_manager()
    return await manager.execute_one(query, *args)
