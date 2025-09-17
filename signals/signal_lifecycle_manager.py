"""
Sinyal Lifecycle Yönetimi - Supersede Yaklaşımı
Aynı sembolde yeni sinyal geldiğinde eski aktif sinyali pasife çeker
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Mypy için tip bilgisi, runtime'da import edilmez
    pass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from database.models import Signal
from database.engine import get_session

logger = logging.getLogger(__name__)

class SignalLifecycleManager:
    """Sinyal yaşam döngüsü yöneticisi - supersede mekaniği"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def add_new_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Yeni sinyal ekler ve gerekirse eski sinyalleri supersede eder
        
        Args:
            signal_data: Sinyal verisi dictionary'si
            
        Returns:
            str: Eklenen sinyalin ID'si
        """
        async with get_session() as session:
            try:
                # 1. Önce aynı sembol + interval + indicators için aktif sinyalleri supersede et (signal_type fark etmez)
                superseded_count = await self._supersede_existing_signals(
                    session, 
                    signal_data['symbol'], 
                    signal_data['interval'],
                    signal_data.get('indicators'),
                    None  # signal_type'ı geçmiyoruz - aynı indikatörün tüm yönleri supersede edilecek
                )
                
                # 2. Yeni sinyali ekle (default status='active')
                new_signal = Signal()
                for key, value in signal_data.items():
                    setattr(new_signal, key, value)
                new_signal.status = 'active'  # type: ignore
                
                session.add(new_signal)
                await session.commit()  # Transaction'ı tamamla
                
                # Primary key (symbol, timestamp) kullan
                new_signal_id = f"{new_signal.symbol}_{new_signal.timestamp}"
                
                self.logger.info(
                    f"Yeni sinyal eklendi: {signal_data['symbol']} "
                    f"{signal_data['signal_type']} ({signal_data.get('indicators', 'N/A')}) "
                    f"(ID: {new_signal_id}) - {superseded_count} eski sinyal supersede edildi"
                )
                
                return new_signal_id
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Sinyal ekleme hatası: {e}", exc_info=True)
                raise
    
    async def _supersede_existing_signals(
        self, 
        session: AsyncSession, 
        symbol: str, 
        interval: str,
        indicators: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> int:
        """
        Aynı sembol + interval + indicators için aktif sinyalleri supersede eder
        (signal_type fark etmez - aynı indikatörün tüm yönleri supersede edilir)
        
        Returns:
            int: Supersede edilen sinyal sayısı
        """
        # Aktif sinyalleri bul - aynı indikatör ve sinyal türü kombinasyonu
        conditions = [
            Signal.symbol == symbol,
            Signal.interval == interval,
            Signal.status == 'active'
        ]
        
        # Eğer indicators belirtilmişse, onu filtrele (signal_type'ı dahil etme)
        if indicators is not None:
            conditions.append(Signal.indicators == indicators)
        # signal_type filtresini kaldırdık - aynı indikatörün tüm yönleri supersede edilecek
            
        stmt = select(Signal).where(and_(*conditions))
        
        # Debug log
        self.logger.info(
            f"Supersede sorgusu: {symbol} {interval} {signal_type} {indicators}"
        )
        
        result = await session.execute(stmt)
        active_signals = result.scalars().all()
        
        if not active_signals:
            return 0
        
        # Aktif sinyalleri supersede et
        superseded_count = 0
        current_time = datetime.now()
        
        for signal in active_signals:
            # Doğrudan nesne üzerinde güncelleme (mypy uyumlu)
            signal.status = 'superseded'  # type: ignore
            signal.superseded_at = current_time  # type: ignore
            signal.lifecycle_end_reason = 'supersede'  # type: ignore
            # performance_period trigger tarafından hesaplanacak
            
            # Değişiklikleri session'a ekle
            session.add(signal)
            
            superseded_count += 1
            
            self.logger.info(
                f"Sinyal supersede edildi: {symbol} {signal.signal_type} "
                f"({signal.indicators}) -> {signal.timestamp}"
            )
        
        return superseded_count
    
    async def get_active_signals(
        self, 
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[Signal]:
        """
        Aktif sinyalleri getirir
        
        Args:
            symbol: Sembol filtresi (opsiyonel)
            interval: Interval filtresi (opsiyonel) 
            signal_type: Sinyal türü filtresi (opsiyonel)
            
        Returns:
            List[Signal]: Aktif sinyaller listesi
        """
        async with get_session() as session:
            # Base query
            stmt = select(Signal).where(Signal.status == 'active')
            
            # Filtreler
            if symbol:
                stmt = stmt.where(Signal.symbol == symbol)
            if interval:
                stmt = stmt.where(Signal.interval == interval)
            if signal_type:
                stmt = stmt.where(Signal.signal_type == signal_type)
            
            # Timestamp'e göre sırala (en yeni önce)
            stmt = stmt.order_by(Signal.timestamp.desc())
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_superseded_signals(
        self, 
        symbol: Optional[str] = None,
        hours_back: int = 24
    ) -> List[Signal]:
        """
        Supersede edilmiş sinyalleri getirir
        
        Args:
            symbol: Sembol filtresi (opsiyonel)
            hours_back: Kaç saat geriye bakılacak
            
        Returns:
            List[Signal]: Supersede edilmiş sinyaller
        """
        async with get_session() as session:
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            stmt = select(Signal).where(
                and_(
                    Signal.status == 'superseded',
                    Signal.superseded_at >= cutoff_time
                )
            )
            
            if symbol:
                stmt = stmt.where(Signal.symbol == symbol)
            
            stmt = stmt.order_by(Signal.superseded_at.desc())
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def manual_close_signal(
        self, 
        signal_id: int, 
        reason: str = 'manual'
    ) -> bool:
        """
        Sinyali manuel olarak kapatır
        
        Args:
            signal_id: Sinyal ID'si
            reason: Kapatma nedeni
            
        Returns:
            bool: İşlem başarılı mı
        """
        async with get_session() as session:
            try:
                stmt = update(Signal).where(
                    and_(
                        Signal.id == signal_id,
                        Signal.status == 'active'
                    )
                ).values(
                    status='completed',
                    superseded_at=datetime.now(),
                    lifecycle_end_reason=reason
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                if result.rowcount > 0:
                    self.logger.info(f"Sinyal manuel kapatıldı: ID {signal_id}, neden: {reason}")
                    return True
                else:
                    self.logger.warning(f"Kapatılacak aktif sinyal bulunamadı: ID {signal_id}")
                    return False
                    
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Sinyal kapatma hatası: {e}", exc_info=True)
                return False
    
    async def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Sinyal istatistiklerini getirir
        
        Returns:
            Dict: İstatistik verileri
        """
        async with get_session() as session:
            from sqlalchemy import func
            
            # Aktif sinyal sayısı
            active_count = await session.scalar(
                select(func.count(Signal.id)).where(Signal.status == 'active')
            )
            
            # Superseded sinyal sayısı (son 24 saat)
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=24)
            
            superseded_count = await session.scalar(
                select(func.count(Signal.id)).where(
                    and_(
                        Signal.status == 'superseded',
                        Signal.superseded_at >= cutoff
                    )
                )
            )
            
            # Ortalama performance period (superseded sinyaller için)
            avg_performance_period = await session.scalar(
                select(func.avg(Signal.performance_period)).where(
                    and_(
                        Signal.status == 'superseded',
                        Signal.performance_period.isnot(None)
                    )
                )
            )
            
            return {
                'active_signals': active_count or 0,
                'superseded_24h': superseded_count or 0,
                'avg_performance_period_minutes': round(avg_performance_period or 0, 2)
            }

# Global instance
signal_lifecycle_manager = SignalLifecycleManager()
