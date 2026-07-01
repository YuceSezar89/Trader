"""
Paper Trading Sistemi - Gerçek zamanlı sanal işlem yapma
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from decimal import Decimal
import logging
import json

from database.engine import get_session
from .models import PaperTrade, PaperTradingSession
from .risk_manager import RiskManager
from .analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Gerçek zamanlı paper trading sistemi
    - Canlı sinyalleri takip eder
    - Sanal işlemler yapar
    - Risk yönetimi uygular
    - Performans takibi yapar
    """
    
    def __init__(self, session_name: str = "Default Session", initial_balance: float = 10000.0):
        self.session_name = session_name
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = self.initial_balance
        
        # Modüller
        self.risk_manager = RiskManager()
        self.analyzer = PerformanceAnalyzer()
        
        # Durum takibi
        self.is_active = False
        self.positions: Dict[str, Dict] = {}  # Açık pozisyonlar {symbol: position_data}
        self.session_id: Optional[int] = None
        self.last_signal_check = datetime.now()
        
        # Callback fonksiyonları
        self.on_trade_opened: Optional[Callable] = None
        self.on_trade_closed: Optional[Callable] = None
        self.on_balance_updated: Optional[Callable] = None
        
        # Paper trading parametreleri
        self.signal_check_interval = 30  # Saniye
        self.max_signal_age = 300       # 5 dakika (saniye)
    
    async def start_session(self) -> int:
        """
        Paper trading oturumu başlat
        
        Returns:
            int: Session ID
        """
        try:
            async with get_session() as session:
                # Yeni oturum oluştur
                trading_session = PaperTradingSession(
                    session_name=self.session_name,
                    initial_balance=self.initial_balance,
                    current_balance=self.current_balance,
                    is_active=True,
                    max_position_size=Decimal(str(self.risk_manager.max_position_size_percentage)),
                    stop_loss_percentage=Decimal(str(self.risk_manager.stop_loss_percentage)),
                    take_profit_percentage=Decimal(str(self.risk_manager.take_profit_percentage))
                )
                
                session.add(trading_session)
                await session.commit()
                await session.refresh(trading_session)
                
                self.session_id = trading_session.id
                self.is_active = True
                
                logger.info(f"Paper trading oturumu başlatıldı: {self.session_name} (ID: {self.session_id})")
                return self.session_id
                
        except Exception as e:
            logger.error(f"Oturum başlatma hatası: {e}")
            raise
    
    async def stop_session(self):
        """Paper trading oturumunu durdur"""
        try:
            if not self.session_id:
                return
            
            # Tüm açık pozisyonları kapat
            await self._close_all_positions("SESSION_ENDED")
            
            # Oturum durumunu güncelle
            async with get_session() as session:
                trading_session = await session.get(PaperTradingSession, self.session_id)
                if trading_session:
                    trading_session.is_active = False
                    trading_session.end_time = datetime.now()
                    trading_session.current_balance = self.current_balance
                    await session.commit()
            
            self.is_active = False
            logger.info(f"Paper trading oturumu durduruldu: {self.session_name}")
            
        except Exception as e:
            logger.error(f"Oturum durdurma hatası: {e}")
    
    async def run_trading_loop(self):
        """Ana trading döngüsü - sürekli çalışır"""
        logger.info("Paper trading döngüsü başlatıldı")
        
        while self.is_active:
            try:
                # Yeni sinyalleri kontrol et
                await self._check_new_signals()
                
                # Mevcut pozisyonları kontrol et
                await self._monitor_positions()
                
                # Risk limitlerini kontrol et
                await self._check_risk_limits()
                
                # Oturum istatistiklerini güncelle
                await self._update_session_stats()
                
                # Bekleme
                await asyncio.sleep(self.signal_check_interval)
                
            except Exception as e:
                logger.error(f"Trading döngüsü hatası: {e}")
                await asyncio.sleep(60)  # Hata durumunda 1 dakika bekle
        
        logger.info("Paper trading döngüsü durduruldu")
    
    async def _check_new_signals(self):
        """Yeni sinyalleri kontrol et ve işle"""
        try:
            from database.crud import get_recent_signals
            async with get_session() as session:
                signals = await get_recent_signals(hours=1)  # Son 1 saatteki sinyaller
                
                for signal in signals:
                    await self._process_signal(signal)
                
                self.last_signal_check = datetime.now()
                
        except Exception as e:
            logger.error(f"Sinyal kontrol hatası: {e}")
    
    async def _process_signal(self, signal):
        """Sinyali işle ve pozisyon aç"""
        try:
            symbol = signal.symbol
            signal_type = signal.signal_type
            
            # Zaten pozisyon var mı?
            if symbol in self.positions:
                logger.debug(f"Zaten pozisyon var: {symbol}")
                return
            
            # Sinyal yaşı kontrolü
            signal_age = (datetime.now() - signal.timestamp).total_seconds()
            if signal_age > self.max_signal_age:
                logger.debug(f"Sinyal çok eski: {symbol} ({signal_age:.0f}s)")
                return
            
            # Güncel fiyat al (son kline verisinden)
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"Fiyat alınamadı: {symbol}")
                return
            
            # Risk kontrolü
            risk_checks = self.risk_manager.check_risk_limits(
                self.current_balance,
                Decimal(str(current_price * 100))  # Örnek trade değeri
            )
            
            if not risk_checks['overall_risk_ok']:
                logger.info(f"Risk limitleri nedeniyle sinyal reddedildi: {symbol}")
                return
            
            # Pozisyon büyüklüğü hesapla
            position_size = self.risk_manager.calculate_position_size(
                self.current_balance, current_price
            )
            
            if position_size <= 0:
                logger.debug(f"Pozisyon büyüklüğü çok küçük: {symbol}")
                return
            
            # Pozisyon aç
            await self._open_position(signal, current_price, position_size)
            
        except Exception as e:
            logger.error(f"Sinyal işleme hatası: {e}")
    
    async def _open_position(self, signal, entry_price: float, quantity: Decimal):
        """Pozisyon aç"""
        try:
            symbol = signal.symbol
            side = signal.signal_type
            
            # Stop loss ve take profit hesapla
            stop_loss = self.risk_manager.calculate_stop_loss(entry_price, side)
            take_profit = self.risk_manager.calculate_take_profit(entry_price, side)
            
            # Paper trade kaydı oluştur
            paper_trade = PaperTrade(
                symbol=symbol,
                side=side,
                entry_price=Decimal(str(entry_price)),
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_id=signal.id,
                vmp_score=signal.vmp_score,
                signal_strength=signal.signal_strength,
                timeframe=signal.timeframe,
                status='OPEN'
            )
            
            # Veritabanına kaydet
            async with get_session() as session:
                session.add(paper_trade)
                await session.commit()
                await session.refresh(paper_trade)
            
            # Pozisyonu takip listesine ekle
            self.positions[symbol] = {
                'trade_id': paper_trade.id,
                'side': side,
                'entry_price': entry_price,
                'quantity': float(quantity),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'entry_time': paper_trade.entry_time,
                'signal_id': signal.id
            }
            
            # Risk manager'ı güncelle
            self.risk_manager.update_position_count(1)
            
            logger.info(f"Pozisyon açıldı: {symbol} {side} @ {entry_price} (Qty: {quantity})")
            
            # Callback çağır
            if self.on_trade_opened:
                await self.on_trade_opened(paper_trade)
                
        except Exception as e:
            logger.error(f"Pozisyon açma hatası: {e}")
    
    async def _monitor_positions(self):
        """Açık pozisyonları izle"""
        for symbol in list(self.positions.keys()):
            try:
                await self._check_position_exit(symbol)
            except Exception as e:
                logger.error(f"Pozisyon izleme hatası {symbol}: {e}")
    
    async def _check_position_exit(self, symbol: str):
        """Pozisyon çıkış koşullarını kontrol et"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = await self._get_current_price(symbol)
        
        if not current_price:
            return
        
        # Stop loss / Take profit kontrolü
        should_close = False
        exit_reason = None
        
        if position['side'] == 'BUY':
            if current_price <= position['stop_loss']:
                should_close = True
                exit_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                should_close = True
                exit_reason = 'TAKE_PROFIT'
        else:  # SELL
            if current_price >= position['stop_loss']:
                should_close = True
                exit_reason = 'STOP_LOSS'
            elif current_price <= position['take_profit']:
                should_close = True
                exit_reason = 'TAKE_PROFIT'
        
        if should_close:
            await self._close_position(symbol, current_price, exit_reason)
    
    async def _close_position(self, symbol: str, exit_price: float, exit_reason: Optional[str] = None):
        """Pozisyon kapat"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            trade_id = position['trade_id']
            
            # P&L hesapla
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            if position['side'] == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
            
            pnl_percentage = (pnl / (entry_price * quantity)) * 100
            
            # Bakiye güncelle
            self.current_balance += Decimal(str(pnl))
            
            # Veritabanını güncelle
            async with get_session() as session:
                paper_trade = await session.get(PaperTrade, trade_id)
                if paper_trade:
                    paper_trade.exit_price = Decimal(str(exit_price))
                    paper_trade.exit_time = datetime.now()
                    paper_trade.pnl = Decimal(str(pnl))
                    paper_trade.pnl_percentage = Decimal(str(pnl_percentage))
                    paper_trade.status = 'CLOSED'
                    paper_trade.notes = exit_reason
                    await session.commit()
            
            # Pozisyonu listeden çıkar
            del self.positions[symbol]
            
            # Risk manager'ı güncelle
            self.risk_manager.update_position_count(-1)
            self.risk_manager.update_daily_pnl(Decimal(str(pnl)))
            
            logger.info(f"Pozisyon kapandı: {symbol} P&L: {pnl:.2f} ({exit_reason})")
            
            # Callback çağır
            if self.on_trade_closed:
                await self.on_trade_closed(paper_trade)
            
            # Bakiye callback'i
            if self.on_balance_updated:
                await self.on_balance_updated(float(self.current_balance))
                
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası {symbol}: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Güncel fiyat al"""
        try:
            # Bu fonksiyon gerçek implementasyonda WebSocket'ten veya API'den fiyat alacak
            # Şimdilik basit bir fiyat simülasyonu yapıyoruz
            import random
            
            # Basit fiyat simülasyonu (gerçek implementasyonda kaldırılacak)
            base_prices = {
                'BTCUSDT': 50000,
                'ETHUSDT': 3000,
                'BNBUSDT': 300
            }
            
            base_price = base_prices.get(symbol, 100)
            # %1 rastgele değişim
            price_variation = random.uniform(-0.01, 0.01)
            current_price = base_price * (1 + price_variation)
            
            return current_price
                    
        except Exception as e:
            logger.error(f"Fiyat alma hatası {symbol}: {e}")
            
        return None
    
    async def _close_all_positions(self, reason: str = "MANUAL_CLOSE"):
        """Tüm açık pozisyonları kapat"""
        try:
            for symbol in list(self.positions.keys()):
                current_price = await self._get_current_price(symbol)
                if current_price:
                    await self._close_position(symbol, current_price, reason)
                    
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {e}")
    
    async def _update_session_stats(self):
        """Oturum istatistiklerini güncelle"""
        try:
            if not self.session_id:
                return
            
            async with get_session() as session:
                trading_session = await session.get(PaperTradingSession, self.session_id)
                if trading_session:
                    trading_session.current_balance = self.current_balance
                    trading_session.current_drawdown = self._calculate_current_drawdown()
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"İstatistik güncelleme hatası: {e}")
    
    def _calculate_current_drawdown(self) -> Decimal:
        """Mevcut drawdown hesapla"""
        if self.current_balance >= self.initial_balance:
            return Decimal('0')
        
        drawdown = ((self.initial_balance - self.current_balance) / self.initial_balance) * 100
        return drawdown
    
    async def get_session_summary(self) -> Dict:
        """Oturum özetini al"""
        try:
            async with get_session() as session:
                # Tamamlanan işlemler
                completed_trades = await session.execute(
                    "SELECT * FROM paper_trades WHERE status = 'CLOSED' ORDER BY exit_time DESC"
                )
                trades = completed_trades.fetchall()
                
                # İstatistikler hesapla
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.pnl > 0])
                total_pnl = sum(t.pnl for t in trades) if trades else 0
                
                return {
                    'session_name': self.session_name,
                    'session_id': self.session_id,
                    'initial_balance': float(self.initial_balance),
                    'current_balance': float(self.current_balance),
                    'total_pnl': float(total_pnl),
                    'total_pnl_percentage': float((total_pnl / self.initial_balance) * 100),
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'open_positions': len(self.positions),
                    'current_drawdown': float(self._calculate_current_drawdown()),
                    'is_active': self.is_active
                }
                
        except Exception as e:
            logger.error(f"Oturum özeti hatası: {e}")
            return {}
    
    def set_callbacks(
        self,
        on_trade_opened: Optional[Callable] = None,
        on_trade_closed: Optional[Callable] = None,
        on_balance_updated: Optional[Callable] = None
    ):
        """Callback fonksiyonlarını ayarla"""
        self.on_trade_opened = on_trade_opened
        self.on_trade_closed = on_trade_closed
        self.on_balance_updated = on_balance_updated


class PaperTradingManager:
    """
    Çoklu paper trading oturumlarını yönetir
    """
    
    def __init__(self):
        self.active_sessions = {}  # {session_id: PaperTrader}
    
    async def create_session(self, session_name: str, initial_balance: float = 10000.0) -> int:
        """Yeni paper trading oturumu oluştur"""
        trader = PaperTrader(session_name, initial_balance)
        session_id = await trader.start_session()
        self.active_sessions[session_id] = trader
        
        # Arka planda trading döngüsünü başlat
        asyncio.create_task(trader.run_trading_loop())
        
        return session_id
    
    async def stop_session(self, session_id: int):
        """Oturumu durdur"""
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].stop_session()
            del self.active_sessions[session_id]
    
    async def get_session(self, session_id: int) -> Optional[PaperTrader]:
        """Oturumu al"""
        return self.active_sessions.get(session_id)
    
    async def list_active_sessions(self) -> List[Dict]:
        """Aktif oturumları listele"""
        summaries = []
        for session_id, trader in self.active_sessions.items():
            summary = await trader.get_session_summary()
            summaries.append(summary)
        return summaries
