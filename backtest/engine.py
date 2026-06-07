"""
Ana Backtest Motoru - VP sisteminin tarihsel performansını test eder
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import logging

from database.engine import get_session
# from database.crud import get_kline_data  # TODO: Bu fonksiyon mevcut değil
from signals.signal_engine import SignalEngine
from indicators.financial_metrics import calculate_metrics
from .models import BacktestResult, BacktestTrade, StrategyParameter
from .risk_manager import RiskManager
from .analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Ana backtest motoru - VP sisteminin tarihsel performansını test eder
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = self.initial_balance
        self.positions: Dict[str, Dict] = {}  # Açık pozisyonlar
        self.trades: List[Dict] = []     # Tamamlanan işlemler
        
        # Modüller
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.analyzer = PerformanceAnalyzer()
        
        # Test parametreleri
        self.commission_rate = Decimal('0.001')  # %0.1
        self.slippage_rate = Decimal('0.0005')   # %0.05
        
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1h',
        strategy_name: str = 'VP_System',
        strategy_params: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Ana backtest fonksiyonu
        
        Args:
            symbols: Test edilecek semboller
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            timeframe: Zaman dilimi (1h, 4h, 1d)
            strategy_name: Strateji adı
            strategy_params: Strateji parametreleri
        
        Returns:
            BacktestResult: Test sonuçları
        """
        logger.info(f"Backtest başlatılıyor: {strategy_name} ({start_date} - {end_date})")
        start_time = datetime.utcnow()
        
        # Parametreleri ayarla
        if strategy_params:
            self._apply_strategy_params(strategy_params)
        
        # Her sembol için test yap
        for symbol in symbols:
            await self._test_symbol(symbol, start_date, end_date, timeframe)
        
        # Sonuçları analiz et
        results = await self._analyze_results(
            strategy_name, timeframe, start_date, end_date, start_time
        )
        
        # Veritabanına kaydet
        await self._save_results(results, strategy_params)
        
        logger.info(f"Backtest tamamlandı: {len(self.trades)} işlem, Win Rate: {results.win_rate}%")
        return results
    
    async def _test_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ):
        """Tek sembol için backtest"""
        logger.info(f"Testing {symbol} ({timeframe})")
        
        # Tarihsel veri çek
        kline_data = await self._get_historical_data(symbol, start_date, end_date, timeframe)
        
        if not kline_data:
            logger.warning(f"Veri bulunamadı: {symbol}")
            return
        
        # Her bar için sinyal kontrolü
        for i in range(len(kline_data)):
            current_bar = kline_data.iloc[i]
            current_time = current_bar['timestamp']
            current_price = float(current_bar['close'])
            
            # Mevcut pozisyonları kontrol et
            await self._check_positions(symbol, current_price, current_time)
            
            # Yeni sinyal kontrol et (en az 20 bar geçmiş veri gerekli)
            if i >= 20:
                signal = await self._generate_signal(
                    symbol, kline_data.iloc[:i+1], current_time, timeframe
                )
                
                if signal:
                    await self._process_signal(signal, current_price, current_time)
    
    async def _get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Veritabanından tarihsel veri çek"""
        try:
            # TODO: get_kline_data fonksiyonu implement edilmeli
            logger.warning(f"Historical data for {symbol} not implemented yet")
            return None
                
        except Exception as e:
            logger.error(f"Veri çekme hatası {symbol}: {e}")
            return None
    
    async def _generate_signal(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        timestamp: datetime,
        timeframe: str
    ) -> Optional[Dict]:
        """VP sistemi ile sinyal üret"""
        try:
            # Son 20 bar ile finansal metrikleri hesapla
            recent_data = historical_data.tail(20)
            
            # Finansal metrikler
            metrics = calculate_metrics(recent_data)
            
            # VP skoru hesapla (basitleştirilmiş)
            vmp_score = self._calculate_vmp_score(recent_data, metrics)
            
            # Sinyal kontrolü
            if abs(vmp_score) >= 0.5:  # Minimum eşik
                signal_type = 'BUY' if vmp_score > 0 else 'SELL'
                signal_strength = min(5, int(abs(vmp_score) * 2))
                
                return {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'vmp_score': vmp_score,
                    'signal_strength': signal_strength,
                    'timestamp': timestamp,
                    'timeframe': timeframe
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Sinyal üretme hatası {symbol}: {e}")
            return None
    
    def _calculate_vmp_score(self, data: pd.DataFrame, metrics: Dict) -> float:
        """Basitleştirilmiş VMP skoru hesaplama"""
        try:
            # Fiyat momentum (son 5 bar)
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6]
            
            # Hacim momentum
            avg_volume = data['volume'].tail(10).mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Basit VMP skoru
            vmp_score = (price_change * 10) + (volume_ratio - 1) * 2
            
            return float(vmp_score)
            
        except Exception as e:
            logger.error(f"VMP hesaplama hatası: {e}")
            return 0.0
    
    async def _process_signal(self, signal: Dict, current_price: float, timestamp: datetime):
        """Sinyal işle ve pozisyon aç"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        
        # Zaten pozisyon var mı kontrol et
        if symbol in self.positions:
            return
        
        # Risk yönetimi kontrolü
        position_size = self.risk_manager.calculate_position_size(
            self.current_balance, current_price
        )
        
        if position_size <= 0:
            return
        
        # Pozisyon aç
        position = {
            'symbol': symbol,
            'side': signal_type,
            'entry_price': Decimal(str(current_price)),
            'quantity': position_size,
            'entry_time': timestamp,
            'vmp_score': signal['vmp_score'],
            'signal_strength': signal['signal_strength'],
            'stop_loss': self.risk_manager.calculate_stop_loss(current_price, signal_type),
            'take_profit': self.risk_manager.calculate_take_profit(current_price, signal_type)
        }
        
        self.positions[symbol] = position
        logger.info(f"Pozisyon açıldı: {symbol} {signal_type} @ {current_price}")
    
    async def _check_positions(self, symbol: str, current_price: float, timestamp: datetime):
        """Açık pozisyonları kontrol et"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = float(position['entry_price'])
        
        # Stop loss / Take profit kontrolü
        should_close = False
        exit_reason = None
        
        if position['side'] == 'BUY':
            if current_price <= float(position['stop_loss']):
                should_close = True
                exit_reason = 'STOP_LOSS'
            elif current_price >= float(position['take_profit']):
                should_close = True
                exit_reason = 'TAKE_PROFIT'
        else:  # SELL
            if current_price >= float(position['stop_loss']):
                should_close = True
                exit_reason = 'STOP_LOSS'
            elif current_price <= float(position['take_profit']):
                should_close = True
                exit_reason = 'TAKE_PROFIT'
        
        if should_close:
            await self._close_position(symbol, current_price, timestamp, exit_reason)
    
    async def _close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        exit_reason: Optional[str] = None
    ):
        """Pozisyon kapat"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = float(position['entry_price'])
        quantity = float(position['quantity'])
        
        # P&L hesapla
        if position['side'] == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
        
        # Commission hesapla
        commission = (entry_price + exit_price) * quantity * float(self.commission_rate)
        net_pnl = pnl - commission
        
        # Bakiye güncelle
        self.current_balance += Decimal(str(net_pnl))
        
        # Trade kaydı oluştur
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': net_pnl,
            'pnl_percentage': (net_pnl / (entry_price * quantity)) * 100,
            'commission': commission,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'exit_reason': exit_reason,
            'vmp_score': position['vmp_score'],
            'signal_strength': position['signal_strength']
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        logger.info(f"Pozisyon kapandı: {symbol} P&L: {net_pnl:.2f} ({exit_reason})")
    
    async def _analyze_results(
        self,
        strategy_name: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        start_time: datetime
    ) -> BacktestResult:
        """Test sonuçlarını analiz et"""
        
        if not self.trades:
            # Hiç işlem yoksa boş sonuç döndür
            return BacktestResult(
                strategy_name=strategy_name,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self.initial_balance,
                total_trades=0,
                win_rate=Decimal('0'),
                total_pnl=Decimal('0'),
                test_duration_seconds=int((datetime.utcnow() - start_time).total_seconds())
            )
        
        # Temel istatistikler
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L hesaplamaları
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pnl_percentage = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Risk metrikleri
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # Drawdown hesapla
        max_drawdown = self.analyzer.calculate_max_drawdown([t['pnl'] for t in self.trades])
        
        # Sharpe ratio (basitleştirilmiş)
        returns = [t['pnl_percentage'] for t in self.trades]
        sharpe_ratio = self.analyzer.calculate_sharpe_ratio(returns)
        
        return BacktestResult(
            strategy_name=strategy_name,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=Decimal(str(round(win_rate, 2))),
            total_pnl=Decimal(str(round(total_pnl, 2))),
            total_pnl_percentage=Decimal(str(round(total_pnl_percentage, 4))),
            max_drawdown=Decimal(str(round(max_drawdown, 4))),
            sharpe_ratio=Decimal(str(round(sharpe_ratio, 4))),
            profit_factor=Decimal(str(round(profit_factor, 4))),
            avg_win=Decimal(str(round(avg_win, 2))),
            avg_loss=Decimal(str(round(avg_loss, 2))),
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            test_duration_seconds=int((datetime.utcnow() - start_time).total_seconds())
        )
    
    async def _save_results(self, result: BacktestResult, strategy_params: Optional[Dict]):
        """Sonuçları veritabanına kaydet"""
        try:
            async with get_session() as session:
                # Ana sonucu kaydet
                session.add(result)
                await session.flush()  # ID almak için
                
                # Trade'leri kaydet
                for trade_data in self.trades:
                    trade = BacktestTrade(
                        backtest_result_id=result.id,
                        symbol=trade_data['symbol'],
                        side=trade_data['side'],
                        entry_price=Decimal(str(trade_data['entry_price'])),
                        exit_price=Decimal(str(trade_data['exit_price'])),
                        quantity=Decimal(str(trade_data['quantity'])),
                        pnl=Decimal(str(trade_data['pnl'])),
                        pnl_percentage=Decimal(str(trade_data['pnl_percentage'])),
                        commission=Decimal(str(trade_data['commission'])),
                        entry_time=trade_data['entry_time'],
                        exit_time=trade_data['exit_time'],
                        exit_reason=trade_data['exit_reason'],
                        vmp_score=Decimal(str(trade_data['vmp_score'])),
                        signal_strength=trade_data['signal_strength']
                    )
                    session.add(trade)
                
                # Parametreleri kaydet
                if strategy_params:
                    for param_name, param_value in strategy_params.items():
                        param = StrategyParameter(
                            strategy_name=result.strategy_name,
                            parameter_name=param_name,
                            parameter_value=str(param_value),
                            parameter_type=type(param_value).__name__,
                            backtest_result_id=result.id
                        )
                        session.add(param)
                
                await session.commit()
                logger.info(f"Backtest sonuçları kaydedildi: ID {result.id}")
                
        except Exception as e:
            logger.error(f"Sonuç kaydetme hatası: {e}")
    
    def _apply_strategy_params(self, params: Dict):
        """Strateji parametrelerini uygula"""
        if 'commission_rate' in params:
            self.commission_rate = Decimal(str(params['commission_rate']))
        
        if 'slippage_rate' in params:
            self.slippage_rate = Decimal(str(params['slippage_rate']))
        
        # Risk manager parametreleri
        if 'stop_loss_percentage' in params:
            self.risk_manager.stop_loss_percentage = params['stop_loss_percentage']
        
        if 'take_profit_percentage' in params:
            self.risk_manager.take_profit_percentage = params['take_profit_percentage']
    
    def reset(self):
        """Backtest durumunu sıfırla"""
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
