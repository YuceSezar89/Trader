"""
Database modelleri - Backtest ve Paper Trading için veri yapıları
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.types import DECIMAL
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal

class Base(DeclarativeBase):
    pass


class PaperTrade(Base):
    """Paper trading işlemleri"""
    __tablename__ = 'paper_trades'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # 'BUY' veya 'SELL'
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(18, 8), nullable=False)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 8), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(18, 8), nullable=False)
    pnl: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 8), nullable=True)
    pnl_percentage: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    
    # Risk yönetimi
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 8), nullable=True)
    take_profit: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 8), nullable=True)
    
    # Sinyal bilgileri
    signal_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vpm_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    signal_strength: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Zaman bilgileri
    entry_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Durum
    status: Mapped[str] = mapped_column(String(20), default='OPEN')  # 'OPEN', 'CLOSED', 'CANCELLED'
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    def __repr__(self):
        return f"<PaperTrade(symbol={self.symbol}, side={self.side}, pnl={self.pnl})>"


class BacktestResult(Base):
    """Backtest sonuçları"""
    __tablename__ = 'backtest_results'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Test parametreleri
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    initial_balance: Mapped[Decimal] = mapped_column(DECIMAL(18, 2), default=10000)
    
    # Performans metrikleri
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 2), nullable=True)
    
    # Finansal metrikler
    total_pnl: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 2), nullable=True)
    total_pnl_percentage: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    profit_factor: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    
    # Risk metrikleri
    avg_win: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 2), nullable=True)
    avg_loss: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(18, 2), nullable=True)
    max_consecutive_losses: Mapped[int] = mapped_column(Integer, default=0)
    max_consecutive_wins: Mapped[int] = mapped_column(Integer, default=0)
    
    # Test ayarları
    commission_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), default=0.001)  # %0.1
    slippage_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), default=0.0005)   # %0.05
    
    # Zaman bilgileri
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    test_duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # İlişkiler
    trades = relationship("BacktestTrade", back_populates="backtest_result")
    
    def __repr__(self):
        return f"<BacktestResult(strategy={self.strategy_name}, win_rate={self.win_rate}%)>"


class BacktestTrade(Base):
    """Backtest sırasında yapılan işlemler"""
    __tablename__ = 'backtest_trades'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_result_id: Mapped[int] = mapped_column(Integer, ForeignKey('backtest_results.id'), nullable=False)
    
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(18, 8), nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(DECIMAL(18, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(18, 8), nullable=False)
    
    # P&L hesaplamaları
    pnl: Mapped[Decimal] = mapped_column(DECIMAL(18, 2), nullable=False)
    pnl_percentage: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    commission: Mapped[Decimal] = mapped_column(DECIMAL(18, 2), nullable=False)
    
    # Sinyal bilgileri
    vpm_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)
    signal_strength: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Zaman bilgileri
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    hold_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Exit nedeni
    exit_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'TAKE_PROFIT', 'STOP_LOSS', 'SIGNAL'
    
    # İlişkiler
    backtest_result = relationship("BacktestResult", back_populates="trades")
    
    def __repr__(self):
        return f"<BacktestTrade(symbol={self.symbol}, pnl={self.pnl})>"


class StrategyParameter(Base):
    """Strateji test parametreleri"""
    __tablename__ = 'strategy_parameters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    parameter_name = Column(String(50), nullable=False)
    parameter_value = Column(String(100), nullable=False)
    parameter_type = Column(String(20), nullable=False)  # 'float', 'int', 'string', 'bool'
    
    # Test sonucu referansı
    backtest_result_id = Column(Integer, ForeignKey('backtest_results.id'), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<StrategyParameter({self.parameter_name}={self.parameter_value})>"


class PaperTradingSession(Base):
    """Paper trading oturumları"""
    __tablename__ = 'paper_trading_sessions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_name: Mapped[str] = mapped_column(String(100), nullable=False)
    initial_balance: Mapped[Decimal] = mapped_column(DECIMAL(18, 2), default=10000)
    current_balance: Mapped[Decimal] = mapped_column(DECIMAL(18, 2), default=10000)
    
    # Durum
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Risk parametreleri
    max_position_size: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=5.0)  # %5
    stop_loss_percentage: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=2.0)  # %2
    take_profit_percentage: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=5.0)  # %5
    
    # İstatistikler
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    current_drawdown: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), default=0.0)
    max_drawdown: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), default=0.0)
    
    def __repr__(self):
        return f"<PaperTradingSession(name={self.session_name}, balance={self.current_balance})>"
