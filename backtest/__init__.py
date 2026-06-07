"""
Backtest & Paper Trading Module

Bu modül VP sisteminin performansını test etmek ve optimize etmek için kullanılır.

Modüller:
- engine: Ana backtest motoru
- paper_trader: Gerçek zamanlı paper trading
- risk_manager: Risk yönetimi sistemi
- analyzer: Performans analizi
- strategy_tester: Strateji optimizasyonu
- models: Database modelleri
"""

from .engine import BacktestEngine
from .paper_trader import PaperTrader
from .risk_manager import RiskManager
from .analyzer import PerformanceAnalyzer
from .strategy_tester import StrategyTester

__version__ = "1.0.0"
__author__ = "TRader Panel"

__all__ = [
    "BacktestEngine",
    "PaperTrader", 
    "RiskManager",
    "PerformanceAnalyzer",
    "StrategyTester"
]
