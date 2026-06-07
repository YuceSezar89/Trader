"""
Performans Analizi Modülü - Backtest ve Paper Trading sonuçlarını analiz eder
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import logging
import math

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Performans analizi ve metrik hesaplamaları
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # %2 risksiz getiri (yıllık)
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: Optional[float] = None) -> float:
        """
        Sharpe oranı hesapla
        
        Args:
            returns: Getiri listesi (yüzde olarak)
            risk_free_rate: Risksiz getiri oranı
        
        Returns:
            float: Sharpe oranı
        """
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            rf_rate = risk_free_rate or self.risk_free_rate
            
            # Günlük getirileri yıllık hale getir
            annual_return = np.mean(returns) * 252  # 252 trading days
            annual_volatility = np.std(returns) * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
            
            sharpe = (annual_return - rf_rate) / annual_volatility
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Sharpe ratio hesaplama hatası: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """
        Maksimum drawdown hesapla
        
        Args:
            pnl_series: P&L serisi
        
        Returns:
            float: Maksimum drawdown (yüzde)
        """
        try:
            if not pnl_series:
                return 0.0
            
            # Kümülatif P&L hesapla
            cumulative_pnl = np.cumsum(pnl_series)
            
            # Running maximum
            running_max = np.maximum.accumulate(cumulative_pnl)
            
            # Drawdown hesapla
            drawdown = (cumulative_pnl - running_max) / running_max * 100
            
            # Maksimum drawdown
            max_dd = abs(np.min(drawdown))
            
            return float(max_dd)
            
        except Exception as e:
            logger.error(f"Max drawdown hesaplama hatası: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: List[float], target_return: float = 0.0) -> float:
        """
        Sortino oranı hesapla (sadece negatif volatilite)
        
        Args:
            returns: Getiri listesi
            target_return: Hedef getiri
        
        Returns:
            float: Sortino oranı
        """
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - target_return
            
            # Sadece negatif getirilerin volatilitesi
            negative_returns = excess_returns[excess_returns < 0]
            
            if len(negative_returns) == 0:
                return float('inf')  # Hiç negatif getiri yok
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = np.mean(excess_returns) / downside_deviation
            return float(sortino)
            
        except Exception as e:
            logger.error(f"Sortino ratio hesaplama hatası: {e}")
            return 0.0
    
    def calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Calmar oranı hesapla (Annual Return / Max Drawdown)
        
        Args:
            total_return: Toplam getiri (yüzde)
            max_drawdown: Maksimum drawdown (yüzde)
        
        Returns:
            float: Calmar oranı
        """
        try:
            if max_drawdown == 0:
                return float('inf') if total_return > 0 else 0.0
            
            calmar = total_return / max_drawdown
            return float(calmar)
            
        except Exception as e:
            logger.error(f"Calmar ratio hesaplama hatası: {e}")
            return 0.0
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Kazanma oranı hesapla
        
        Args:
            trades: Trade listesi
        
        Returns:
            float: Kazanma oranı (yüzde)
        """
        try:
            if not trades:
                return 0.0
            
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            total_trades = len(trades)
            
            win_rate = (winning_trades / total_trades) * 100
            return float(win_rate)
            
        except Exception as e:
            logger.error(f"Win rate hesaplama hatası: {e}")
            return 0.0
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Profit factor hesapla (Gross Profit / Gross Loss)
        
        Args:
            trades: Trade listesi
        
        Returns:
            float: Profit factor
        """
        try:
            if not trades:
                return 0.0
            
            gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            profit_factor = gross_profit / gross_loss
            return float(profit_factor)
            
        except Exception as e:
            logger.error(f"Profit factor hesaplama hatası: {e}")
            return 0.0
    
    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """
        Expectancy hesapla (ortalama kazanç per trade)
        
        Args:
            trades: Trade listesi
        
        Returns:
            float: Expectancy
        """
        try:
            if not trades:
                return 0.0
            
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            expectancy = total_pnl / len(trades)
            
            return float(expectancy)
            
        except Exception as e:
            logger.error(f"Expectancy hesaplama hatası: {e}")
            return 0.0
    
    def calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """
        Recovery factor hesapla (Net Profit / Max Drawdown)
        
        Args:
            total_return: Net kar
            max_drawdown: Maksimum drawdown
        
        Returns:
            float: Recovery factor
        """
        try:
            if max_drawdown == 0:
                return float('inf') if total_return > 0 else 0.0
            
            recovery = total_return / max_drawdown
            return float(recovery)
            
        except Exception as e:
            logger.error(f"Recovery factor hesaplama hatası: {e}")
            return 0.0
    
    def analyze_trade_distribution(self, trades: List[Dict]) -> Dict:
        """
        Trade dağılımını analiz et
        
        Args:
            trades: Trade listesi
        
        Returns:
            Dict: Dağılım analizi
        """
        try:
            if not trades:
                return {}
            
            pnl_values = [t.get('pnl', 0) for t in trades]
            winning_trades = [p for p in pnl_values if p > 0]
            losing_trades = [p for p in pnl_values if p < 0]
            
            analysis = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'break_even_trades': len(pnl_values) - len(winning_trades) - len(losing_trades),
                
                # P&L istatistikleri
                'total_pnl': sum(pnl_values),
                'avg_pnl': np.mean(pnl_values),
                'median_pnl': np.median(pnl_values),
                'std_pnl': np.std(pnl_values),
                
                # Kazanan trade'ler
                'avg_win': np.mean(winning_trades) if winning_trades else 0,
                'max_win': max(winning_trades) if winning_trades else 0,
                'min_win': min(winning_trades) if winning_trades else 0,
                
                # Kaybeden trade'ler
                'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                'max_loss': min(losing_trades) if losing_trades else 0,  # En büyük kayıp (negatif)
                'min_loss': max(losing_trades) if losing_trades else 0,
                
                # Oranlar
                'avg_win_loss_ratio': abs(np.mean(winning_trades) / np.mean(losing_trades)) if winning_trades and losing_trades else 0,
                'largest_win_loss_ratio': abs(max(winning_trades) / min(losing_trades)) if winning_trades and losing_trades else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Trade dağılım analizi hatası: {e}")
            return {}
    
    def calculate_consecutive_stats(self, trades: List[Dict]) -> Dict:
        """
        Ardışık kazanç/kayıp istatistikleri
        
        Args:
            trades: Trade listesi (zaman sıralı)
        
        Returns:
            Dict: Ardışık istatistikler
        """
        try:
            if not trades:
                return {}
            
            # P&L serisi
            pnl_series = [t.get('pnl', 0) for t in trades]
            
            # Ardışık kazanç/kayıp sayma
            current_win_streak = 0
            current_loss_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            
            win_streaks = []
            loss_streaks = []
            
            for pnl in pnl_series:
                if pnl > 0:  # Kazanç
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                elif pnl < 0:  # Kayıp
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
                # pnl == 0 durumunda streak'ler devam eder
            
            # Son streak'leri ekle
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
            
            return {
                'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
                'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
                'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
                'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
                'total_win_streaks': len(win_streaks),
                'total_loss_streaks': len(loss_streaks)
            }
            
        except Exception as e:
            logger.error(f"Ardışık istatistik hesaplama hatası: {e}")
            return {}
    
    def calculate_time_based_metrics(self, trades: List[Dict]) -> Dict:
        """
        Zaman bazlı metrikler
        
        Args:
            trades: Trade listesi
        
        Returns:
            Dict: Zaman bazlı analiz
        """
        try:
            if not trades:
                return {}
            
            # Trade sürelerini hesapla
            durations = []
            for trade in trades:
                if 'entry_time' in trade and 'exit_time' in trade:
                    entry_time = trade['entry_time']
                    exit_time = trade['exit_time']
                    
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time)
                    
                    duration = (exit_time - entry_time).total_seconds() / 3600  # Saat cinsinden
                    durations.append(duration)
            
            if not durations:
                return {}
            
            # Kazanan ve kaybeden trade süreleri
            winning_durations = []
            losing_durations = []
            
            for i, trade in enumerate(trades):
                if i < len(durations):
                    if trade.get('pnl', 0) > 0:
                        winning_durations.append(durations[i])
                    elif trade.get('pnl', 0) < 0:
                        losing_durations.append(durations[i])
            
            return {
                'avg_trade_duration_hours': np.mean(durations),
                'median_trade_duration_hours': np.median(durations),
                'max_trade_duration_hours': max(durations),
                'min_trade_duration_hours': min(durations),
                'avg_winning_duration_hours': np.mean(winning_durations) if winning_durations else 0,
                'avg_losing_duration_hours': np.mean(losing_durations) if losing_durations else 0,
                'total_trading_time_hours': sum(durations)
            }
            
        except Exception as e:
            logger.error(f"Zaman bazlı metrik hesaplama hatası: {e}")
            return {}
    
    def generate_comprehensive_report(self, trades: List[Dict], initial_balance: float = 10000) -> Dict:
        """
        Kapsamlı performans raporu oluştur
        
        Args:
            trades: Trade listesi
            initial_balance: Başlangıç bakiyesi
        
        Returns:
            Dict: Kapsamlı analiz raporu
        """
        try:
            if not trades:
                return {'error': 'No trades to analyze'}
            
            # Temel metrikler
            pnl_values = [t.get('pnl', 0) for t in trades]
            pnl_percentages = [t.get('pnl_percentage', 0) for t in trades]
            
            total_pnl = sum(pnl_values)
            total_return_percentage = (total_pnl / initial_balance) * 100
            
            # Ana metrikler
            win_rate = self.calculate_win_rate(trades)
            profit_factor = self.calculate_profit_factor(trades)
            sharpe_ratio = self.calculate_sharpe_ratio(pnl_percentages)
            sortino_ratio = self.calculate_sortino_ratio(pnl_percentages)
            max_drawdown = self.calculate_max_drawdown(pnl_values)
            calmar_ratio = self.calculate_calmar_ratio(total_return_percentage, max_drawdown)
            expectancy = self.calculate_expectancy(trades)
            
            # Detaylı analizler
            trade_distribution = self.analyze_trade_distribution(trades)
            consecutive_stats = self.calculate_consecutive_stats(trades)
            time_metrics = self.calculate_time_based_metrics(trades)
            
            # Kapsamlı rapor
            report = {
                'summary': {
                    'total_trades': len(trades),
                    'initial_balance': initial_balance,
                    'final_balance': initial_balance + total_pnl,
                    'total_pnl': total_pnl,
                    'total_return_percentage': total_return_percentage,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor
                },
                
                'risk_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'max_drawdown': max_drawdown,
                    'expectancy': expectancy
                },
                
                'trade_analysis': trade_distribution,
                'consecutive_analysis': consecutive_stats,
                'time_analysis': time_metrics,
                
                'performance_grade': self._calculate_performance_grade(
                    win_rate, profit_factor, sharpe_ratio, max_drawdown
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Kapsamlı rapor oluşturma hatası: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_grade(
        self,
        win_rate: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float
    ) -> Dict:
        """
        Performans notu hesapla
        
        Returns:
            Dict: Performans notu ve açıklama
        """
        try:
            score = 0
            
            # Win rate puanı (0-25)
            if win_rate >= 60:
                score += 25
            elif win_rate >= 50:
                score += 20
            elif win_rate >= 40:
                score += 15
            elif win_rate >= 30:
                score += 10
            else:
                score += 5
            
            # Profit factor puanı (0-25)
            if profit_factor >= 2.0:
                score += 25
            elif profit_factor >= 1.5:
                score += 20
            elif profit_factor >= 1.2:
                score += 15
            elif profit_factor >= 1.0:
                score += 10
            else:
                score += 0
            
            # Sharpe ratio puanı (0-25)
            if sharpe_ratio >= 2.0:
                score += 25
            elif sharpe_ratio >= 1.5:
                score += 20
            elif sharpe_ratio >= 1.0:
                score += 15
            elif sharpe_ratio >= 0.5:
                score += 10
            else:
                score += 5
            
            # Max drawdown puanı (0-25)
            if max_drawdown <= 5:
                score += 25
            elif max_drawdown <= 10:
                score += 20
            elif max_drawdown <= 15:
                score += 15
            elif max_drawdown <= 20:
                score += 10
            else:
                score += 5
            
            # Not ve açıklama
            if score >= 90:
                grade = 'A+'
                description = 'Mükemmel performans'
            elif score >= 80:
                grade = 'A'
                description = 'Çok iyi performans'
            elif score >= 70:
                grade = 'B+'
                description = 'İyi performans'
            elif score >= 60:
                grade = 'B'
                description = 'Orta performans'
            elif score >= 50:
                grade = 'C+'
                description = 'Zayıf performans'
            elif score >= 40:
                grade = 'C'
                description = 'Çok zayıf performans'
            else:
                grade = 'D'
                description = 'Kötü performans'
            
            return {
                'score': score,
                'grade': grade,
                'description': description,
                'breakdown': {
                    'win_rate_score': min(25, max(5, int(win_rate / 2.4))),
                    'profit_factor_score': min(25, max(0, int(profit_factor * 12.5))),
                    'sharpe_score': min(25, max(5, int(sharpe_ratio * 12.5))),
                    'drawdown_score': min(25, max(5, int((25 - max_drawdown) * 1.25)))
                }
            }
            
        except Exception as e:
            logger.error(f"Performans notu hesaplama hatası: {e}")
            return {'score': 0, 'grade': 'N/A', 'description': 'Hesaplanamadı'}
