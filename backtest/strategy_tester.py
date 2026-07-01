"""
Strateji Test Modülü - VP sisteminin farklı parametrelerini test eder ve optimize eder
"""

import asyncio
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import logging
import json

from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer
from .models import BacktestResult, StrategyParameter
from database.engine import get_session

logger = logging.getLogger(__name__)


class StrategyTester:
    """
    VP sistemi parametre optimizasyonu ve strateji testi
    """
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.default_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        
        # Test parametreleri
        self.default_test_params = {
            'momentum_periods': [1, 3, 5, 10],
            'vmp_thresholds': [0.2, 0.4, 0.6, 0.8],
            'volume_thresholds': [1.5, 1.8, 2.0, 2.5],
            'stop_loss_percentages': [1.0, 2.0, 3.0],
            'take_profit_percentages': [3.0, 5.0, 8.0],
            'timeframes': ['1h', '4h']
        }
    
    async def run_parameter_optimization(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        test_params: Optional[Dict] = None,
        max_combinations: int = 50
    ) -> List[Dict]:
        """
        Parametre optimizasyonu yap
        
        Args:
            symbols: Test edilecek semboller
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            test_params: Test parametreleri
            max_combinations: Maksimum kombinasyon sayısı
        
        Returns:
            List[Dict]: Sıralı test sonuçları
        """
        logger.info("Parametre optimizasyonu başlatılıyor...")
        
        # Varsayılan değerleri ayarla
        symbols = symbols or self.default_symbols
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        test_params = test_params or self.default_test_params
        
        # Parametre kombinasyonlarını oluştur
        combinations = self._generate_parameter_combinations(test_params, max_combinations)
        
        logger.info(f"{len(combinations)} parametre kombinasyonu test edilecek")
        
        # Her kombinasyonu test et
        results = []
        for i, params in enumerate(combinations):
            try:
                logger.info(f"Test {i+1}/{len(combinations)}: {params}")
                
                # Backtest yap
                result = await self._test_parameter_combination(
                    params, symbols, start_date, end_date
                )
                
                if result:
                    results.append(result)
                
                # Her 10 testte bir progress log
                if (i + 1) % 10 == 0:
                    logger.info(f"İlerleme: {i+1}/{len(combinations)} tamamlandı")
                    
            except Exception as e:
                logger.error(f"Parametre testi hatası {params}: {e}")
                continue
        
        # Sonuçları sırala (Sharpe ratio'ya göre)
        results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        
        logger.info(f"Parametre optimizasyonu tamamlandı: {len(results)} başarılı test")
        return results
    
    async def compare_strategies(
        self,
        strategy_configs: List[Dict],
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Farklı stratejileri karşılaştır
        
        Args:
            strategy_configs: Strateji konfigürasyonları
            symbols: Test sembolleri
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
        
        Returns:
            Dict: Karşılaştırma sonuçları
        """
        logger.info(f"{len(strategy_configs)} strateji karşılaştırılıyor...")
        
        # Varsayılan değerleri ayarla
        symbols = symbols or self.default_symbols
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        strategy_results = []
        
        for config in strategy_configs:
            try:
                strategy_name = config.get('name', 'Unknown Strategy')
                parameters = config.get('parameters', {})
                
                logger.info(f"Test ediliyor: {strategy_name}")
                
                # Backtest engine oluştur
                engine = BacktestEngine()
                
                # Strateji parametrelerini uygula
                if parameters:
                    engine._apply_strategy_params(parameters)
                
                # Backtest yap
                result = await engine.run_backtest(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=strategy_name,
                    strategy_params=parameters
                )
                
                # Sonucu kaydet
                strategy_results.append({
                    'strategy_name': strategy_name,
                    'parameters': parameters,
                    'result': result,
                    'performance_metrics': {
                        'total_trades': result.total_trades,
                        'win_rate': float(result.win_rate or 0),
                        'total_pnl': float(result.total_pnl or 0),
                        'sharpe_ratio': float(result.sharpe_ratio or 0),
                        'max_drawdown': float(result.max_drawdown or 0),
                        'profit_factor': float(result.profit_factor or 0)
                    }
                })
                
            except Exception as e:
                logger.error(f"Strateji testi hatası {strategy_name}: {e}")
                continue
        
        # Karşılaştırma analizi
        comparison = self._analyze_strategy_comparison(strategy_results)
        
        logger.info("Strateji karşılaştırması tamamlandı")
        return comparison
    
    async def run_momentum_period_analysis(
        self,
        symbols: Optional[List[str]] = None,
        periods: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Momentum periyodu analizi (memory'den gelen gereksinim)
        
        Args:
            symbols: Test sembolleri
            periods: Test edilecek periyotlar
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
        
        Returns:
            Dict: Momentum periyodu analiz sonuçları
        """
        logger.info("Momentum periyodu analizi başlatılıyor...")
        
        # Varsayılan değerler
        symbols = symbols or self.default_symbols
        periods = periods or [1, 3, 5, 10]  # Memory'den gelen öneriler
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        period_results = []
        
        for period in periods:
            try:
                logger.info(f"Momentum periyodu test ediliyor: {period} bar")
                
                # Parametre seti oluştur
                params = {
                    'momentum_period': period,
                    'vmp_threshold': 0.5,  # Sabit eşik
                    'stop_loss_percentage': 2.0,
                    'take_profit_percentage': 5.0
                }
                
                # Backtest yap
                engine = BacktestEngine()
                result = await engine.run_backtest(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=f'VP_Momentum_{period}bar',
                    strategy_params=params
                )
                
                # Sonucu kaydet
                period_results.append({
                    'momentum_period': period,
                    'total_trades': result.total_trades,
                    'win_rate': float(result.win_rate or 0),
                    'total_pnl': float(result.total_pnl or 0),
                    'sharpe_ratio': float(result.sharpe_ratio or 0),
                    'max_drawdown': float(result.max_drawdown or 0),
                    'profit_factor': float(result.profit_factor or 0),
                    'avg_win': float(result.avg_win or 0),
                    'avg_loss': float(result.avg_loss or 0)
                })
                
            except Exception as e:
                logger.error(f"Momentum periyodu {period} testi hatası: {e}")
                continue
        
        # En iyi periyodu bul
        best_period = max(period_results, key=lambda x: x['sharpe_ratio']) if period_results else None
        
        analysis = {
            'period_results': period_results,
            'best_period': best_period,
            'analysis_summary': self._analyze_momentum_periods(period_results),
            'recommendations': self._generate_momentum_recommendations(period_results)
        }
        
        logger.info("Momentum periyodu analizi tamamlandı")
        return analysis
    
    async def run_vmp_threshold_analysis(
        self,
        symbols: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        VMP eşik değeri analizi
        
        Args:
            symbols: Test sembolleri
            thresholds: Test edilecek eşik değerleri
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
        
        Returns:
            Dict: VMP eşik analiz sonuçları
        """
        logger.info("VMP eşik değeri analizi başlatılıyor...")
        
        # Varsayılan değerler
        symbols = symbols or self.default_symbols
        thresholds = thresholds or [0.2, 0.4, 0.6, 0.8, 1.0]
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        threshold_results = []
        
        for threshold in thresholds:
            try:
                logger.info(f"VMP eşiği test ediliyor: {threshold}")
                
                # Parametre seti oluştur
                params = {
                    'vmp_threshold': threshold,
                    'momentum_period': 5,  # Sabit periyot
                    'stop_loss_percentage': 2.0,
                    'take_profit_percentage': 5.0
                }
                
                # Backtest yap
                engine = BacktestEngine()
                result = await engine.run_backtest(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=f'VP_Threshold_{threshold}',
                    strategy_params=params
                )
                
                # Sonucu kaydet
                threshold_results.append({
                    'vmp_threshold': threshold,
                    'total_trades': result.total_trades,
                    'win_rate': float(result.win_rate or 0),
                    'total_pnl': float(result.total_pnl or 0),
                    'sharpe_ratio': float(result.sharpe_ratio or 0),
                    'max_drawdown': float(result.max_drawdown or 0),
                    'profit_factor': float(result.profit_factor or 0)
                })
                
            except Exception as e:
                logger.error(f"VMP eşiği {threshold} testi hatası: {e}")
                continue
        
        # En iyi eşiği bul
        best_threshold = max(threshold_results, key=lambda x: x['sharpe_ratio']) if threshold_results else None
        
        analysis = {
            'threshold_results': threshold_results,
            'best_threshold': best_threshold,
            'analysis_summary': self._analyze_vmp_thresholds(threshold_results),
            'recommendations': self._generate_threshold_recommendations(threshold_results)
        }
        
        logger.info("VMP eşik değeri analizi tamamlandı")
        return analysis
    
    def _generate_parameter_combinations(self, test_params: Dict, max_combinations: int) -> List[Dict]:
        """Parametre kombinasyonlarını oluştur"""
        try:
            # Tüm parametrelerin kartezyen çarpımını al
            param_names = list(test_params.keys())
            param_values = list(test_params.values())
            
            all_combinations = list(itertools.product(*param_values))
            
            # Maksimum kombinasyon sayısını sınırla
            if len(all_combinations) > max_combinations:
                # Rastgele örnekleme yap
                import random
                random.shuffle(all_combinations)
                all_combinations = all_combinations[:max_combinations]
            
            # Dict formatına çevir
            combinations = []
            for combo in all_combinations:
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Parametre kombinasyonu oluşturma hatası: {e}")
            return []
    
    async def _test_parameter_combination(
        self,
        params: Dict,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict]:
        """Tek parametre kombinasyonunu test et"""
        try:
            # Backtest engine oluştur
            engine = BacktestEngine()
            
            # Parametreleri uygula
            engine._apply_strategy_params(params)
            
            # Strateji adı oluştur
            strategy_name = f"VP_Opt_{hash(str(params)) % 10000}"
            
            # Backtest yap
            result = await engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                strategy_name=strategy_name,
                strategy_params=params
            )
            
            # Sonucu formatla
            return {
                'parameters': params,
                'strategy_name': strategy_name,
                'total_trades': result.total_trades,
                'win_rate': float(result.win_rate or 0),
                'total_pnl': float(result.total_pnl or 0),
                'sharpe_ratio': float(result.sharpe_ratio or 0),
                'max_drawdown': float(result.max_drawdown or 0),
                'profit_factor': float(result.profit_factor or 0),
                'backtest_result_id': result.id
            }
            
        except Exception as e:
            logger.error(f"Parametre kombinasyonu test hatası: {e}")
            return None
    
    def _analyze_strategy_comparison(self, strategy_results: List[Dict]) -> Dict:
        """Strateji karşılaştırma analizi"""
        try:
            if not strategy_results:
                return {}
            
            # En iyi performans metriklerini bul
            best_win_rate = max(strategy_results, key=lambda x: x['performance_metrics']['win_rate'])
            best_sharpe = max(strategy_results, key=lambda x: x['performance_metrics']['sharpe_ratio'])
            best_profit = max(strategy_results, key=lambda x: x['performance_metrics']['total_pnl'])
            lowest_drawdown = min(strategy_results, key=lambda x: x['performance_metrics']['max_drawdown'])
            
            # Genel sıralama (çoklu kriter)
            for result in strategy_results:
                metrics = result['performance_metrics']
                # Basit skor hesaplama
                score = (
                    metrics['win_rate'] * 0.3 +
                    metrics['sharpe_ratio'] * 0.3 +
                    (metrics['total_pnl'] / 1000) * 0.2 +  # Normalize et
                    (100 - metrics['max_drawdown']) * 0.2
                )
                result['overall_score'] = score
            
            # Sırala
            strategy_results.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return {
                'strategies': strategy_results,
                'best_performers': {
                    'highest_win_rate': best_win_rate['strategy_name'],
                    'best_sharpe_ratio': best_sharpe['strategy_name'],
                    'highest_profit': best_profit['strategy_name'],
                    'lowest_drawdown': lowest_drawdown['strategy_name']
                },
                'overall_winner': strategy_results[0]['strategy_name'] if strategy_results else None,
                'summary_stats': {
                    'avg_win_rate': sum(r['performance_metrics']['win_rate'] for r in strategy_results) / len(strategy_results),
                    'avg_sharpe_ratio': sum(r['performance_metrics']['sharpe_ratio'] for r in strategy_results) / len(strategy_results),
                    'avg_max_drawdown': sum(r['performance_metrics']['max_drawdown'] for r in strategy_results) / len(strategy_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Strateji karşılaştırma analizi hatası: {e}")
            return {}
    
    def _analyze_momentum_periods(self, period_results: List[Dict]) -> Dict:
        """Momentum periyodu analizi"""
        try:
            if not period_results:
                return {}
            
            # En iyi periyotları bul
            best_win_rate = max(period_results, key=lambda x: x['win_rate'])
            best_sharpe = max(period_results, key=lambda x: x['sharpe_ratio'])
            best_profit_factor = max(period_results, key=lambda x: x['profit_factor'])
            
            # Trend analizi
            periods = [r['momentum_period'] for r in period_results]
            win_rates = [r['win_rate'] for r in period_results]
            sharpe_ratios = [r['sharpe_ratio'] for r in period_results]
            
            return {
                'best_periods': {
                    'win_rate': best_win_rate['momentum_period'],
                    'sharpe_ratio': best_sharpe['momentum_period'],
                    'profit_factor': best_profit_factor['momentum_period']
                },
                'trends': {
                    'win_rate_trend': 'increasing' if win_rates[-1] > win_rates[0] else 'decreasing',
                    'sharpe_trend': 'increasing' if sharpe_ratios[-1] > sharpe_ratios[0] else 'decreasing'
                },
                'optimal_range': {
                    'min_period': min(periods),
                    'max_period': max(periods),
                    'recommended_period': best_sharpe['momentum_period']
                }
            }
            
        except Exception as e:
            logger.error(f"Momentum periyodu analizi hatası: {e}")
            return {}
    
    def _generate_momentum_recommendations(self, period_results: List[Dict]) -> List[str]:
        """Momentum periyodu önerileri"""
        try:
            if not period_results:
                return []
            
            recommendations = []
            
            # En iyi performans
            best_period = max(period_results, key=lambda x: x['sharpe_ratio'])
            recommendations.append(f"En iyi performans: {best_period['momentum_period']} bar (Sharpe: {best_period['sharpe_ratio']:.2f})")
            
            # Kısa vs uzun periyot
            short_periods = [r for r in period_results if r['momentum_period'] <= 3]
            long_periods = [r for r in period_results if r['momentum_period'] >= 5]
            
            if short_periods and long_periods:
                avg_short_win = sum(r['win_rate'] for r in short_periods) / len(short_periods)
                avg_long_win = sum(r['win_rate'] for r in long_periods) / len(long_periods)
                
                if avg_short_win > avg_long_win:
                    recommendations.append("Kısa periyotlar (1-3 bar) daha yüksek win rate sağlıyor")
                else:
                    recommendations.append("Uzun periyotlar (5+ bar) daha yüksek win rate sağlıyor")
            
            # Risk analizi
            low_drawdown = min(period_results, key=lambda x: x['max_drawdown'])
            recommendations.append(f"En düşük risk: {low_drawdown['momentum_period']} bar (Drawdown: {low_drawdown['max_drawdown']:.2f}%)")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Momentum önerisi oluşturma hatası: {e}")
            return []
    
    def _analyze_vmp_thresholds(self, threshold_results: List[Dict]) -> Dict:
        """VMP eşik analizi"""
        try:
            if not threshold_results:
                return {}
            
            # Trade sayısı vs kalite analizi
            trade_counts = [r['total_trades'] for r in threshold_results]
            win_rates = [r['win_rate'] for r in threshold_results]
            
            return {
                'trade_frequency_analysis': {
                    'highest_frequency': max(threshold_results, key=lambda x: x['total_trades']),
                    'best_quality': max(threshold_results, key=lambda x: x['win_rate']),
                    'optimal_balance': max(threshold_results, key=lambda x: x['sharpe_ratio'])
                },
                'threshold_effects': {
                    'low_threshold_effect': 'Daha fazla sinyal, düşük kalite' if trade_counts[0] > trade_counts[-1] else 'Beklenmedik sonuç',
                    'high_threshold_effect': 'Daha az sinyal, yüksek kalite' if win_rates[-1] > win_rates[0] else 'Beklenmedik sonuç'
                }
            }
            
        except Exception as e:
            logger.error(f"VMP eşik analizi hatası: {e}")
            return {}
    
    def _generate_threshold_recommendations(self, threshold_results: List[Dict]) -> List[str]:
        """VMP eşik önerileri"""
        try:
            if not threshold_results:
                return []
            
            recommendations = []
            
            # En iyi eşik
            best_threshold = max(threshold_results, key=lambda x: x['sharpe_ratio'])
            recommendations.append(f"Optimal eşik: {best_threshold['vmp_threshold']} (Sharpe: {best_threshold['sharpe_ratio']:.2f})")
            
            # Trade frekansı analizi
            high_freq = max(threshold_results, key=lambda x: x['total_trades'])
            recommendations.append(f"En yüksek sinyal frekansı: {high_freq['vmp_threshold']} ({high_freq['total_trades']} sinyal)")
            
            # Kalite analizi
            high_quality = max(threshold_results, key=lambda x: x['win_rate'])
            recommendations.append(f"En yüksek kalite: {high_quality['vmp_threshold']} (%{high_quality['win_rate']:.1f} win rate)")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Eşik önerisi oluşturma hatası: {e}")
            return []
    
    async def save_optimization_results(self, results: List[Dict], optimization_name: str):
        """Optimizasyon sonuçlarını kaydet"""
        try:
            async with get_session() as session:
                for result in results:
                    # Her sonuç için parametre kayıtları oluştur
                    for param_name, param_value in result['parameters'].items():
                        param_record = StrategyParameter(
                            strategy_name=f"{optimization_name}_{result['strategy_name']}",
                            parameter_name=param_name,
                            parameter_value=str(param_value),
                            parameter_type=type(param_value).__name__,
                            backtest_result_id=result.get('backtest_result_id')
                        )
                        session.add(param_record)
                
                await session.commit()
                logger.info(f"Optimizasyon sonuçları kaydedildi: {optimization_name}")
                
        except Exception as e:
            logger.error(f"Optimizasyon sonucu kaydetme hatası: {e}")
