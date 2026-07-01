"""
VPM Calculator Unit Tests
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.vpm_calculator import VPMCalculator, calculate_vpm


class TestVPMCalculator:
    """VPM Calculator test suite"""
    
    def test_basic_calculation(self):
        """Temel VPM hesaplama testi"""
        score = VPMCalculator.calculate(
            volume=1000,
            volume_sma=500,
            price_change_pct=2.0,
            rsi_delta=10.0,
            interval='1m',
            signal_type='Long'
        )
        
        assert 0 <= score <= 100
        assert score > 0  # Pozitif değerler için skor > 0 olmalı
    
    def test_timeframe_weights(self):
        """Timeframe ağırlıkları testi"""
        base_params = {
            'volume': 1000,
            'volume_sma': 500,
            'price_change_pct': 2.0,
            'rsi_delta': 10.0,
            'signal_type': 'Long'
        }
        
        score_1m = VPMCalculator.calculate(**base_params, interval='1m')
        score_5m = VPMCalculator.calculate(**base_params, interval='5m')
        score_1h = VPMCalculator.calculate(**base_params, interval='1h')
        
        # Daha yüksek TF = daha yüksek skor
        assert score_1m < score_5m < score_1h
    
    def test_signal_type_direction(self):
        """Long vs Short yön testi"""
        params = {
            'volume': 1000,
            'volume_sma': 500,
            'price_change_pct': 2.0,  # Pozitif değişim
            'rsi_delta': 10.0,
            'interval': '1m'
        }
        
        score_long = VPMCalculator.calculate(**params, signal_type='Long')
        score_short = VPMCalculator.calculate(**params, signal_type='Short')
        
        # Pozitif değişimde Long > Short
        assert score_long > score_short
    
    def test_volume_score(self):
        """Volume skor hesaplama testi (sigmoid 0-100)"""
        # Normal volume (1x SMA) -> 50
        score1 = VPMCalculator._calculate_volume_score(100, 100)
        assert 45 < score1 < 55
        
        # 2x volume -> ~88
        score2 = VPMCalculator._calculate_volume_score(200, 100)
        assert 85 < score2 < 90
        
        # 3x volume -> ~98
        score3 = VPMCalculator._calculate_volume_score(300, 100)
        assert 95 < score3 < 99
        
        # Düşük volume (0.5x) -> ~27
        score4 = VPMCalculator._calculate_volume_score(50, 100)
        assert 25 < score4 < 30
    
    def test_price_score(self):
        """Price skor hesaplama testi (sigmoid 0-100)"""
        # Long pozitif değişim (+2%) -> ~88
        score1 = VPMCalculator._calculate_price_score(2.0, 'Long')
        assert 85 < score1 < 92
        
        # Short pozitif değişim (ters yön) -> ~12
        score2 = VPMCalculator._calculate_price_score(2.0, 'Short')
        assert 10 < score2 < 15
        
        # Long negatif değişim (-2%) -> ~12
        score3 = VPMCalculator._calculate_price_score(-2.0, 'Long')
        assert 10 < score3 < 15
        
        # Nötr (0%) -> 50
        score4 = VPMCalculator._calculate_price_score(0.0, 'Long')
        assert 48 < score4 < 52
    
    def test_momentum_score(self):
        """Momentum skor hesaplama testi (sigmoid 0-100)"""
        # Long pozitif RSI delta (+10) -> ~73
        score1 = VPMCalculator._calculate_momentum_score(10.0, 'Long')
        assert 70 < score1 < 75
        
        # Short pozitif RSI delta (ters yön) -> ~27
        score2 = VPMCalculator._calculate_momentum_score(10.0, 'Short')
        assert 25 < score2 < 30
        
        # Nötr (0) -> 50
        score3 = VPMCalculator._calculate_momentum_score(0.0, 'Long')
        assert 48 < score3 < 52
    
    def test_validation(self):
        """Input validation testi"""
        # Geçerli inputs
        assert VPMCalculator.validate_inputs(100, 50, 2.0, 10.0) is True
        
        # None değer
        assert VPMCalculator.validate_inputs(None, 50, 2.0, 10.0) is False
        
        # Negatif volume
        assert VPMCalculator.validate_inputs(-100, 50, 2.0, 10.0) is False
    
    def test_convenience_function(self):
        """Convenience function testi"""
        score = calculate_vpm(
            volume=1000,
            volume_sma=500,
            price_change_pct=2.0,
            rsi_delta=10.0,
            interval='5m',
            signal_type='Long'
        )
        
        assert 0 <= score <= 100
    
    def test_edge_cases(self):
        """Edge case'ler testi"""
        # Zero volume SMA
        score1 = VPMCalculator.calculate(100, 0, 2.0, 10.0)
        assert score1 >= 0
        
        # Çok yüksek price change
        score2 = VPMCalculator.calculate(100, 50, 50.0, 10.0)
        assert 0 <= score2 <= 100
        
        # Çok yüksek RSI delta
        score3 = VPMCalculator.calculate(100, 50, 2.0, 100.0)
        assert 0 <= score3 <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
