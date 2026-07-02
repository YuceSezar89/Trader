"""
VPMCalculator testleri — güncel API: normalize bileşenlerden ağırlıklı ortalama.

Bileşen hesaplama (volume/momentum/volatilite/fiyat skorları) artık
utils/vpmv.py ve signals/signal_processor._compute_vpmv_scores içinde;
bu dosya yalnızca birleştirme katmanını test eder.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.vpm_calculator import VPMCalculator, calculate_vpm


class TestVPMCalculator:

    def test_equal_components(self):
        """Tüm bileşenler aynıysa skor da aynı olmalı (ağırlıklardan bağımsız)."""
        assert VPMCalculator.calculate(70.0, 70.0, 70.0, 70.0) == pytest.approx(70.0)

    def test_weighted_average(self):
        """Varsayılan ağırlıklar: V=0.35, M=0.35, Vlt=0.20, P=0.10."""
        assert VPMCalculator.calculate(100.0, 0.0, 0.0, 0.0) == pytest.approx(35.0)
        assert VPMCalculator.calculate(0.0, 100.0, 0.0, 0.0) == pytest.approx(35.0)
        assert VPMCalculator.calculate(0.0, 0.0, 100.0, 0.0) == pytest.approx(20.0)
        assert VPMCalculator.calculate(0.0, 0.0, 0.0, 100.0) == pytest.approx(10.0)

    def test_custom_weights(self):
        """Özel ağırlıklar toplamı normalize edilmeli."""
        weights = {"V": 1.0, "M": 1.0, "Vlt": 1.0, "P": 1.0}
        score = VPMCalculator.calculate(80.0, 40.0, 60.0, 20.0, weights=weights)
        assert score == pytest.approx(50.0)

    def test_zero_weights_returns_zero(self):
        """Ağırlık toplamı 0 ise güvenli 0.0 dönmeli (sıfıra bölme yok)."""
        weights = {"V": 0.0, "M": 0.0, "Vlt": 0.0, "P": 0.0}
        assert VPMCalculator.calculate(80.0, 80.0, 80.0, 80.0, weights=weights) == 0.0

    def test_clamped_to_0_100(self):
        """Skor 0-100 aralığına sıkıştırılmalı."""
        assert VPMCalculator.calculate(150.0, 150.0, 150.0, 150.0) == 100.0
        assert VPMCalculator.calculate(-50.0, -50.0, -50.0, -50.0) == 0.0

    def test_boundaries(self):
        assert VPMCalculator.calculate(0.0, 0.0, 0.0, 0.0) == 0.0
        assert VPMCalculator.calculate(100.0, 100.0, 100.0, 100.0) == pytest.approx(100.0)

    def test_config_weights_match_default(self):
        """Config.VPM WEIGHTS ile sınıfın varsayılanı senkron kalmalı."""
        from config import Config
        assert VPMCalculator.DEFAULT_WEIGHTS == Config.VPM["WEIGHTS"]

    def test_convenience_function(self):
        """calculate_vpm sınıf metoduyla aynı sonucu vermeli."""
        args = (75.0, 60.0, 40.0, 55.0)
        assert calculate_vpm(*args) == VPMCalculator.calculate(*args)

    def test_incomplete_weights_raise(self):
        """Eksik anahtarlı ağırlık dict'i KeyError fırlatır (mevcut sözleşme)."""
        with pytest.raises(KeyError):
            VPMCalculator.calculate(50.0, 50.0, 50.0, 50.0, weights={"V": 1.0})
