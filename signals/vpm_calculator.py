"""
VPMV (Volume-Price-Momentum-Volatility) Calculator

Her bileşen dışarıda rolling normalize edilerek 0-100 arasında gelir.
Bu sınıf sadece ağırlıklı toplama yapar.

Normalizasyon sorumluluğu signal_processor.py'dedir:
  V   → normalize_volume_0_100   (log + rolling min-max)
  M   → normalize_momentum_0_100 (yönlü z-score + sigmoid)
  Vlt → normalize_volatility_0_100 (ATR percentile rank)
  P   → normalize_price_0_100    (rolling IQR)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VPMCalculator:
    DEFAULT_WEIGHTS = {
        "V":   0.35,
        "M":   0.35,
        "Vlt": 0.20,
        "P":   0.10,
    }

    @classmethod
    def calculate(
        cls,
        vol_score: float,
        momentum_score: float,
        vlt_score: float,
        price_score: float,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Normalize edilmiş bileşenlerden VPMV skoru hesaplar.

        Args:
            vol_score:      Volume bileşeni (0-100)
            momentum_score: Momentum bileşeni, yönlü (0-100)
            vlt_score:      Volatilite bileşeni (0-100)
            price_score:    Fiyat bileşeni, yönlü (0-100)
            weights:        Opsiyonel ağırlık dict {V, M, Vlt, P}

        Returns:
            float: 0-100 arası VPMV skoru
        """
        w = weights or cls.DEFAULT_WEIGHTS
        total_w = w["V"] + w["M"] + w["Vlt"] + w["P"]
        if total_w == 0:
            return 0.0
        try:
            score = (
                w["V"]   * vol_score +
                w["M"]   * momentum_score +
                w["Vlt"] * vlt_score +
                w["P"]   * price_score
            ) / total_w
            return max(0.0, min(100.0, score))
        except Exception as e:
            logger.error(f"VPMV hesaplama hatası: {e}")
            return 0.0


def calculate_vpm(
    vol_score: float,
    momentum_score: float,
    vlt_score: float,
    price_score: float,
) -> float:
    return VPMCalculator.calculate(vol_score, momentum_score, vlt_score, price_score)
