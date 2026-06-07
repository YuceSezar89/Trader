"""
VPM (Volume-Price-Momentum) Calculator
Standardize VPM hesaplama - tüm sistem bu modülü kullanır.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VPMCalculator:
    """
    Standardize VPM (Volume-Price-Momentum) hesaplama sınıfı.

    Tüm sistem (live, backfill, analyzer) bu sınıfı kullanmalı.
    """

    # Default weights
    DEFAULT_WEIGHTS = {"P": 0.4, "V": 0.3, "M": 0.3}  # Price  # Volume  # Momentum

    # Timeframe weights (daha yüksek TF = daha güvenilir)
    TIMEFRAME_WEIGHTS = {
        "1m": 1.0,
        "5m": 1.2,
        "15m": 1.5,
        "1h": 2.0,
        "4h": 2.5,
        "1d": 3.0,
    }

    @classmethod
    def calculate(
        cls,
        volume: float,
        volume_sma: float,
        price_change_pct: float,
        rsi_delta: float,
        interval: str = "1m",
        signal_type: str = "Long",
        weights: Optional[dict] = None,
    ) -> float:
        """
        Standardize VPM skoru hesapla.

        Args:
            volume: Mevcut volume
            volume_sma: Volume SMA (20 period)
            price_change_pct: Fiyat değişim yüzdesi
            rsi_delta: RSI değişimi
            interval: Timeframe (1m, 5m, 15m, 1h, 4h)
            signal_type: 'Long' veya 'Short'
            weights: Custom weights (opsiyonel)

        Returns:
            float: 0-100 arası VPM skoru
        """
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS

        try:
            # 1. Volume Score (Z-score normalize)
            v_score = cls._calculate_volume_score(volume, volume_sma)

            # 2. Price Score (tanh normalize)
            p_score = cls._calculate_price_score(price_change_pct, signal_type)

            # 3. Momentum Score (RSI delta normalize)
            m_score = cls._calculate_momentum_score(rsi_delta, signal_type)

            # 4. Weighted average (0-100 range)
            base_score = (
                weights["P"] * p_score + weights["V"] * v_score + weights["M"] * m_score
            ) / 3.0  # Average of 3 scores

            # 5. Timeframe weight
            tf_weight = cls.TIMEFRAME_WEIGHTS.get(interval, 1.0)

            # 6. Apply timeframe weight
            # base_score: 0-100 range (average of P, V, M)
            # tf_weight: 1.0-3.0
            # final_score can go up to 300 for 1d interval
            final_score = base_score * tf_weight

            # Clamp to 0-100 (but allow higher TF to boost score)
            # For 1m (tf=1.0): max 100
            # For 5m (tf=1.2): max 120 -> clamped to 100
            # For 1h (tf=2.0): max 200 -> clamped to 100
            # This ensures higher TF signals reach 100 more easily
            return max(0.0, min(100.0, final_score))

        except Exception as e:
            logger.error(f"VPM hesaplama hatası: {e}")
            return 0.0

    @staticmethod
    def _calculate_volume_score(volume: float, volume_sma: float) -> float:
        """
        Volume skorunu hesapla (sigmoid normalize 0-100).

        Returns:
            float: 0 ile 100 arası sigmoid normalize skor
        """
        if volume_sma <= 0:
            return 50.0  # Neutral

        # Volume ratio
        ratio = volume / volume_sma

        # Sigmoid normalize (0-1)
        # ratio 1.0 (normal) -> 0.5
        # ratio 2.0 (2x) -> 0.88
        # ratio 3.0 (3x) -> 0.95
        # ratio 0.5 (düşük) -> 0.27
        sigmoid = 1 / (1 + np.exp(-(ratio - 1.0) * 2))

        # Scale to 0-100
        # 0.5 -> 50 (neutral)
        # 0.88 -> 88 (good)
        # 0.95 -> 95 (excellent)
        return sigmoid * 100

    @staticmethod
    def _calculate_price_score(price_change_pct: float, signal_type: str) -> float:
        """
        Price değişim skorunu hesapla (sigmoid normalize 0-100).

        Returns:
            float: 0 ile 100 arası sigmoid normalize skor (yönlü)
        """
        # Yön ayarı (Long için pozitif, Short için negatif)
        side = 1.0 if signal_type == "Long" else -1.0

        # Directional price change
        directional_change = price_change_pct * side

        # Sigmoid normalize (0 to 1)
        # 0% değişim = 0.5
        # +2% = 0.73 (Long için iyi)
        # -2% = 0.27 (Long için kötü)
        # +5% = 0.88
        # -5% = 0.12
        sigmoid = 1 / (1 + np.exp(-directional_change))

        # Scale to 0-100
        # 0.5 -> 50 (neutral)
        # 0.73 -> 73 (good)
        # 0.88 -> 88 (excellent)
        return sigmoid * 100

    @staticmethod
    def _calculate_momentum_score(rsi_delta: float, signal_type: str) -> float:
        """
        Momentum (RSI delta) skorunu hesapla (sigmoid normalize 0-100).

        Returns:
            float: 0 ile 100 arası sigmoid normalize skor (yönlü)
        """
        # Yön ayarı
        side = 1.0 if signal_type == "Long" else -1.0

        # Directional RSI delta
        directional_delta = rsi_delta * side

        # Sigmoid normalize (0 to 1)
        # RSI delta 0 = 0.5 (nötr)
        # RSI delta +10 = 0.73 (Long için iyi)
        # RSI delta -10 = 0.27 (Long için kötü)
        # RSI delta +20 = 0.88
        # RSI delta -20 = 0.12
        sigmoid = 1 / (1 + np.exp(-directional_delta / 10.0))

        # Scale to 0-100
        # 0.5 -> 50 (neutral)
        # 0.73 -> 73 (good)
        # 0.88 -> 88 (excellent)
        return sigmoid * 100

    @classmethod
    def validate_inputs(
        cls, volume: float, volume_sma: float, price_change_pct: float, rsi_delta: float
    ) -> bool:
        """
        Input değerlerini validate et.

        Returns:
            bool: Geçerliyse True
        """
        try:
            # None check
            if any(
                x is None for x in [volume, volume_sma, price_change_pct, rsi_delta]
            ):
                return False

            # Type check
            if not all(
                isinstance(x, (int, float))
                for x in [volume, volume_sma, price_change_pct, rsi_delta]
            ):
                return False

            # Range check
            if volume < 0 or volume_sma < 0:
                return False

            # Reasonable bounds
            if abs(price_change_pct) > 100:  # %100'den fazla değişim şüpheli
                logger.warning(f"Şüpheli price change: {price_change_pct}%")

            if abs(rsi_delta) > 100:  # RSI delta 100'den fazla olamaz
                logger.warning(f"Şüpheli RSI delta: {rsi_delta}")

            return True

        except Exception as e:
            logger.error(f"VPM validation hatası: {e}")
            return False


# Convenience function
def calculate_vpm(
    volume: float,
    volume_sma: float,
    price_change_pct: float,
    rsi_delta: float,
    interval: str = "1m",
    signal_type: str = "Long",
) -> float:
    """
    VPM skorunu hesapla (convenience function).

    Returns:
        float: 0-100 arası VPM skoru
    """
    return VPMCalculator.calculate(
        volume=volume,
        volume_sma=volume_sma,
        price_change_pct=price_change_pct,
        rsi_delta=rsi_delta,
        interval=interval,
        signal_type=signal_type,
    )
