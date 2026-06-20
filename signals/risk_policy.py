"""
Risk politikası arayüzü ve uygulamaları.

RiskPolicy.calculate_levels() → (sl_price, tp_price, sl_mult, tp_mult)

Gelecekte ML politikaları bu arayüzü implement eder:
    class MLPolicy(RiskPolicy):
        def calculate_levels(self, signal, features): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional


class RiskLevels(NamedTuple):
    sl_price:    float
    tp_price:    float
    sl_multiplier: float
    tp_multiplier: float


class RiskPolicy(ABC):
    @abstractmethod
    def calculate_levels(
        self,
        signal_type: str,
        open_price: float,
        atr: float,
        features: Optional[dict] = None,
    ) -> RiskLevels:
        """
        SL ve TP fiyat seviyelerini hesaplar.

        Args:
            signal_type: 'Long' veya 'Short'
            open_price:  Sinyalin giriş fiyatı
            atr:         Giriş anındaki ATR değeri
            features:    ML için ek özellikler (rejim, z_score, vb.)

        Returns:
            RiskLevels(sl_price, tp_price, sl_multiplier, tp_multiplier)
        """


class FixedATRPolicy(RiskPolicy):
    """
    Sabit ATR çarpanlı politika.
    sl_mult ve tp_mult config'den alınır, değiştirilebilir.
    """

    def __init__(self, sl_mult: float = 1.5, tp_mult: float = 3.0):
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def calculate_levels(
        self,
        signal_type: str,
        open_price: float,
        atr: float,
        features: Optional[dict] = None,
    ) -> RiskLevels:
        sl_dist = atr * self.sl_mult
        tp_dist = atr * self.tp_mult

        if signal_type == "Long":
            sl_price = open_price - sl_dist
            tp_price = open_price + tp_dist
        else:
            sl_price = open_price + sl_dist
            tp_price = open_price - tp_dist

        return RiskLevels(
            sl_price=round(sl_price, 10),
            tp_price=round(tp_price, 10),
            sl_multiplier=self.sl_mult,
            tp_multiplier=self.tp_mult,
        )


from config import Config  # pylint: disable=wrong-import-position
default_policy = FixedATRPolicy(
    sl_mult=Config.RISK_SL_MULTIPLIER,
    tp_mult=Config.RISK_TP_MULTIPLIER,
)
