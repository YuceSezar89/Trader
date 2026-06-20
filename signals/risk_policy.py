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


class DynamicATRPolicy(RiskPolicy):
    """
    Dinamik R:R politikası.
    SL sabit (ATR × sl_mult), TP sinyal kalitesine göre genişler.

    Her faktör TP çarpanına +0.5 bonus ekler:
      - VPMV ≥ eşik  → güçlü hacim momentumu
      - MTF = 100%   → tüm üst TF'ler uyumlu
      - İnterval 15m → en kaliteli kısa vadeli TF (veriden)
    """

    def __init__(
        self,
        sl_mult: float,
        tp_base: float,
        tp_max: float,
        vpmv_threshold: float,
        mtf_full: float,
        bonus_intervals: tuple[str, ...],
        bonus_per_factor: float = 0.5,
    ):
        self.sl_mult          = sl_mult
        self.tp_base          = tp_base
        self.tp_max           = tp_max
        self.vpmv_threshold   = vpmv_threshold
        self.mtf_full         = mtf_full
        self.bonus_intervals  = bonus_intervals
        self.bonus_per_factor = bonus_per_factor

    def calculate_levels(
        self,
        signal_type: str,
        open_price: float,
        atr: float,
        features: Optional[dict] = None,
    ) -> RiskLevels:
        f = features or {}

        bonus = 0.0
        vpmv     = f.get("vpms_score")
        mtf      = f.get("mtf_score")
        interval = f.get("interval", "")

        if vpmv is not None and float(vpmv) >= self.vpmv_threshold:
            bonus += self.bonus_per_factor
        if mtf is not None and float(mtf) >= self.mtf_full:
            bonus += self.bonus_per_factor
        if interval in self.bonus_intervals:
            bonus += self.bonus_per_factor

        tp_mult  = min(self.tp_base + bonus, self.tp_max)
        sl_dist  = atr * self.sl_mult
        tp_dist  = atr * tp_mult

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
            tp_multiplier=round(tp_mult, 2),
        )


from config import Config  # pylint: disable=wrong-import-position
default_policy = DynamicATRPolicy(
    sl_mult=Config.RISK_SL_MULTIPLIER,
    tp_base=Config.RISK_TP_MULTIPLIER,
    tp_max=Config.DYNAMIC_RR_TP_MAX,
    vpmv_threshold=Config.DYNAMIC_RR_VPMV_THRESHOLD,
    mtf_full=Config.DYNAMIC_RR_MTF_FULL,
    bonus_intervals=Config.DYNAMIC_RR_BONUS_INTERVALS,
)
