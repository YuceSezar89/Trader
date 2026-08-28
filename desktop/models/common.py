"""Panel Model/View göçü için paylaşılan küçük yardımcılar.

Buraya SADECE en az 3 panelde birebir aynı şekilde tekrarlanmış, kanıtlanmış
desenler eklenir (28 Ağu 2026 Model/View göçü, Faz 0) — spekülatif/"ileride
lazım olur" soyutlama yok. Yeni bir pano taşınırken aynı şekli üçüncü kez
görürsen buraya ekle, görmeden ekleme.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtGui import QColor

from desktop.theme import COLORS

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])


def none_safe_lt(a: Optional[float], b: Optional[float]) -> bool:
    """QSortFilterProxyModel.lessThan içinde kullanılan None-güvenli
    karşılaştırma — None her zaman "en küçük" sayılır (eksik veri sıralamada
    sona/başa tutarlı şekilde gider). SignalsProxyModel.lessThan'daki `_cmp`
    ile aynı."""
    if a is None and b is None:
        return False
    if a is None:
        return True
    if b is None:
        return False
    return a < b


def sign_color(val: Optional[float], *, zero_is_muted: bool = True) -> QColor:
    """val > 0 → yeşil, val < 0 → kırmızı, val == 0 veya None → soluk.
    ranking_panel._set_rscore/_set_vs_btc ve SignalsModel'in alpha
    renklendirmesiyle aynı desen."""
    if val is None:
        return _C_MUTED
    if val > 0:
        return _C_GREEN
    if val < 0:
        return _C_RED
    return _C_MUTED if zero_is_muted else QColor(COLORS["text_primary"])
