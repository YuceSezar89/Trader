"""StalePanelMixin — worker-heartbeat tabanlı veri bayatlığı göstergesi.

27-28 Ağu 2026'da 4 panele (ranking_panel, tf_alignment_panel,
vpmv_divergence_panel, trade_xray_panel) ayrı ayrı elle eklenen "worker X
saniyedir güncelleme yapmadıysa durum etiketini kırmızıya çevir" deseninin
tek hâli (28 Ağu 2026, Model/View göçü Faz 0).

Kullanım: panel sınıfı `class Foo(QWidget, StalePanelMixin)` olur,
__init__'te `self._init_staleness(status_label)` çağrılır, worker'dan her
veri geldiğinde `self._mark_fresh()` çağrılır.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

from desktop.theme import COLORS

_DEFAULT_THRESHOLD_SEC = 90.0
_DEFAULT_CHECK_INTERVAL_MS = 10_000


class StalePanelMixin:
    def _init_staleness(
        self,
        status_label: QLabel,
        *,
        threshold_sec: float = _DEFAULT_THRESHOLD_SEC,
        check_interval_ms: int = _DEFAULT_CHECK_INTERVAL_MS,
    ) -> None:
        self._stale_status_label = status_label
        self._stale_threshold_sec = threshold_sec
        self._last_update_monotonic = time.monotonic()
        self._is_stale = False
        self._stale_timer = QTimer(self)  # self: QWidget+mixin, QObject parent şart
        self._stale_timer.setInterval(check_interval_ms)
        self._stale_timer.timeout.connect(self._check_stale)
        self._stale_timer.start()

    def _mark_fresh(self) -> None:
        self._last_update_monotonic = time.monotonic()
        if self._is_stale:
            self._is_stale = False
            self._stale_status_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 11px;"
            )

    def _check_stale(self) -> None:
        age = time.monotonic() - self._last_update_monotonic
        if age > self._stale_threshold_sec and not self._is_stale:
            self._is_stale = True
            self._stale_status_label.setStyleSheet(
                f"color: {COLORS['red']}; font-size: 11px; font-weight: bold;"
            )
            self._stale_status_label.setText(f"⚠ {age:.0f}sn'dir veri güncellenmiyor")
