"""
ManualTradeDialog — paper trade'e manuel işlem ekleme dialog'u.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import psycopg2
import redis as _redis
from PyQt6.QtCore import Qt  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from desktop.theme import COLORS

logger = logging.getLogger(__name__)

_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
_POSITION_USD = 100.0
_FEE_RATE = 0.0005


class ManualTradeDialog(QDialog):

    def __init__(self, db_config: dict[str, Any], redis_url: str, parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._redis_url = redis_url
        self.setWindowTitle("Manuel Paper Trade Aç")
        self.setMinimumWidth(340)
        self._direction = "Long"
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 12)

        form = QFormLayout()
        form.setSpacing(8)

        # Sembol
        self._sym = QLineEdit()
        self._sym.setPlaceholderText("örn. BTCUSDT")
        self._sym.textChanged.connect(lambda t: self._sym.setText(t.upper()))
        form.addRow("Sembol:", self._sym)

        # Yön
        dir_row = QHBoxLayout()
        self._btn_long  = QPushButton("LONG")
        self._btn_short = QPushButton("SHORT")
        for btn in (self._btn_long, self._btn_short):
            btn.setCheckable(True)
            btn.setFixedHeight(28)
        self._btn_long.setChecked(True)
        self._btn_long.clicked.connect(lambda: self._set_dir("Long"))
        self._btn_short.clicked.connect(lambda: self._set_dir("Short"))
        self._update_dir_style()
        dir_row.addWidget(self._btn_long)
        dir_row.addWidget(self._btn_short)
        form.addRow("Yön:", dir_row)

        # TF
        self._tf = QComboBox()
        self._tf.addItems(_TFS)
        self._tf.setCurrentText("15m")
        form.addRow("TF:", self._tf)

        # Giriş fiyatı
        self._entry = QDoubleSpinBox()
        self._entry.setDecimals(6)
        self._entry.setRange(0.0, 9_999_999.0)
        self._entry.setSpecialValueText("Auto (güncel fiyat)")
        self._entry.setValue(0.0)
        form.addRow("Giriş $:", self._entry)

        # SL
        self._sl = QDoubleSpinBox()
        self._sl.setDecimals(6)
        self._sl.setRange(0.0, 9_999_999.0)
        self._sl.setSpecialValueText("—  (yok)")
        self._sl.setValue(0.0)
        form.addRow("Stop Loss $:", self._sl)

        # TP
        self._tp = QDoubleSpinBox()
        self._tp.setDecimals(6)
        self._tp.setRange(0.0, 9_999_999.0)
        self._tp.setSpecialValueText("—  (yok)")
        self._tp.setValue(0.0)
        form.addRow("Take Profit $:", self._tp)

        root.addLayout(form)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Aç")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _set_dir(self, direction: str) -> None:
        self._direction = direction
        self._btn_long.setChecked(direction == "Long")
        self._btn_short.setChecked(direction == "Short")
        self._update_dir_style()

    def _update_dir_style(self) -> None:
        self._btn_long.setStyleSheet(
            f"QPushButton {{ background: {'#1a5c2a' if self._direction == 'Long' else COLORS['bg_tertiary']};"
            f" color: {COLORS['green'] if self._direction == 'Long' else COLORS['text_muted']};"
            f" border: 1px solid {COLORS['green'] if self._direction == 'Long' else COLORS['border']};"
            f" border-radius: 3px; font-weight: bold; }}"
        )
        self._btn_short.setStyleSheet(
            f"QPushButton {{ background: {'#5c1a1a' if self._direction == 'Short' else COLORS['bg_tertiary']};"
            f" color: {COLORS['red'] if self._direction == 'Short' else COLORS['text_muted']};"
            f" border: 1px solid {COLORS['red'] if self._direction == 'Short' else COLORS['border']};"
            f" border-radius: 3px; font-weight: bold; }}"
        )

    def _get_current_price(self, symbol: str) -> float | None:
        try:
            r = _redis.Redis.from_url(self._redis_url, socket_connect_timeout=2, decode_responses=True)
            raw = r.get(f"ticker:{symbol}")
            if raw:
                import json
                d = json.loads(raw)
                return float(d.get("price") or d.get("last_price") or 0) or None
        except Exception:
            pass
        return None

    def _on_accept(self) -> None:
        symbol = self._sym.text().strip()
        if not symbol:
            self._status.setText("Sembol boş olamaz.")
            self._status.setStyleSheet(f"color: {COLORS['red']}; font-size: 11px;")
            return

        entry = self._entry.value()
        if entry == 0.0:
            entry = self._get_current_price(symbol)
            if not entry:
                self._status.setText(f"{symbol} için fiyat alınamadı. Manuel gir.")
                self._status.setStyleSheet(f"color: {COLORS['red']}; font-size: 11px;")
                return

        sl = self._sl.value() or None
        tp = self._tp.value() or None
        tf = self._tf.currentText()
        now = datetime.now()

        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO paper_trades
                        (strategy, source, symbol, signal_type, interval,
                         position_usd, entry_price, stop_loss_price, take_profit_price,
                         status, opened_at)
                    VALUES
                        ('manual', 'manual', %s, %s, %s,
                         %s, %s, %s, %s,
                         'open', %s)
                """, (symbol, self._direction, tf, _POSITION_USD, entry, sl, tp, now))
            conn.commit()
            conn.close()
        except Exception as exc:
            self._status.setText(f"DB hatası: {exc}")
            self._status.setStyleSheet(f"color: {COLORS['red']}; font-size: 11px;")
            logger.error("[ManualTrade] DB insert hatası: %s", exc)
            return

        try:
            r = _redis.Redis.from_url(self._redis_url, socket_connect_timeout=2)
            r.set("manual_trade:refresh", "1", ex=60)
        except Exception:
            pass

        logger.info("[ManualTrade] %s %s %s @ %.6f açıldı", symbol, self._direction, tf, entry)
        self.accept()
