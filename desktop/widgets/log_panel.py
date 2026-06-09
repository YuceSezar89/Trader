"""
Log paneli — log dosyasını tail ederek tüm process loglarını gösterir.
"""

import os

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget

from desktop.theme import COLORS

_LOG_PATH = os.path.join("logs", "trader_panel.log")

_INFO_KEYWORDS = (
    "sinyal", "filtreden", "senkronizasyon", "websocket", "bağlantı",
    "bağlandı", "koptu", "mum kaydedildi", "tamamlandı", "başlatılıyor",
    "yüklendi", "mtf batch", "signal", "connected", "disconnected",
    "initialization", "✅", "❌", "🚀",
)

_LEVEL_COLORS = {
    "ERROR":   "#ef5350",
    "WARNING": "#ffa726",
    "INFO":    "#c9d1d9",
    "DEBUG":   "#6e7681",
}


def _should_show(line: str) -> bool:
    if " [ERROR] " in line or " [WARNING] " in line:
        return True
    if " [INFO] " in line:
        lower = line.lower()
        return any(kw in lower for kw in _INFO_KEYWORDS)
    return False


def _level_color(line: str) -> str:
    for level, color in _LEVEL_COLORS.items():
        if f" [{level}] " in line:
            return color
    return _LEVEL_COLORS["INFO"]


class LogPanel(QWidget):
    """Log dosyasını tail ederek logları gösteren panel."""

    _MAX_LINES = 400
    _POLL_MS   = 800

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pos = 0  # dosyada okunan byte pozisyonu

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            f"background: {COLORS['bg_primary']}; color: {COLORS['text_primary']};"
            "font-family: monospace; font-size: 11px; border: none;"
        )

        btn_clear = QPushButton("Temizle")
        btn_clear.setFixedWidth(70)
        btn_clear.setStyleSheet(
            f"color: {COLORS['text_muted']}; background: {COLORS['bg_secondary']};"
            "border: none; padding: 2px 6px; font-size: 11px;"
        )
        btn_clear.clicked.connect(self._text.clear)

        top = QHBoxLayout()
        top.addStretch()
        top.addWidget(btn_clear)
        top.setContentsMargins(0, 0, 4, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top)
        layout.addWidget(self._text)

        # Mevcut dosya sonundan başla (geçmişi yükleme)
        self._seek_to_end()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(self._POLL_MS)

    def _seek_to_end(self) -> None:
        if os.path.exists(_LOG_PATH):
            self._pos = os.path.getsize(_LOG_PATH)

    def _poll(self) -> None:
        if not os.path.exists(_LOG_PATH):
            return
        size = os.path.getsize(_LOG_PATH)
        if size < self._pos:
            self._pos = 0  # dosya rotate edildi
        if size == self._pos:
            return

        try:
            with open(_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                new_data = f.read()
                self._pos = f.tell()
        except OSError:
            return

        for line in new_data.splitlines():
            line = line.strip()
            if not line or not _should_show(line):
                continue
            color = _level_color(line)
            # timestamp + mesaj ayır (format: "2026-06-09 14:23:01 [INFO] [module] msg")
            parts = line.split("] ", 2)
            if len(parts) == 3:
                ts_level = parts[0] + "] " + parts[1] + "]"
                msg = parts[2]
            else:
                ts_level = ""
                msg = line
            html = (
                f'<span style="color:{COLORS["text_muted"]}">{ts_level} </span>'
                f'<span style="color:{color}">{msg}</span>'
            )
            self._text.append(html)

        # Satır limitini aş
        doc = self._text.document()
        while doc.blockCount() > self._MAX_LINES:
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

        self._text.ensureCursorVisible()

    def cleanup(self) -> None:
        self._timer.stop()
