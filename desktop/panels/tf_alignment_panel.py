"""
TFAlignmentPanel — TF Hizalanma + Erken Ayrışma sisteminin süresiz izlenen
adaylarını gösterir (bkz. signals/tf_alignment_gate.py, 20 Tem 2026 dinamik-
eşik mimarisi).

Long ve Short ayrı tablolarda (kendi popülasyonları içinde percentile'a göre
sıralanıyor, bkz. backend). Her satırda "Aç" butonu — ManualTradeDialog'u
sembol/yön/TF/güncel fiyat önceden doldurulmuş açar.
"""

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS
from desktop.workers.tf_alignment_worker import TFAlignmentWorker

_COL_SYMBOL = 0
_COL_TF = 1
_COL_INDICATOR = 2
_COL_OPEN_PRICE = 3
_COL_EARLY_PCT = 4
_COL_ACTION = 5
_HEADERS = ["Sembol", "TF", "Gösterge", "Açılış", "Erken %", ""]

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


class _DirectionTable(QTableWidget):
    """Long veya Short adaylarını gösteren tek bir tablo."""

    def __init__(self, direction: str, on_open, parent=None):
        super().__init__(0, len(_HEADERS), parent)
        self._direction = direction
        self._on_open = on_open
        self.setHorizontalHeaderLabels(_HEADERS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(False)
        self.setSortingEnabled(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(
            f"""
            QTableWidget {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: none;
                font-size: 12px;
            }}
            QHeaderView::section {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_muted']};
                border: none;
                padding: 4px;
                font-size: 11px;
            }}
            QTableWidget::item:selected {{
                background: {COLORS['bg_tertiary']};
            }}
            """
        )
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.Stretch)

    def render(self, rows: list, search_text: str) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))

        for row_idx, row_data in enumerate(rows):
            sym_item = QTableWidgetItem(row_data.get("symbol", ""))
            sym_item.setForeground(_C_WHITE)
            self.setItem(row_idx, _COL_SYMBOL, sym_item)

            tf_item = QTableWidgetItem(row_data.get("interval", ""))
            tf_item.setForeground(_C_MUTED)
            tf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row_idx, _COL_TF, tf_item)

            ind_item = QTableWidgetItem(row_data.get("indicators", ""))
            ind_item.setForeground(_C_MUTED)
            ind_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row_idx, _COL_INDICATOR, ind_item)

            open_price = row_data.get("open_price")
            price_item = _NumericItem(f"{open_price:.6g}" if open_price is not None else "—")
            price_item.setData(Qt.ItemDataRole.UserRole, open_price or 0)
            price_item.setForeground(_C_MUTED)
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row_idx, _COL_OPEN_PRICE, price_item)

            early_pct = row_data.get("early_pct")
            if early_pct is None:
                early_item = QTableWidgetItem("—")
                early_item.setForeground(_C_MUTED)
                early_item.setData(Qt.ItemDataRole.UserRole, 0)
            else:
                sign = "+" if early_pct > 0 else ""
                early_item = _NumericItem(f"{sign}{early_pct:.3f}")
                early_item.setData(Qt.ItemDataRole.UserRole, early_pct)
                early_item.setForeground(_C_GREEN if early_pct > 0 else _C_RED)
            early_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row_idx, _COL_EARLY_PCT, early_item)

            # Mevcut satırda buton varsa YENİDEN KULLAN — her render()'da
            # setCellWidget ile yeni QPushButton yaratmak, her birinin
            # setVisible(True)'da ensurePolished() → tüm QSS stylesheet'in
            # YENİDEN AYRIŞTIRILMASINI (QCss::Parser::parse) tetikliyordu.
            # 20sn'de bir ~225 aday × yeni buton = sürekli CPU/bellek
            # tüketimi — `sample` ile yakalanan masaüstü donma kök nedeni
            # (26 Tem 2026).
            open_btn = self.cellWidget(row_idx, _COL_ACTION)
            if not isinstance(open_btn, QPushButton):
                open_btn = QPushButton("Aç")
                open_btn.setFixedHeight(22)
                open_btn.setStyleSheet(
                    f"QPushButton {{ background: {COLORS['bg_tertiary']}; color: {COLORS['accent']};"
                    f" border: 1px solid {COLORS['border']}; border-radius: 3px; font-size: 11px; }}"
                    f" QPushButton:hover {{ background: #2a2a3a; }}"
                )
                open_btn.clicked.connect(self._on_open_clicked)
                self.setCellWidget(row_idx, _COL_ACTION, open_btn)
            open_btn.setProperty("row_data", row_data)

        self.setSortingEnabled(True)
        self.resizeColumnsToContents()
        self._apply_search_filter(search_text)

    def _on_open_clicked(self) -> None:
        btn = self.sender()
        row_data = btn.property("row_data") if btn is not None else None
        if row_data:
            self._on_open(row_data)

    def _apply_search_filter(self, search_text: str) -> None:
        for row in range(self.rowCount()):
            item = self.item(row, _COL_SYMBOL)
            if item is None:
                continue
            hidden = bool(search_text) and search_text not in item.text().upper()
            self.setRowHidden(row, hidden)


class TFAlignmentPanel(QWidget):
    def __init__(self, db_config: dict, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._db_config = db_config or {}
        self._worker = TFAlignmentWorker(redis_url, parent=self)
        self._search_text = ""
        self._last_rows: list = []
        self._setup_ui()
        self._connect_worker()
        self._worker.start()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        title = QLabel("TF Hizalanma Adayları")
        title.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")

        self._status = QLabel("Yükleniyor…")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Ara…")
        self._search_box.setFixedHeight(24)
        self._search_box.setFixedWidth(110)
        self._search_box.setStyleSheet(
            f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 0 4px; font-size: 11px;"
        )
        self._search_box.textChanged.connect(self._on_search_changed)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setStyleSheet(
            f"color: {COLORS['accent']}; background: transparent; border: none; font-size: 14px;"
        )
        refresh_btn.clicked.connect(self._worker.refresh)

        top.addWidget(title)
        top.addStretch()
        top.addWidget(self._search_box)
        top.addWidget(self._status)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        tables_row = QHBoxLayout()
        tables_row.setSpacing(6)

        long_col = QVBoxLayout()
        long_label = QLabel("Long")
        long_label.setStyleSheet(f"color: {COLORS['green']}; font-weight: bold; font-size: 11px;")
        self._long_table = _DirectionTable("Long", self._open_manual_trade)
        long_col.addWidget(long_label)
        long_col.addWidget(self._long_table)

        short_col = QVBoxLayout()
        short_label = QLabel("Short")
        short_label.setStyleSheet(f"color: {COLORS['red']}; font-weight: bold; font-size: 11px;")
        self._short_table = _DirectionTable("Short", self._open_manual_trade)
        short_col.addWidget(short_label)
        short_col.addWidget(self._short_table)

        tables_row.addLayout(long_col)
        tables_row.addLayout(short_col)
        layout.addLayout(tables_row)

    def _connect_worker(self) -> None:
        self._worker.candidates_updated.connect(self._on_updated)
        self._worker.status_updated.connect(self._on_status)

    @pyqtSlot(object)
    def _on_updated(self, rows: list) -> None:
        self._last_rows = rows
        self._render(rows)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status.setText(msg)

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip().upper()
        self._render(self._last_rows)

    def _render(self, rows: list) -> None:
        long_rows = [r for r in rows if r.get("signal_type") == "Long"]
        short_rows = [r for r in rows if r.get("signal_type") == "Short"]
        self._long_table.render(long_rows, self._search_text)
        self._short_table.render(short_rows, self._search_text)

    def _open_manual_trade(self, row_data: dict) -> None:
        from desktop.dialogs.manual_trade_dialog import ManualTradeDialog

        dlg = ManualTradeDialog(
            self._db_config,
            self._redis_url,
            parent=self,
            prefill_symbol=row_data.get("symbol", ""),
            prefill_direction=row_data.get("signal_type", "Long"),
            prefill_interval=row_data.get("interval", ""),
        )
        dlg.exec()
