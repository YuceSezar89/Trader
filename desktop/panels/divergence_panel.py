"""
DivergencePanel — sinyal coinlerin fiyat Z-score ayrışma tablosu.
Z = (close - EMA200) / StdDev200

Sol tablo: pozitif ayrışanlar (Z > 0), büyükten küçüğe
Sağ tablo: negatif ayrışanlar (Z < 0), küçükten büyüğe (en negatif üstte)
"""

from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

_COLS = ["Sembol", "Z-score", "Zaman"]
_COL_SYMBOL = 0
_COL_ZSCORE = 1
_COL_TIME   = 2

_C_GREEN      = QColor(COLORS["green"])
_C_RED        = QColor(COLORS["red"])
_C_MUTED      = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)

_BG_GREEN_STRONG = QColor(0, 120, 40, 150)
_BG_GREEN_SOFT   = QColor(0, 80, 20, 80)
_BG_RED_STRONG   = QColor(180, 20, 20, 150)
_BG_RED_SOFT     = QColor(120, 10, 10, 80)


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


def _make_table() -> QTableWidget:
    t = QTableWidget(0, len(_COLS))
    t.setHorizontalHeaderLabels(_COLS)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(False)
    t.setSortingEnabled(False)
    t.setShowGrid(False)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(_COL_TIME,   QHeaderView.ResizeMode.ResizeToContents)
    return t


class DivergencePanel(QWidget):
    """Pozitif ve negatif Z-score ayrışmalarını yan yana tablolarla gösterir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Kontrol çubuğu
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(self._muted_label("TF:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setCurrentText("1h")
        self._tf_combo.setFixedWidth(70)
        ctrl.addWidget(self._tf_combo)
        ctrl.addStretch()
        self._status_label = QLabel("Sinyal bekleniyor…")
        self._status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        ctrl.addWidget(self._status_label)
        layout.addLayout(ctrl)

        # Başlık satırı
        header_row = QHBoxLayout()
        pos_title = QLabel("▲ POZİTİF AYRIŞMA")
        pos_title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        neg_title = QLabel("▼ NEGATİF AYRIŞMA")
        neg_title.setStyleSheet(
            f"color: {COLORS['red']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        header_row.addWidget(pos_title)
        header_row.addWidget(neg_title)
        layout.addLayout(header_row)

        # İki tablo yan yana
        tables_row = QHBoxLayout()
        tables_row.setSpacing(6)
        self._pos_table = _make_table()
        self._neg_table = _make_table()
        tables_row.addWidget(self._pos_table)
        tables_row.addWidget(self._neg_table)
        layout.addLayout(tables_row)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    def tf_combo(self) -> QComboBox:
        return self._tf_combo

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def on_divergence_updated(self, result: dict) -> None:
        self._last_result = result
        n = len(result.get("current", {}))
        self._status_label.setText(
            f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}  •  {n} sembol"
        )
        self._populate(result)

    @pyqtSlot(str)
    def on_status_updated(self, msg: str) -> None:
        self._status_label.setText(msg)

    # ── Doldurma ─────────────────────────────────────────────────────────

    def _populate(self, result: dict) -> None:
        current = result.get("current", {})
        diverge_since = result.get("diverge_since", {})

        pos_rows = sorted(
            [(sym, z) for sym, z in current.items() if z > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        neg_rows = sorted(
            [(sym, z) for sym, z in current.items() if z <= 0],
            key=lambda x: x[1],
        )

        self._fill_table(self._pos_table, pos_rows, diverge_since, positive=True)
        self._fill_table(self._neg_table, neg_rows, diverge_since, positive=False)

    def _fill_table(
        self,
        table: QTableWidget,
        rows: list,
        diverge_since: dict,
        positive: bool,
    ) -> None:
        table.setRowCount(len(rows))

        mono = QFont("Monospace", 11)
        bold = QFont("Monospace", 11, QFont.Weight.Bold)
        now = datetime.now()
        z_color = _C_GREEN if positive else _C_RED

        for row_idx, (symbol, z) in enumerate(rows):
            # Sembol
            sym_item = QTableWidgetItem(symbol)
            sym_item.setFont(bold)
            sym_item.setForeground(z_color)
            table.setItem(row_idx, _COL_SYMBOL, sym_item)

            # Z-score + arka plan rengi
            z_item = _NumericItem(f"{z:+.2f}")
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            z_item.setFont(mono)
            z_item.setData(Qt.ItemDataRole.UserRole, z)
            z_item.setForeground(z_color)
            abs_z = abs(z)
            if abs_z >= 2.0:
                z_item.setBackground(_BG_GREEN_STRONG if positive else _BG_RED_STRONG)
            elif abs_z >= 1.0:
                z_item.setBackground(_BG_GREEN_SOFT if positive else _BG_RED_SOFT)
            else:
                z_item.setBackground(_C_TRANSPARENT)
            table.setItem(row_idx, _COL_ZSCORE, z_item)

            # Zaman
            ts = diverge_since.get(symbol)
            if ts:
                dt = datetime.fromtimestamp(ts)
                time_str = (
                    dt.strftime("%H:%M")
                    if dt.date() == now.date()
                    else dt.strftime("%m/%d %H:%M")
                )
            else:
                time_str = "—"
            t_item = QTableWidgetItem(time_str)
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t_item.setFont(mono)
            t_item.setForeground(_C_MUTED)
            table.setItem(row_idx, _COL_TIME, t_item)
