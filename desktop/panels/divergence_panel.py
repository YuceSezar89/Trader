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
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

_COLS = ["Sembol", "Z-score", "Rank", "Zaman"]
_COL_SYMBOL = 0
_COL_ZSCORE = 1
_COL_RANK   = 2
_COL_TIME   = 3

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
    hh.setSectionResizeMode(_COL_RANK,   QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(_COL_TIME,   QHeaderView.ResizeMode.ResizeToContents)
    return t


def _make_search_box(placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(24)
    box.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 0 4px; font-size: 11px;"
    )
    return box


class DivergencePanel(QWidget):
    """Pozitif ve negatif Z-score ayrışmalarını yan yana tablolarla gösterir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._prev_pos_ranks: dict[str, int] = {}
        self._prev_neg_ranks: dict[str, int] = {}
        self._ranking: dict[str, int] = {}
        self._pos_search = ""
        self._neg_search = ""
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

        # İki sütun yan yana
        tables_row = QHBoxLayout()
        tables_row.setSpacing(8)

        # Pozitif sütun
        pos_col = QVBoxLayout()
        pos_col.setSpacing(4)
        pos_title = QLabel("▲ POZİTİF AYRIŞMA")
        pos_title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._pos_search_box = _make_search_box("Ara…")
        self._pos_search_box.textChanged.connect(self._on_pos_search)
        pos_hdr = QHBoxLayout()
        pos_hdr.addWidget(pos_title)
        pos_hdr.addStretch()
        pos_hdr.addWidget(self._pos_search_box)
        self._pos_table = _make_table()
        pos_col.addLayout(pos_hdr)
        pos_col.addWidget(self._pos_table)

        # Negatif sütun
        neg_col = QVBoxLayout()
        neg_col.setSpacing(4)
        neg_title = QLabel("▼ NEGATİF AYRIŞMA")
        neg_title.setStyleSheet(
            f"color: {COLORS['red']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._neg_search_box = _make_search_box("Ara…")
        self._neg_search_box.textChanged.connect(self._on_neg_search)
        neg_hdr = QHBoxLayout()
        neg_hdr.addWidget(neg_title)
        neg_hdr.addStretch()
        neg_hdr.addWidget(self._neg_search_box)
        self._neg_table = _make_table()
        neg_col.addLayout(neg_hdr)
        neg_col.addWidget(self._neg_table)

        tables_row.addLayout(pos_col)
        tables_row.addLayout(neg_col)
        layout.addLayout(tables_row)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    def tf_combo(self) -> QComboBox:
        return self._tf_combo

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def on_ranking_updated(self, result: list) -> None:
        self._ranking = {r["symbol"]: r["rank"] for r in result}
        if self._last_result:
            self._populate(self._last_result)

    def _on_pos_search(self, text: str) -> None:
        self._pos_search = text.strip().upper()
        self._apply_filter(self._pos_table, self._pos_search)

    def _on_neg_search(self, text: str) -> None:
        self._neg_search = text.strip().upper()
        self._apply_filter(self._neg_table, self._neg_search)

    @staticmethod
    def _apply_filter(table: QTableWidget, search: str) -> None:
        for row in range(table.rowCount()):
            item = table.item(row, _COL_SYMBOL)
            if item is None:
                continue
            symbol = item.text().split()[0]
            table.setRowHidden(row, bool(search) and search not in symbol)

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

        pos_deltas = {sym: self._prev_pos_ranks[sym] - i for i, (sym, _) in enumerate(pos_rows) if sym in self._prev_pos_ranks}
        neg_deltas = {sym: self._prev_neg_ranks[sym] - i for i, (sym, _) in enumerate(neg_rows) if sym in self._prev_neg_ranks}

        self._prev_pos_ranks = {sym: i for i, (sym, _) in enumerate(pos_rows)}
        self._prev_neg_ranks = {sym: i for i, (sym, _) in enumerate(neg_rows)}

        self._fill_table(self._pos_table, pos_rows, diverge_since, positive=True, rank_deltas=pos_deltas)
        self._fill_table(self._neg_table, neg_rows, diverge_since, positive=False, rank_deltas=neg_deltas)
        self._apply_filter(self._pos_table, self._pos_search)
        self._apply_filter(self._neg_table, self._neg_search)

    def _fill_table(  # pylint: disable=too-many-locals
        self,
        table: QTableWidget,
        rows: list,
        diverge_since: dict,
        positive: bool,
        rank_deltas: Optional[dict] = None,
    ) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))

        mono = QFont("Monospace", 11)
        bold = QFont("Monospace", 11, QFont.Weight.Bold)
        now = datetime.now()
        z_color = _C_GREEN if positive else _C_RED
        if rank_deltas is None:
            rank_deltas = {}

        for row_idx, (symbol, z) in enumerate(rows):
            # Sembol + sıra değişimi
            delta = rank_deltas.get(symbol, 0)
            if delta > 0:
                sym_text = f"{symbol} ↑{delta}"
                sym_color = _C_GREEN
            elif delta < 0:
                sym_text = f"{symbol} ↓{abs(delta)}"
                sym_color = _C_RED
            else:
                sym_text = symbol
                sym_color = z_color
            sym_item = QTableWidgetItem(sym_text)
            sym_item.setFont(bold)
            sym_item.setForeground(sym_color)
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

            # Rank
            rank = self._ranking.get(symbol)
            rank_item = _NumericItem(str(rank) if rank is not None else "—")
            rank_item.setData(Qt.ItemDataRole.UserRole, rank if rank is not None else 9999)
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            rank_item.setFont(mono)
            rank_item.setForeground(_C_MUTED)
            table.setItem(row_idx, _COL_RANK, rank_item)

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

        table.setSortingEnabled(True)
