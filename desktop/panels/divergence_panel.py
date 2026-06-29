"""
DivergencePanel — sinyal coinlerin fiyat Z-score ayrışma tablosu + çizgi grafik.
Z = (close - EMA200) / StdDev200

Üst: tüm sembollerin z-score zaman serisi çizgileri (Pine benzeri)
Alt sol: pozitif ayrışanlar (Z > 0), büyükten küçüğe
Alt sağ: negatif ayrışanlar (Z < 0), küçükten büyüğe
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

pg.setConfigOption("background", "#0d0d12")
pg.setConfigOption("foreground", "#555566")

_COLS = ["Sembol", "Z-score", "VPMV", "Rank", "Zaman"]
_COL_SYMBOL = 0
_COL_ZSCORE = 1
_COL_VPMV   = 2
_COL_RANK   = 3
_COL_TIME   = 4

_C_VPMV_HIGH = QColor(80,  200,  80)
_C_VPMV_MID  = QColor(200, 180,  60)
_C_VPMV_LOW  = QColor(200,  80,  80)

_C_GREEN       = QColor(COLORS["green"])
_C_RED         = QColor(COLORS["red"])
_C_MUTED       = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)

_BG_GREEN_STRONG = QColor(0, 120, 40, 150)
_BG_GREEN_SOFT   = QColor(0, 80, 20, 80)
_BG_RED_STRONG   = QColor(180, 20, 20, 150)
_BG_RED_SOFT     = QColor(120, 10, 10, 80)

_PALETTE = [
    (100, 220, 100), (220, 100, 100), (100, 160, 240),
    (240, 180,  80), (180, 100, 240), (80,  220, 220),
    (240, 240,  80), (240, 140, 180), (140, 240, 160),
    (200, 160, 240), (240, 140,  80), (80,  180, 200),
    (160, 240, 100), (240, 100, 140), (100, 200, 240),
    (200, 240, 120), (240, 200,  80), (120, 120, 240),
]


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
    hh.setSectionResizeMode(_COL_VPMV,   QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(_COL_RANK,   QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(_COL_TIME,   QHeaderView.ResizeMode.ResizeToContents)
    return t


def _make_search_box(placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(24)
    box.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 3px; "
        f"padding: 0 4px; font-size: 11px;"
    )
    return box


class DivergencePanel(QWidget):
    """Z-score çizgi grafik + pozitif/negatif ayrışma tabloları."""

    _INDICATOR_FILTERS = [
        ("Tümü",        ""),
        ("RSI Cross",   "RSI_Cross"),
        ("Supertrend",  "Supertrend"),
        ("HA Cross",    "HA_Cross"),
        ("MA200 Cross", "MA200_Cross"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._prev_pos_ranks: dict[str, int] = {}
        self._prev_neg_ranks: dict[str, int] = {}
        self._ranking: dict[str, int] = {}
        self._pos_search = ""
        self._neg_search = ""
        self._indicator_filter = ""
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._labels: dict[str, pg.TextItem] = {}
        self._sym_colors: dict[str, tuple] = {}
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Kontrol çubuğu
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(self._muted_label("TF:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setCurrentText("1h")
        self._tf_combo.setFixedWidth(70)
        ctrl.addWidget(self._tf_combo)
        ctrl.addWidget(self._muted_label("İndikatör:"))
        self._ind_combo = QComboBox()
        for label, _ in self._INDICATOR_FILTERS:
            self._ind_combo.addItem(label)
        self._ind_combo.setFixedWidth(110)
        self._ind_combo.currentIndexChanged.connect(self._on_indicator_changed)
        ctrl.addWidget(self._ind_combo)
        ctrl.addStretch()
        self._status_label = QLabel("Sinyal bekleniyor…")
        self._status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        ctrl.addWidget(self._status_label)
        root.addLayout(ctrl)

        # Tab widget
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # ── Tab 0: Tablo ──────────────────────────────────────────────────
        tables_widget = QWidget()
        tables_layout = QHBoxLayout(tables_widget)
        tables_layout.setContentsMargins(0, 4, 0, 0)
        tables_layout.setSpacing(8)

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

        tables_layout.addLayout(pos_col)
        tables_layout.addLayout(neg_col)

        tabs.addTab(tables_widget, "Tablo")

        # ── Tab 1: Grafik ─────────────────────────────────────────────────
        self._chart = pg.PlotWidget()
        self._chart.showGrid(x=False, y=True, alpha=0.15)
        self._chart.getAxis("bottom").hide()
        self._chart.getAxis("left").setStyle(tickFont=QFont("Courier New", 9))
        self._chart.setLabel("left", "Z-score", color="#555566", size="9pt")
        self._chart.addLine(y=0, pen=pg.mkPen("#888888", width=1))
        for y_val in [1, -1, 2, -2]:
            self._chart.addLine(
                y=y_val,
                pen=pg.mkPen("#444455", width=1, style=Qt.PenStyle.DashLine),
            )

        tabs.addTab(self._chart, "Grafik")

        root.addWidget(tabs)

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

    def _on_indicator_changed(self, index: int) -> None:
        self._indicator_filter = self._INDICATOR_FILTERS[index][1]
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

    # ── Grafik güncelleme ─────────────────────────────────────────────────

    def _update_chart(self, series: dict) -> None:
        active = set(series.keys())
        stale  = set(self._curves.keys()) - active

        # Eski sembolleri kaldır
        for sym in stale:
            self._chart.removeItem(self._curves.pop(sym))
            if sym in self._labels:
                self._chart.removeItem(self._labels.pop(sym))
            self._sym_colors.pop(sym, None)

        # Renk ata (yeni sembollere)
        color_idx = 0
        for sym in sorted(active):
            if sym not in self._sym_colors:
                self._sym_colors[sym] = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1

        # Çizgileri güncelle / oluştur
        for sym, z_arr in series.items():
            color = self._sym_colors[sym]
            x_data = np.arange(len(z_arr), dtype=np.float32)
            y_data = z_arr.astype(np.float32)

            if sym in self._curves:
                self._curves[sym].setData(x_data, y_data)
            else:
                pen = pg.mkPen(color=color, width=1.5)
                curve = self._chart.plot(x_data, y_data, pen=pen)
                self._curves[sym] = curve

            # Uç etiket
            if len(z_arr) == 0:
                continue
            label_text = sym.replace("USDT", "")
            if sym in self._labels:
                self._labels[sym].setPos(float(x_data[-1]), float(y_data[-1]))
                self._labels[sym].setText(label_text)
            else:
                txt = pg.TextItem(
                    text=label_text,
                    color=color,
                    anchor=(0, 0.5),
                )
                txt.setFont(QFont("Courier New", 8))
                txt.setPos(float(x_data[-1]), float(y_data[-1]))
                self._chart.addItem(txt)
                self._labels[sym] = txt

    # ── Tablo doldurma ────────────────────────────────────────────────────

    def _populate(self, result: dict) -> None:
        current        = result.get("current", {})
        diverge_since  = result.get("diverge_since", {})
        indicators_map = result.get("indicators", {})
        series         = result.get("series", {})
        vpmv_map       = result.get("vpmv", {})

        ind_filter = self._indicator_filter
        if ind_filter:
            current = {
                sym: z for sym, z in current.items()
                if ind_filter in (indicators_map.get(sym) or "")
            }
            series = {
                sym: s for sym, s in series.items()
                if ind_filter in (indicators_map.get(sym) or "")
            }

        self._update_chart(series)

        def _score(sym: str, z: float) -> float:
            return (vpmv_map.get(sym) or 0.0) * abs(z)

        pos_rows = sorted(
            [(sym, z) for sym, z in current.items() if z > 0],
            key=lambda x: _score(x[0], x[1]), reverse=True,
        )
        neg_rows = sorted(
            [(sym, z) for sym, z in current.items() if z <= 0],
            key=lambda x: _score(x[0], x[1]), reverse=True,
        )

        pos_deltas = {
            sym: self._prev_pos_ranks[sym] - i
            for i, (sym, _) in enumerate(pos_rows)
            if sym in self._prev_pos_ranks
        }
        neg_deltas = {
            sym: self._prev_neg_ranks[sym] - i
            for i, (sym, _) in enumerate(neg_rows)
            if sym in self._prev_neg_ranks
        }

        self._prev_pos_ranks = {sym: i for i, (sym, _) in enumerate(pos_rows)}
        self._prev_neg_ranks = {sym: i for i, (sym, _) in enumerate(neg_rows)}

        self._fill_table(self._pos_table, pos_rows, diverge_since, vpmv_map,
                         positive=True,  rank_deltas=pos_deltas)
        self._fill_table(self._neg_table, neg_rows, diverge_since, vpmv_map,
                         positive=False, rank_deltas=neg_deltas)
        self._apply_filter(self._pos_table, self._pos_search)
        self._apply_filter(self._neg_table, self._neg_search)

    def _fill_table(  # pylint: disable=too-many-locals
        self,
        table: QTableWidget,
        rows: list,
        diverge_since: dict,
        vpmv_map: dict,
        positive: bool,
        rank_deltas: Optional[dict] = None,
    ) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))

        mono = QFont("Courier New", 11)
        bold = QFont("Courier New", 11, QFont.Weight.Bold)
        now = datetime.now()
        z_color = _C_GREEN if positive else _C_RED
        if rank_deltas is None:
            rank_deltas = {}

        for row_idx, (symbol, z) in enumerate(rows):
            delta = rank_deltas.get(symbol, 0)
            if delta > 0:
                sym_text  = f"{symbol} ↑{delta}"
                sym_color = _C_GREEN
            elif delta < 0:
                sym_text  = f"{symbol} ↓{abs(delta)}"
                sym_color = _C_RED
            else:
                sym_text  = symbol
                sym_color = z_color
            sym_item = QTableWidgetItem(sym_text)
            sym_item.setFont(bold)
            sym_item.setForeground(sym_color)
            table.setItem(row_idx, _COL_SYMBOL, sym_item)

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

            vpmv = vpmv_map.get(symbol) or 0.0
            vpmv_item = _NumericItem(f"{vpmv:.0f}")
            vpmv_item.setData(Qt.ItemDataRole.UserRole, vpmv)
            vpmv_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            vpmv_item.setFont(mono)
            if vpmv >= 60:
                vpmv_item.setForeground(_C_VPMV_HIGH)
            elif vpmv >= 45:
                vpmv_item.setForeground(_C_VPMV_MID)
            else:
                vpmv_item.setForeground(_C_VPMV_LOW)
            table.setItem(row_idx, _COL_VPMV, vpmv_item)

            rank = self._ranking.get(symbol)
            rank_item = _NumericItem(str(rank) if rank is not None else "—")
            rank_item.setData(Qt.ItemDataRole.UserRole,
                              rank if rank is not None else 9999)
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            rank_item.setFont(mono)
            rank_item.setForeground(_C_MUTED)
            table.setItem(row_idx, _COL_RANK, rank_item)

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
