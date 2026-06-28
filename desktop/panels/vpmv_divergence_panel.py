"""
VpmvDivergencePanel — sinyal sonrası VPMV momentum ayrışması.

Delta = vpmv_şimdi - vpmv_sinyal_anı
  Pozitif → sinyal sonrası momentum büyüdü  (yukarı tablo)
  Negatif → momentum söndü                  (aşağı tablo)

Grafik tabında sinyal barından itibaren VPMV serisi (0-100).
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

_COLS       = ["Sembol", "Δ VPMV", "Şimdi", "vs Med", "Sinyal", "Pre", "Zaman"]
_COL_SYMBOL = 0
_COL_DELTA  = 1
_COL_NOW    = 2
_COL_VS_MED = 3
_COL_SIG    = 4
_COL_PRE    = 5
_COL_TIME   = 6

_C_GREEN       = QColor(COLORS["green"])
_C_RED         = QColor(COLORS["red"])
_C_MUTED       = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)

_BG_POS_STRONG = QColor(0, 120, 40, 150)
_BG_POS_SOFT   = QColor(0, 80, 20, 80)
_BG_NEG_STRONG = QColor(180, 20, 20, 150)
_BG_NEG_SOFT   = QColor(120, 10, 10, 80)

_PALETTE = [
    (100, 220, 100), (220, 100, 100), (100, 160, 240),
    (240, 180,  80), (180, 100, 240), (80,  220, 220),
    (240, 240,  80), (240, 140, 180), (140, 240, 160),
    (200, 160, 240), (240, 140,  80), (80,  180, 200),
    (160, 240, 100), (240, 100, 140), (100, 200, 240),
]

_INDICATOR_FILTERS = [
    ("Tümü",        ""),
    ("RSI Cross",   "RSI_Cross"),
    ("Supertrend",  "Supertrend"),
    ("HA Cross",    "HA_Cross"),
    ("MA200 Cross", "MA200_Cross"),
]

_TF_FILTERS = ["Tümü", "1m", "5m", "15m", "1h", "4h", "1d"]


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


def _make_search(placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(24)
    box.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 3px; "
        f"padding: 0 4px; font-size: 11px;"
    )
    return box


class VpmvDivergencePanel(QWidget):
    """Sinyal sonrası VPMV delta tablosu + çizgi grafik."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._indicator_filter = ""
        self._tf_filter = ""
        self._pos_search = ""
        self._neg_search = ""
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._labels: dict[str, pg.TextItem] = {}
        self._sym_colors: dict[str, tuple] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Kontrol çubuğu
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(self._muted("TF:"))
        self._tf_combo = QComboBox()
        for tf in _TF_FILTERS:
            self._tf_combo.addItem(tf)
        self._tf_combo.setFixedWidth(70)
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)
        ctrl.addWidget(self._tf_combo)
        ctrl.addWidget(self._muted("İndikatör:"))
        self._ind_combo = QComboBox()
        for label, _ in _INDICATOR_FILTERS:
            self._ind_combo.addItem(label)
        self._ind_combo.setFixedWidth(110)
        self._ind_combo.currentIndexChanged.connect(self._on_ind_changed)
        ctrl.addWidget(self._ind_combo)
        ctrl.addStretch()
        self._status_label = QLabel("Sinyal bekleniyor…")
        self._status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        ctrl.addWidget(self._status_label)
        root.addLayout(ctrl)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # ── Tab 0: Tablo ──────────────────────────────────────────────────
        tables_w = QWidget()
        tbl_lay  = QHBoxLayout(tables_w)
        tbl_lay.setContentsMargins(0, 4, 0, 0)
        tbl_lay.setSpacing(8)

        pos_col = QVBoxLayout()
        pos_col.setSpacing(4)
        pos_title = QLabel("▲ MOMENTUM DEVAM EDİYOR")
        pos_title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._pos_search_box = _make_search("Ara…")
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
        neg_title = QLabel("▼ MOMENTUM SÖNDÜ")
        neg_title.setStyleSheet(
            f"color: {COLORS['red']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._neg_search_box = _make_search("Ara…")
        self._neg_search_box.textChanged.connect(self._on_neg_search)
        neg_hdr = QHBoxLayout()
        neg_hdr.addWidget(neg_title)
        neg_hdr.addStretch()
        neg_hdr.addWidget(self._neg_search_box)
        self._neg_table = _make_table()
        neg_col.addLayout(neg_hdr)
        neg_col.addWidget(self._neg_table)

        tbl_lay.addLayout(pos_col)
        tbl_lay.addLayout(neg_col)
        tabs.addTab(tables_w, "Tablo")

        # ── Tab 1: Grafik ─────────────────────────────────────────────────
        self._chart = pg.PlotWidget()
        self._chart.showGrid(x=False, y=True, alpha=0.15)
        self._chart.getAxis("bottom").hide()
        self._chart.getAxis("left").setStyle(tickFont=QFont("Courier New", 9))
        self._chart.setLabel("left", "VPMV", color="#555566", size="9pt")
        self._chart.setYRange(0, 100)
        for y_val in [25, 50, 75]:
            self._chart.addLine(
                y=y_val,
                pen=pg.mkPen("#333344", width=1, style=Qt.PenStyle.DashLine),
            )
        self._chart.addLine(y=50, pen=pg.mkPen("#555566", width=1))
        tabs.addTab(self._chart, "Grafik")

        root.addWidget(tabs)

    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def on_status_updated(self, msg: str) -> None:
        self._status_label.setText(msg)

    @pyqtSlot(object)
    def on_vpmv_updated(self, result: dict) -> None:
        self._last_result = result
        current = result.get("current", {})
        n = len(current)
        med = float(np.median(list(current.values()))) if current else 0.0
        self._status_label.setText(
            f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}"
            f"  •  {n} sembol  •  Med: {med:.0f}"
        )
        self._populate(result)

    def _on_tf_changed(self, text: str) -> None:
        self._tf_filter = "" if text == "Tümü" else text
        if self._last_result:
            self._populate(self._last_result)

    def _on_ind_changed(self, index: int) -> None:
        self._indicator_filter = _INDICATOR_FILTERS[index][1]
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
            table.setRowHidden(row, bool(search) and search not in item.text())

    # ── Grafik ────────────────────────────────────────────────────────────

    def _update_chart(self, series: dict) -> None:
        active = set(series.keys())
        for sym in set(self._curves) - active:
            self._chart.removeItem(self._curves.pop(sym))
            if sym in self._labels:
                self._chart.removeItem(self._labels.pop(sym))
            self._sym_colors.pop(sym, None)

        for i, sym in enumerate(sorted(active)):
            if sym not in self._sym_colors:
                self._sym_colors[sym] = _PALETTE[i % len(_PALETTE)]

        for sym, vpmv_arr in series.items():
            color  = self._sym_colors[sym]
            x_data = np.arange(len(vpmv_arr), dtype=np.float32)
            y_data = vpmv_arr.astype(np.float32)

            if sym in self._curves:
                self._curves[sym].setData(x_data, y_data)
            else:
                pen = pg.mkPen(color=color, width=1.5)
                self._curves[sym] = self._chart.plot(x_data, y_data, pen=pen)

            if len(vpmv_arr) == 0:
                continue
            label_text = sym.replace("USDT", "")
            if sym in self._labels:
                self._labels[sym].setPos(float(x_data[-1]), float(y_data[-1]))
                self._labels[sym].setText(label_text)
            else:
                txt = pg.TextItem(text=label_text, color=color, anchor=(0, 0.5))
                txt.setFont(QFont("Courier New", 8))
                txt.setPos(float(x_data[-1]), float(y_data[-1]))
                self._chart.addItem(txt)
                self._labels[sym] = txt

    # ── Tablo ─────────────────────────────────────────────────────────────

    def _populate(self, result: dict) -> None:
        delta        = result.get("delta", {})
        current_vpmv = result.get("current", {})
        signal_vpmv  = result.get("signal", {})
        pre_vpmv     = result.get("pre", {})
        series       = result.get("series", {})
        indicators   = result.get("indicators", {})
        tf_map       = result.get("tf", {})
        time_map     = result.get("time", {})

        tf = self._tf_filter
        if tf:
            keep = {sym for sym, v in tf_map.items() if v == tf}
            delta        = {s: v for s, v in delta.items()        if s in keep}
            current_vpmv = {s: v for s, v in current_vpmv.items() if s in keep}
            series       = {s: v for s, v in series.items()       if s in keep}

        ind = self._indicator_filter
        if ind:
            keep = {sym for sym, v in indicators.items() if ind in v}
            delta        = {s: v for s, v in delta.items()        if s in keep}
            current_vpmv = {s: v for s, v in current_vpmv.items() if s in keep}
            series       = {s: v for s, v in series.items()       if s in keep}

        self._update_chart(series)

        all_now = list(current_vpmv.values())
        median_vpmv = float(np.median(all_now)) if all_now else 0.0

        pos_rows = sorted(
            [(sym, d) for sym, d in delta.items() if d >= 0],
            key=lambda x: x[1], reverse=True,
        )
        neg_rows = sorted(
            [(sym, d) for sym, d in delta.items() if d < 0],
            key=lambda x: x[1],
        )

        self._fill_table(
            self._pos_table, pos_rows, current_vpmv, signal_vpmv,
            pre_vpmv, time_map, median_vpmv, positive=True,
        )
        self._fill_table(
            self._neg_table, neg_rows, current_vpmv, signal_vpmv,
            pre_vpmv, time_map, median_vpmv, positive=False,
        )
        self._apply_filter(self._pos_table, self._pos_search)
        self._apply_filter(self._neg_table, self._neg_search)

    def _fill_table(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        table: QTableWidget,
        rows: list,
        current_vpmv: dict,
        signal_vpmv: dict,
        pre_vpmv: dict,
        time_map: dict,
        median_vpmv: float,
        positive: bool,
    ) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))

        mono     = QFont("Courier New", 11)
        bold     = QFont("Courier New", 11, QFont.Weight.Bold)
        d_color  = _C_GREEN if positive else _C_RED
        now      = datetime.now()

        for row_idx, (symbol, delta) in enumerate(rows):
            sym_item = QTableWidgetItem(symbol)
            sym_item.setFont(bold)
            sym_item.setForeground(d_color)
            table.setItem(row_idx, _COL_SYMBOL, sym_item)

            d_item = _NumericItem(f"{delta:+.1f}")
            d_item.setData(Qt.ItemDataRole.UserRole, delta)
            d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            d_item.setFont(mono)
            d_item.setForeground(d_color)
            abs_d = abs(delta)
            if abs_d >= 20:
                d_item.setBackground(_BG_POS_STRONG if positive else _BG_NEG_STRONG)
            elif abs_d >= 10:
                d_item.setBackground(_BG_POS_SOFT if positive else _BG_NEG_SOFT)
            else:
                d_item.setBackground(_C_TRANSPARENT)
            table.setItem(row_idx, _COL_DELTA, d_item)

            for col, val in (
                (_COL_NOW, current_vpmv.get(symbol, 0.0)),
                (_COL_SIG, signal_vpmv.get(symbol, 0.0)),
                (_COL_PRE, pre_vpmv.get(symbol, 0.0)),
            ):
                it = _NumericItem(f"{val:.0f}")
                it.setData(Qt.ItemDataRole.UserRole, val)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setFont(mono)
                it.setForeground(_C_MUTED)
                table.setItem(row_idx, col, it)

            vs_med = current_vpmv.get(symbol, 0.0) - median_vpmv
            vs_item = _NumericItem(f"{vs_med:+.0f}")
            vs_item.setData(Qt.ItemDataRole.UserRole, vs_med)
            vs_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            vs_item.setFont(mono)
            vs_item.setForeground(_C_GREEN if vs_med >= 0 else _C_RED)
            if abs(vs_med) >= 15:
                vs_item.setBackground(
                    _BG_POS_STRONG if vs_med >= 0 else _BG_NEG_STRONG
                )
            elif abs(vs_med) >= 8:
                vs_item.setBackground(
                    _BG_POS_SOFT if vs_med >= 0 else _BG_NEG_SOFT
                )
            else:
                vs_item.setBackground(_C_TRANSPARENT)
            table.setItem(row_idx, _COL_VS_MED, vs_item)

            sig_dt = time_map.get(symbol)
            if sig_dt:
                time_str = (
                    sig_dt.strftime("%H:%M")
                    if sig_dt.date() == now.date()
                    else sig_dt.strftime("%m/%d %H:%M")
                )
            else:
                time_str = "—"
            t_item = QTableWidgetItem(time_str)
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t_item.setFont(mono)
            t_item.setForeground(_C_MUTED)
            table.setItem(row_idx, _COL_TIME, t_item)

        table.setSortingEnabled(True)
