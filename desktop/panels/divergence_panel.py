"""
DivergencePanel — sinyal coinlerin fiyat Z-score zaman serisi.
Her coinin kendi EMA200'üne göre: z = (close - EMA) / StdDev
"""

from datetime import datetime
from itertools import cycle
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

_COLS = ["Sembol", "Z-score", "Yön", "Zaman"]
_COL_IDX = {name: i for i, name in enumerate(_COLS)}

_PALETTE = [
    "#3fb950", "#58a6ff", "#f0883e", "#a371f7", "#d29922",
    "#8edbc1", "#f85149", "#92cbfa", "#ffa657", "#56d364",
    "#d2a8ff", "#ff6e96", "#79c0ff", "#ff7b72", "#39d353",
]

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])


class _NumericItem(QTableWidgetItem):  # pylint: disable=too-few-public-methods
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


class DivergencePanel(QWidget):  # pylint: disable=too-many-instance-attributes
    """Sinyal coinlerinin Z-score zaman serilerini tablo + grafik olarak gösterir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._color_map: dict[str, str] = {}
        self._color_cycle = cycle(_PALETTE)
        self._plot: pg.PlotWidget
        self._curves: dict[str, pg.PlotCurveItem]
        self._labels: dict[str, pg.TextItem]
        self._avg_pos_line: pg.InfiniteLine
        self._avg_neg_line: pg.InfiniteLine
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Kontrol çubuğu ────────────────────────────────────────────────
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

        # ── Splitter ──────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_IDX["Zaman"], QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.verticalHeader().setVisible(False)
        splitter.addWidget(self._table)

        splitter.addWidget(self._build_chart())
        splitter.setSizes([200, 400])
        layout.addWidget(splitter)

    def _build_chart(self) -> pg.PlotWidget:
        pg.setConfigOptions(antialias=True)
        date_axis = pg.DateAxisItem(orientation="bottom")
        date_axis.setStyle(tickFont=QFont("Monospace", 8))
        self._plot = pg.PlotWidget(axisItems={"bottom": date_axis})
        self._plot.setBackground(COLORS["bg_primary"])
        self._plot.setLabel("left", "Z-score", color=COLORS["text_muted"])
        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._curves = {}
        self._labels = {}

        dash = Qt.PenStyle.DashLine
        dot = Qt.PenStyle.DotLine

        # Y=0
        self._plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(COLORS["border"], width=1, style=dash),
        ))
        # ±1
        for y in (1.0, -1.0):
            self._plot.addItem(pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen(COLORS["border_hover"], width=1, style=dot),
            ))
        # ±2
        for y in (2.0, -2.0):
            self._plot.addItem(pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen(COLORS["accent"], width=1, style=dot),
            ))

        # Ortalama çizgileri (Pine Script'teki avgPos / avgNeg)
        white = pg.mkPen(QColor(255, 255, 255, 160), width=1.5, style=dash)
        self._avg_pos_line = pg.InfiniteLine(pos=0, angle=0, pen=white)
        self._avg_neg_line = pg.InfiniteLine(pos=0, angle=0, pen=white)
        self._avg_pos_line.setVisible(False)
        self._avg_neg_line.setVisible(False)
        self._plot.addItem(self._avg_pos_line)
        self._plot.addItem(self._avg_neg_line)

        return self._plot

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    def tf_combo(self) -> QComboBox:
        """Zaman dilimi seçici widget'ını döner."""
        return self._tf_combo

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def on_divergence_updated(self, result: dict) -> None:
        """Worker'dan gelen Z-score sonucunu tabloya ve grafiğe yazar."""
        self._last_result = result
        n = len(result.get("current", {}))
        self._status_label.setText(
            f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}  •  {n} sembol"
        )
        self._populate(result)

    @pyqtSlot(str)
    def on_status_updated(self, msg: str) -> None:
        """Durum etiketini günceller."""
        self._status_label.setText(msg)

    # ── Doldurma ─────────────────────────────────────────────────────────

    def _populate(self, result: dict) -> None:
        """Tablo ve grafiği güncel veriyle doldurur."""
        current = result.get("current", {})
        series = result.get("series", {})
        diverge_since = result.get("diverge_since", {})
        timestamps = result.get("timestamps", {})
        self._populate_table(current, diverge_since)
        self._populate_chart(series, current, timestamps)

    def _populate_table(self, current: dict, diverge_since: dict) -> None:
        rows = sorted(current.items(), key=lambda x: abs(x[1]), reverse=True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))

        mono = QFont("Monospace", 11)
        bold = QFont("Monospace", 11, QFont.Weight.Bold)
        now = datetime.now()

        for row, (symbol, z) in enumerate(rows):
            color_hex = self._symbol_color(symbol)
            sym_color = QColor(color_hex)

            sym_item = QTableWidgetItem(symbol)
            sym_item.setForeground(sym_color)
            sym_item.setFont(bold)
            self._table.setItem(row, _COL_IDX["Sembol"], sym_item)

            z_item = _NumericItem(f"{z:+.2f}")
            z_item.setData(Qt.ItemDataRole.UserRole, z)
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            z_item.setFont(mono)
            if z >= 2.0:
                z_item.setBackground(QColor(0, 120, 40, 140))
            elif z >= 1.0:
                z_item.setBackground(QColor(0, 80, 20, 80))
            elif z <= -2.0:
                z_item.setBackground(QColor(180, 20, 20, 140))
            elif z <= -1.0:
                z_item.setBackground(QColor(120, 10, 10, 80))
            self._table.setItem(row, _COL_IDX["Z-score"], z_item)

            if z > 0.5:
                direction, fg = "▲", _C_GREEN
            elif z < -0.5:
                direction, fg = "▼", _C_RED
            else:
                direction, fg = "—", _C_MUTED
            d_item = QTableWidgetItem(direction)
            d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            d_item.setForeground(fg)
            self._table.setItem(row, _COL_IDX["Yön"], d_item)

            ts = diverge_since.get(symbol)
            if ts:
                dt = datetime.fromtimestamp(ts)
                time_str = dt.strftime("%H:%M") if dt.date() == now.date() else dt.strftime("%m/%d %H:%M")
            else:
                time_str = "—"
            t_item = QTableWidgetItem(time_str)
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t_item.setFont(mono)
            t_item.setForeground(_C_MUTED)
            self._table.setItem(row, _COL_IDX["Zaman"], t_item)

        self._table.setSortingEnabled(True)

    def _populate_chart(self, series: dict, current: dict, timestamps: dict) -> None:
        if not series:
            for sym in list(self._curves):
                self._plot.removeItem(self._curves.pop(sym))
                self._plot.removeItem(self._labels.pop(sym))
            self._avg_pos_line.setVisible(False)
            self._avg_neg_line.setVisible(False)
            return

        # Artık olmayan sembolleri temizle
        for sym in list(self._curves):
            if sym not in series:
                self._plot.removeItem(self._curves.pop(sym))
                self._plot.removeItem(self._labels.pop(sym))

        for symbol, z_arr in series.items():
            ts = timestamps.get(symbol)
            xs = ts.tolist() if ts is not None and len(ts) == len(z_arr) else list(range(len(z_arr)))
            color_hex = self._symbol_color(symbol)
            color = QColor(color_hex)
            color.setAlpha(200)

            abs_z = abs(current.get(symbol, 0.0))
            width = 2.5 if abs_z >= 2.0 else (1.8 if abs_z >= 1.0 else 1.0)

            if symbol in self._curves:
                self._curves[symbol].setData(x=xs, y=z_arr.tolist())
                self._curves[symbol].setPen(pg.mkPen(color, width=width))
                self._labels[symbol].setPos(xs[-1], float(z_arr[-1]))
            else:
                curve = pg.PlotCurveItem(
                    x=xs, y=z_arr.tolist(),
                    pen=pg.mkPen(color, width=width),
                )
                self._plot.addItem(curve)
                self._curves[symbol] = curve

                lbl = pg.TextItem(
                    text=symbol.replace("USDT", ""),
                    color=color,
                    anchor=(0.0, 0.5),
                )
                lbl.setFont(QFont("Monospace", 8))
                lbl.setPos(xs[-1], float(z_arr[-1]))
                lbl.setZValue(15)
                self._plot.addItem(lbl)
                self._labels[symbol] = lbl

        # Ortalama çizgileri
        pos_vals = [v for v in current.values() if v > 0]
        neg_vals = [v for v in current.values() if v < 0]
        if pos_vals:
            self._avg_pos_line.setValue(float(np.mean(pos_vals)))
            self._avg_pos_line.setVisible(True)
        else:
            self._avg_pos_line.setVisible(False)
        if neg_vals:
            self._avg_neg_line.setValue(float(np.mean(neg_vals)))
            self._avg_neg_line.setVisible(True)
        else:
            self._avg_neg_line.setVisible(False)

    def _symbol_color(self, symbol: str) -> str:
        if symbol not in self._color_map:
            self._color_map[symbol] = next(self._color_cycle)
        return self._color_map[symbol]
