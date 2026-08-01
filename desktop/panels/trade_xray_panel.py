"""
TradeXRayPanel — açık/kapalı işlemlerin trade_snapshots zaman serisini
grafik olarak gösterir (20 Tem 2026, "her işlemin röntgenini çekelim").

Üstte tüm işlemler (açık+kapalı, tüm stratejiler) listesi, satıra tıklayınca
altta 4 alt-grafik yükleniyor: PnL%, VPMV bileşenleri, CVD, VP skorları.
DB sorguları QThread'de (UI donmasın diye — bkz. bugünkü panel donma dersi).
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import Optional

import psycopg2
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

logger = logging.getLogger(__name__)

pg.setConfigOption("background", "#0d0d12")
pg.setConfigOption("foreground", "#555566")
pg.setConfigOptions(antialias=True)

_TITLE_STYLE = 'style="color:#c8c8d2;font-size:11pt;font-weight:600;"'
_AXIS_COLOR = "#6a6a7a"
_GRID_ALPHA = 0.12
_ZERO_PEN = pg.mkPen((70, 70, 85), width=1, style=Qt.PenStyle.DashLine)
_CROSSHAIR_PEN = pg.mkPen((120, 120, 140), width=1, style=Qt.PenStyle.DotLine)

_TRADE_COLS = ["Sembol", "Strateji", "Yön", "Durum", "Açılış", "PnL%"]
_COL_SYMBOL, _COL_STRATEGY, _COL_SIDE, _COL_STATUS, _COL_OPENED, _COL_PNL = range(6)

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])

_VPMV_SERIES = [
    ("vol_score", (100, 220, 100), "V"),
    ("mom_score", (100, 160, 240), "M"),
    ("volat_score", (240, 180, 80), "Vlt"),
    ("price_score", (220, 100, 100), "P"),
    ("vpmv_combined", (240, 240, 80), "Combined"),
]


class _TradeListWorker(QThread):
    trades_loaded = pyqtSignal(object)

    def __init__(self, db_config: dict, parent=None):
        super().__init__(parent)
        self._db_config = db_config

    def run(self) -> None:
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.id, p.symbol, p.strategy, p.signal_type, p.status, p.opened_at,
                           COALESCE(p.pnl_pct, latest.price_since_entry_pct) AS pnl_pct,
                           p.entry_features
                    FROM paper_trades p
                    LEFT JOIN LATERAL (
                        SELECT price_since_entry_pct FROM trade_snapshots ts
                        WHERE ts.trade_id = p.id ORDER BY taken_at DESC LIMIT 1
                    ) latest ON true
                    ORDER BY p.opened_at DESC
                    LIMIT 500
                    """
                )
                rows = cur.fetchall()
            conn.close()
            trades = [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "strategy": r[2],
                    "signal_type": r[3],
                    "status": r[4],
                    "opened_at": r[5],
                    "pnl_pct": r[6],
                    "entry_features": r[7] if isinstance(r[7], dict) else (json.loads(r[7]) if r[7] else None),
                }
                for r in rows
            ]
            self.trades_loaded.emit(trades)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[TradeXRay] işlem listesi yüklenemedi: %s", exc)
            self.trades_loaded.emit([])


class _SnapshotWorker(QThread):
    snapshots_loaded = pyqtSignal(object)

    def __init__(self, db_config: dict, trade_id: int, parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._trade_id = trade_id

    def run(self) -> None:
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT taken_at, price, cvd_slope, vp_score, vp_score_real,
                           vol_score, mom_score, volat_score, price_score,
                           price_since_entry_pct, vpmv_combined, smc_market_structure
                    FROM trade_snapshots WHERE trade_id = %s ORDER BY taken_at ASC
                    """,
                    (self._trade_id,),
                )
                rows = cur.fetchall()
            conn.close()
            snaps = [
                {
                    "taken_at": r[0],
                    "price": r[1],
                    "cvd_slope": r[2],
                    "vp_score": r[3],
                    "vp_score_real": r[4],
                    "vol_score": r[5],
                    "mom_score": r[6],
                    "volat_score": r[7],
                    "price_score": r[8],
                    "price_since_entry_pct": r[9],
                    "vpmv_combined": r[10],
                    "smc_market_structure": r[11],
                }
                for r in rows
            ]
            self.snapshots_loaded.emit(snaps)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[TradeXRay] snapshot yüklenemedi: %s", exc)
            self.snapshots_loaded.emit([])


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


class TradeXRayPanel(QWidget):
    def __init__(self, db_config: dict, parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._trades: list[dict] = []
        self._filtered_trades: list[dict] = []
        self._search_text = ""
        self._status_filter = "Tümü"
        self._side_filter = "Tümü"
        self._strategy_filter = "Tümü"
        self._status_buttons: dict[str, QPushButton] = {}
        self._side_buttons: dict[str, QPushButton] = {}
        self._list_worker: Optional[_TradeListWorker] = None
        self._snap_worker: Optional[_SnapshotWorker] = None
        self._setup_ui()
        self.refresh()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(15_000)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start()

    @staticmethod
    def _make_combo(options: list[str]) -> QComboBox:
        cb = QComboBox()
        cb.addItems(options)
        cb.setFixedHeight(24)
        cb.setStyleSheet(
            f"""
            QComboBox {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                border: 1px solid #444; border-radius: 3px;
                font-size: 11px; padding: 0 6px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']};
            }}
            """
        )
        return cb

    @staticmethod
    def _style_filter_btn(btn: QPushButton) -> None:
        btn.setCheckable(True)
        btn.setFixedHeight(22)
        btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['bg_tertiary']}; color: {COLORS['text_muted']};"
            f" border: 1px solid {COLORS['border']}; border-radius: 3px; font-size: 11px; padding: 0 8px; }}"
            f" QPushButton:checked {{ background: {COLORS['bg_secondary']}; color: {COLORS['accent']};"
            f" border: 1px solid {COLORS['accent']}; font-weight: bold; }}"
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        title = QLabel("İşlem Röntgeni")
        title.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")

        self._status = QLabel("")
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
        refresh_btn.clicked.connect(self.refresh)

        top.addWidget(title)
        top.addStretch()
        top.addWidget(self._search_box)
        top.addWidget(self._status)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        durum_label = QLabel("Durum:")
        durum_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        filter_row.addWidget(durum_label)
        for label, value in (("Tümü", "Tümü"), ("Açık", "open"), ("Kapalı", "closed")):
            btn = QPushButton(label)
            self._style_filter_btn(btn)
            btn.setChecked(value == "Tümü")
            btn.clicked.connect(lambda _checked=False, v=value: self._set_status_filter(v))
            self._status_buttons[value] = btn
            filter_row.addWidget(btn)

        filter_row.addSpacing(14)
        yon_label = QLabel("Yön:")
        yon_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        filter_row.addWidget(yon_label)
        for label, value in (("Tümü", "Tümü"), ("Long", "Long"), ("Short", "Short")):
            btn = QPushButton(label)
            self._style_filter_btn(btn)
            btn.setChecked(value == "Tümü")
            btn.clicked.connect(lambda _checked=False, v=value: self._set_side_filter(v))
            self._side_buttons[value] = btn
            filter_row.addWidget(btn)

        filter_row.addSpacing(14)
        strateji_label = QLabel("Strateji:")
        strateji_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        filter_row.addWidget(strateji_label)
        self._strategy_combo = self._make_combo(["Tümü"])
        self._strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        filter_row.addWidget(self._strategy_combo)

        filter_row.addStretch()
        layout.addLayout(filter_row)

        # TA/kovalama/HA-hizalanma bilgisi (24 Tem 2026, ta_kovalama_live) —
        # entry_features JSONB'de bu alanlar varsa seçili işlem için gösterilir,
        # yoksa (eski/başka strateji) boş kalır.
        self._ta_info_label = QLabel("")
        self._ta_info_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 2px 4px;"
        )
        layout.addWidget(self._ta_info_label)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self._table = QTableWidget(0, len(_TRADE_COLS))
        self._table.setHorizontalHeaderLabels(_TRADE_COLS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
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
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.ci.layout.setSpacing(2)
        date_axis = pg.DateAxisItem(orientation="bottom")

        def _title(text: str) -> str:
            return f'<span {_TITLE_STYLE}>{text}</span>'

        self._plot_pnl = self._plot_widget.addPlot(row=0, col=0, title=_title("Kâr/Zarar % (giriş yönlü)"))
        self._plot_widget.nextRow()
        self._plot_vpmv = self._plot_widget.addPlot(row=1, col=0, title=_title("VPMV Bileşenleri"))
        self._plot_widget.nextRow()
        self._plot_cvd = self._plot_widget.addPlot(row=2, col=0, title=_title("CVD Slope"))
        self._plot_widget.nextRow()
        self._plot_vp = self._plot_widget.addPlot(
            row=3, col=0, title=_title("VP Score"), axisItems={"bottom": date_axis}
        )

        self._plot_pnl.setLabel("left", "%", color=_AXIS_COLOR)
        self._plot_vpmv.setLabel("left", "% (girişe göre)", color=_AXIS_COLOR)
        self._plot_cvd.setLabel("left", "eğim", color=_AXIS_COLOR)
        self._plot_vp.setLabel("left", "-100..100", color=_AXIS_COLOR)
        self._plot_vp.setLabel("bottom", "Zaman", color=_AXIS_COLOR)

        for p in (self._plot_pnl, self._plot_vpmv, self._plot_cvd, self._plot_vp):
            p.showGrid(x=True, y=True, alpha=_GRID_ALPHA)
            p.getAxis("left").setWidth(46)
            legend = p.addLegend(offset=(5, 5))
            legend.setBrush(pg.mkBrush(10, 10, 15, 200))
            legend.setLabelTextColor((190, 190, 200))
        for p in (self._plot_pnl, self._plot_vpmv, self._plot_cvd):
            p.getAxis("bottom").setStyle(showValues=False)
        self._plot_vpmv.setXLink(self._plot_pnl)
        self._plot_cvd.setXLink(self._plot_pnl)
        self._plot_vp.setXLink(self._plot_pnl)

        self._plot_pnl.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_ZERO_PEN))
        self._plot_cvd.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_ZERO_PEN))

        # Senkronize imleç (crosshair) — 4 grafikte birlikte hareket eden dikey
        # çizgi + üstte an itibariyle değerleri gösteren etiket.
        self._crosshair_label = pg.LabelItem(justify="left")
        self._plot_widget.addItem(self._crosshair_label, row=0, col=1, rowspan=4)
        self._plot_widget.ci.layout.setColumnFixedWidth(1, 150)
        self._vlines = []
        for p in (self._plot_pnl, self._plot_vpmv, self._plot_cvd, self._plot_vp):
            vline = pg.InfiniteLine(angle=90, movable=False, pen=_CROSSHAIR_PEN)
            vline.setVisible(False)
            p.addItem(vline, ignoreBounds=True)
            self._vlines.append(vline)
        self._snapshot_cache: list[dict] = []
        self._vpmv_baseline: dict[str, Optional[float]] = {}
        self._proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )

        splitter.addWidget(self._plot_widget)
        splitter.setSizes([180, 520])
        layout.addWidget(splitter)

    def refresh(self) -> None:
        self._status.setText("Yükleniyor…")
        self._list_worker = _TradeListWorker(self._db_config, parent=self)
        self._list_worker.trades_loaded.connect(self._on_trades_loaded)
        self._list_worker.start()

    @pyqtSlot(object)
    def _on_trades_loaded(self, trades: list) -> None:
        self._trades = trades
        self._status.setText(f"{len(trades)} işlem")
        self._refresh_strategy_options()
        self._apply_search_filter()

    def _refresh_strategy_options(self) -> None:
        strategies = sorted({t["strategy"] for t in self._trades if t.get("strategy")})
        current = self._strategy_combo.currentText()
        self._strategy_combo.blockSignals(True)
        self._strategy_combo.clear()
        self._strategy_combo.addItems(["Tümü", *strategies])
        idx = self._strategy_combo.findText(current)
        self._strategy_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._strategy_combo.blockSignals(False)
        self._strategy_filter = self._strategy_combo.currentText()

    def _on_strategy_changed(self, value: str) -> None:
        self._strategy_filter = value
        self._apply_search_filter()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip().upper()
        self._apply_search_filter()

    def _set_status_filter(self, value: str) -> None:
        self._status_filter = value
        for v, btn in self._status_buttons.items():
            btn.setChecked(v == value)
        self._apply_search_filter()

    def _set_side_filter(self, value: str) -> None:
        self._side_filter = value
        for v, btn in self._side_buttons.items():
            btn.setChecked(v == value)
        self._apply_search_filter()

    def _apply_search_filter(self) -> None:
        trades = self._trades
        if self._search_text:
            trades = [t for t in trades if self._search_text in t["symbol"].upper()]
        if self._strategy_filter != "Tümü":
            trades = [t for t in trades if t.get("strategy") == self._strategy_filter]
        if self._status_filter != "Tümü":
            trades = [t for t in trades if t.get("status") == self._status_filter]
        if self._side_filter != "Tümü":
            trades = [t for t in trades if t.get("signal_type") == self._side_filter]
        self._filtered_trades = trades
        self._render_table()

    def _render_table(self) -> None:
        selected_id = self._selected_trade_id()

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._filtered_trades))

        for row_idx, t in enumerate(self._filtered_trades):
            sym_item = QTableWidgetItem(t["symbol"])
            sym_item.setForeground(_C_WHITE)
            sym_item.setData(Qt.ItemDataRole.UserRole, t["id"])
            self._table.setItem(row_idx, _COL_SYMBOL, sym_item)

            strat_item = QTableWidgetItem(t["strategy"])
            strat_item.setForeground(_C_MUTED)
            self._table.setItem(row_idx, _COL_STRATEGY, strat_item)

            side = t.get("signal_type", "")
            side_color = _C_GREEN if side == "Long" else _C_RED if side == "Short" else _C_MUTED
            side_item = QTableWidgetItem(side)
            side_item.setForeground(side_color)
            side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, _COL_SIDE, side_item)

            status = t.get("status", "")
            status_item = QTableWidgetItem(
                "Açık" if status == "open" else "Kapalı" if status == "closed" else status
            )
            status_item.setForeground(_C_GREEN if status == "open" else _C_MUTED)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, _COL_STATUS, status_item)

            opened = t.get("opened_at")
            opened_item = QTableWidgetItem(opened.strftime("%d/%m %H:%M") if opened else "—")
            opened_item.setForeground(_C_MUTED)
            opened_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, _COL_OPENED, opened_item)

            pnl = t.get("pnl_pct")
            if pnl is None:
                pnl_item = QTableWidgetItem("—")
                pnl_item.setForeground(_C_MUTED)
                pnl_item.setData(Qt.ItemDataRole.UserRole, 0)
            else:
                sign = "+" if pnl > 0 else ""
                pnl_item = _NumericItem(f"{sign}{pnl:.2f}")
                pnl_item.setData(Qt.ItemDataRole.UserRole, pnl)
                pnl_item.setForeground(_C_GREEN if pnl > 0 else _C_RED)
            pnl_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, _COL_PNL, pnl_item)

        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

        if selected_id is not None:
            self._reselect_trade(selected_id)

    def _selected_trade_id(self) -> Optional[int]:
        row = self._table.currentRow()
        if row < 0:
            return None
        sym_item = self._table.item(row, _COL_SYMBOL)
        if sym_item is None:
            return None
        trade_id = sym_item.data(Qt.ItemDataRole.UserRole)
        return int(trade_id) if trade_id is not None else None

    def _reselect_trade(self, trade_id: int) -> None:
        """Refresh sonrası sıralama/satır kayması nedeniyle seçili işlemin
        grafiği sessizce yanlış işlemi göstermeye devam etmesin diye —
        satırı bulup yeniden seçer ve grafiği tazeler."""
        for row in range(self._table.rowCount()):
            sym_item = self._table.item(row, _COL_SYMBOL)
            if sym_item is not None and sym_item.data(Qt.ItemDataRole.UserRole) == trade_id:
                self._table.selectRow(row)
                self._update_ta_info_label(trade_id)
                self._load_snapshots(trade_id)
                return

    def _on_row_selected(self) -> None:
        trade_id = self._selected_trade_id()
        if trade_id is None:
            return
        self._update_ta_info_label(trade_id)
        self._load_snapshots(trade_id)

    def _update_ta_info_label(self, trade_id: int) -> None:
        trade = next((t for t in self._filtered_trades if t["id"] == trade_id), None)
        features = (trade or {}).get("entry_features") or {}
        if "ta_pct_1h" not in features:
            self._ta_info_label.setText("")
            return
        ha = features.get("ha_aligned")
        ha_text = "✓ hizalı" if ha is True else ("✗ hizasız" if ha is False else "—")
        self._ta_info_label.setText(
            f"TA percentile 1h/4h: {features.get('ta_pct_1h', '—')} / {features.get('ta_pct_4h', '—')}"
            f"   |   slope 1h/4h: {features.get('ta_slope_1h', '—')} / {features.get('ta_slope_4h', '—')}"
            f"   |   HA: {ha_text}"
            f"   |   kalabalık (4sa): {features.get('concurrent_count', '—')}"
            f"   |   gösterge: {features.get('indicators', '—')}"
        )

    def _load_snapshots(self, trade_id: int) -> None:
        self._snap_worker = _SnapshotWorker(self._db_config, int(trade_id), parent=self)
        self._snap_worker.snapshots_loaded.connect(self._on_snapshots_loaded)
        self._snap_worker.start()

    @pyqtSlot(object)
    def _on_snapshots_loaded(self, snaps: list) -> None:
        self._render_charts(snaps)

    def _render_charts(self, snaps: list) -> None:
        # clear() TÜM item'ları siler (sıfır çizgisi/crosshair dahil) — o yüzden
        # veri eğrilerinden sonra kalıcı işaretleyicileri (zero-line, vline)
        # yeniden ekliyoruz.
        for p in (self._plot_pnl, self._plot_vpmv, self._plot_cvd, self._plot_vp):
            p.clear()
        self._plot_pnl.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_ZERO_PEN))
        self._plot_cvd.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_ZERO_PEN))
        self._vlines = []
        for p in (self._plot_pnl, self._plot_vpmv, self._plot_cvd, self._plot_vp):
            vline = pg.InfiniteLine(angle=90, movable=False, pen=_CROSSHAIR_PEN)
            vline.setVisible(False)
            p.addItem(vline, ignoreBounds=True)
            self._vlines.append(vline)
        self._crosshair_label.setText("")
        self._snapshot_cache = snaps
        if not snaps:
            return

        x = [s["taken_at"].timestamp() for s in snaps]

        y_pnl = [s["price_since_entry_pct"] or 0 for s in snaps]
        self._plot_pnl.plot(
            x,
            y_pnl,
            pen=pg.mkPen(_C_GREEN, width=2),
            name="PnL%",
            fillLevel=0,
            brush=pg.mkBrush(76, 175, 80, 35),
        )

        smc_x, smc_y, smc_brush = [], [], []
        for xi, s in zip(x, snaps):
            struct = s.get("smc_market_structure") or "-"
            if struct == "-":
                continue
            smc_x.append(xi)
            smc_y.append(s["price_since_entry_pct"] or 0)
            smc_brush.append((100, 220, 100) if "Uyum" in struct else (220, 100, 100))
        if smc_x:
            self._plot_pnl.plot(
                smc_x, smc_y, pen=None, symbol="o", symbolSize=6, symbolBrush=smc_brush
            )

        self._vpmv_baseline = {}
        for key, color, name in _VPMV_SERIES:
            base = next((s[key] for s in snaps if s[key] is not None), None)
            self._vpmv_baseline[key] = base
            if base:
                y = [
                    ((s[key] - base) / base * 100.0) if s[key] is not None else None
                    for s in snaps
                ]
            else:
                y = [None for _ in snaps]
            self._plot_vpmv.plot(x, y, pen=pg.mkPen(color, width=2), name=name)

        y_cvd = [s["cvd_slope"] or 0 for s in snaps]
        self._plot_cvd.plot(x, y_cvd, pen=pg.mkPen((80, 220, 220), width=2), name="CVD")

        y_vp = [s["vp_score"] or 0 for s in snaps]
        y_vpr = [s["vp_score_real"] or 0 for s in snaps]
        self._plot_vp.plot(x, y_vp, pen=pg.mkPen((240, 180, 80), width=2), name="Proxy")
        self._plot_vp.plot(x, y_vpr, pen=pg.mkPen((100, 220, 100), width=2), name="Gerçek")

    def _on_mouse_moved(self, evt) -> None:
        if not self._snapshot_cache:
            return
        pos = evt[0]
        plots = (self._plot_pnl, self._plot_vpmv, self._plot_cvd, self._plot_vp)
        if not any(p.sceneBoundingRect().contains(pos) for p in plots):
            for vline in self._vlines:
                vline.setVisible(False)
            return

        mouse_point = self._plot_pnl.vb.mapSceneToView(pos)
        target_x = mouse_point.x()
        xs = [s["taken_at"].timestamp() for s in self._snapshot_cache]
        idx = bisect.bisect_left(xs, target_x)
        idx = max(0, min(idx, len(xs) - 1))
        if idx > 0 and abs(xs[idx - 1] - target_x) < abs(xs[idx] - target_x):
            idx -= 1
        snap = self._snapshot_cache[idx]

        for vline in self._vlines:
            vline.setPos(xs[idx])
            vline.setVisible(True)

        def _row(label: str, value, fmt: str = "{:.2f}", color: str = "#aaaaaa") -> str:
            if value is None:
                return f"<div style='color:#555566;font-size:9pt'>{label}: —</div>"
            return f"<div style='color:{color};font-size:9pt'>{label}: {fmt.format(value)}</div>"

        def _pct_vs_base(key: str):
            base = self._vpmv_baseline.get(key)
            val = snap.get(key)
            if base is None or val is None:
                return None
            return (val - base) / base * 100.0

        pnl = snap.get("price_since_entry_pct")
        pnl_color = COLORS["green"] if (pnl or 0) >= 0 else COLORS["red"]
        struct = snap.get("smc_market_structure")
        html = [
            f"<div style='color:#c8c8d2;font-size:10pt;font-weight:600'>"
            f"{snap['taken_at'].strftime('%d/%m %H:%M:%S')}</div>",
            _row("PnL", pnl, "{:+.3f}%", pnl_color),
            _row("V", _pct_vs_base("vol_score"), "{:+.1f}%", "#64dc64"),
            _row("M", _pct_vs_base("mom_score"), "{:+.1f}%", "#64a0f0"),
            _row("Vlt", _pct_vs_base("volat_score"), "{:+.1f}%", "#f0b450"),
            _row("P", _pct_vs_base("price_score"), "{:+.1f}%", "#dc6464"),
            _row("Combined", _pct_vs_base("vpmv_combined"), "{:+.1f}%", "#f0f050"),
            _row("CVD", snap.get("cvd_slope"), "{:.4f}", "#50dcdc"),
            _row("VP proxy", snap.get("vp_score"), color="#f0b450"),
            _row("VP gerçek", snap.get("vp_score_real"), color="#64dc64"),
            f"<div style='color:#aaaaaa;font-size:9pt'>SMC: {struct}</div>" if struct else "",
        ]
        self._crosshair_label.setText("".join(html))
