"""
RankingPanel — tüm coinleri VPMV güç skoruna göre sıralayan panel.

Kolonlar: Rank | Sembol | 5m | 15m | 1h | Birleşik | TF Uyum | VS BTC
"""

from typing import Optional

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
from desktop.workers.ranking_worker import RankingWorker

_COL_RANK      = 0
_COL_SYMBOL    = 1
_COL_5M        = 2
_COL_15M       = 3
_COL_1H        = 4
_COL_4H        = 5
_COL_COMBINED  = 6
_COL_ZCONF     = 7
_COL_RSCORE    = 8
_COL_ALIGN     = 9
_COL_VSBTC     = 10
_HEADERS = ["#", "Sembol", "5m", "15m", "1h", "4h", "Birleşik", "Z-Conf", "R-Score", "TF Uyum", "VS BTC"]

_C_GREEN  = QColor(COLORS["green"])
_C_RED    = QColor(COLORS["red"])
_C_YELLOW = QColor(COLORS["yellow"])
_C_MUTED  = QColor(COLORS["text_muted"])
_C_WHITE  = QColor(COLORS["text_primary"])

_BG_STRONG_BULL = QColor(0, 120, 40, 120)
_BG_SOFT_BULL   = QColor(0, 80, 20, 60)
_BG_STRONG_BEAR = QColor(180, 20, 20, 120)
_BG_SOFT_BEAR   = QColor(120, 10, 10, 60)

_PINE_20 = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "LTCUSDT", "DOTUSDT", "SOLUSDT", "AVAXUSDT", "TRXUSDT",
    "UNIUSDT", "LINKUSDT", "VETUSDT", "XLMUSDT", "NEARUSDT",
    "WIFUSDT", "ZRXUSDT", "ATOMUSDT", "CAKEUSDT", "KSMUSDT",
}


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


class RankingPanel(QWidget):
    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._worker = RankingWorker(redis_url, parent=self)
        self._pine_filter = False
        self._search_text = ""
        self._last_result: list = []
        self._prev_ranks: dict[str, int] = {}
        self._setup_ui()
        self._connect_worker()
        self._worker.start()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Üst bar
        top = QHBoxLayout()
        title = QLabel("Güç Sıralaması")
        title.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")

        self._status = QLabel("Yükleniyor…")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self._pine_btn = QPushButton("Pine 20")
        self._pine_btn.setCheckable(True)
        self._pine_btn.setFixedHeight(24)
        self._pine_btn.setFixedWidth(60)
        self._pine_btn.setStyleSheet(self._filter_btn_style(False))
        self._pine_btn.clicked.connect(self._on_pine_toggled)

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
        top.addWidget(self._pine_btn)
        top.addWidget(self._status)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        # Tablo
        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
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
        # ResizeToContents sürekli modda HER setItem() çağrısında tüm sütunu yeniden
        # ölçüyor (O(satır) maliyet × N setItem = O(satır²)) — 550 sembolle bu, ana
        # thread'i kilitleyip panel kasmasına yol açıyordu. Interactive + _render
        # sonunda tek seferlik resizeColumnsToContents() aynı görünümü verir, sürekli
        # yeniden ölçüm olmadan.
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self._table)

    @staticmethod
    def _filter_btn_style(active: bool) -> str:
        if active:
            return (f"QPushButton {{ background: {COLORS['accent']}; color: #fff; "
                    f"border: none; border-radius: 3px; font-size: 10px; }}")
        return (f"QPushButton {{ background: {COLORS['bg_tertiary']}; "
                f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
                f"border-radius: 3px; font-size: 10px; }}")

    def _on_pine_toggled(self, checked: bool) -> None:
        self._pine_filter = checked
        self._pine_btn.setStyleSheet(self._filter_btn_style(checked))
        self._render(self._last_result)

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip().upper()
        self._apply_search_filter()

    def _apply_search_filter(self) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_SYMBOL)
            if item is None:
                continue
            symbol = item.text().split()[0]  # "BTCUSDT ↑3" → "BTCUSDT"
            hidden = bool(self._search_text) and self._search_text not in symbol
            self._table.setRowHidden(row, hidden)

    def _connect_worker(self) -> None:
        self._worker.ranking_updated.connect(self._on_updated)
        self._worker.status_updated.connect(self._on_status)

    # ------------------------------------------------------------------
    @pyqtSlot(object)
    def _on_updated(self, result: list) -> None:
        self._last_result = result
        self._render(result)

    def _render(self, result: list) -> None:
        if self._pine_filter:
            result = [r for r in result if r["symbol"] in _PINE_20]

        # Sıra değişimlerini hesapla
        rank_deltas: dict[str, int] = {}
        for row_data in result:
            sym = row_data["symbol"]
            if sym in self._prev_ranks:
                rank_deltas[sym] = self._prev_ranks[sym] - row_data["rank"]
        self._prev_ranks = {r["symbol"]: r["rank"] for r in result}

        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for row_data in result:
            row = self._table.rowCount()
            self._table.insertRow(row)

            rank_score = row_data.get("rank_score", 50)
            direction  = row_data.get("direction", "long")
            combined   = row_data.get("combined", 50)

            # Satır arka plan rengi
            if rank_score >= 80:
                bg = _BG_STRONG_BULL if direction == "long" else _BG_STRONG_BEAR
            elif rank_score >= 60:
                bg = _BG_SOFT_BULL if direction == "long" else _BG_SOFT_BEAR
            else:
                bg = None

            # Rank
            self._set_num(row, _COL_RANK, row_data["rank"], bg)

            # Sembol + sıra değişimi
            sym = row_data["symbol"]
            delta = rank_deltas.get(sym, 0)
            if delta > 0:
                sym_text = f"{sym} ↑{delta}"
                sym_color = _C_GREEN
            elif delta < 0:
                sym_text = f"{sym} ↓{abs(delta)}"
                sym_color = _C_RED
            else:
                sym_text = sym
                sym_color = _C_WHITE
            sym_item = QTableWidgetItem(sym_text)
            sym_item.setForeground(sym_color)
            if bg:
                sym_item.setBackground(bg)
            self._table.setItem(row, _COL_SYMBOL, sym_item)

            # TF skorları
            for col, key in (
                (_COL_5M,  "score_5m"),
                (_COL_15M, "score_15m"),
                (_COL_1H,  "score_1h"),
                (_COL_4H,  "score_4h"),
            ):
                val = row_data.get(key)
                self._set_score(row, col, val, bg)

            # Birleşik
            self._set_score(row, _COL_COMBINED, combined, bg, bold=True)

            # Z-Conf
            self._set_zconf(row, row_data.get("z_confluence"), bg)

            # R-Score
            self._set_rscore(row, row_data.get("r_score"), bg)

            # TF Uyum
            align_count = row_data.get("alignment_count", 0)
            tf_count    = row_data.get("tf_count", 0)
            aligned     = row_data.get("aligned", False)
            align_text  = f"{'✓' if aligned else '~'} {align_count}/{tf_count}"
            align_item  = QTableWidgetItem(align_text)
            align_item.setForeground(_C_GREEN if aligned else _C_YELLOW)
            if bg:
                align_item.setBackground(bg)
            self._table.setItem(row, _COL_ALIGN, align_item)

            # VS BTC
            vs_btc = row_data.get("vs_btc")
            self._set_vs_btc(row, vs_btc, bg)

        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        self._apply_search_filter()

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status.setText(msg)

    # ------------------------------------------------------------------
    def _set_num(self, row: int, col: int, val: Optional[float], bg) -> None:
        item = _NumericItem(str(int(val)) if val is not None else "—")
        item.setData(Qt.ItemDataRole.UserRole, val if val is not None else 0)
        item.setForeground(_C_MUTED)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if bg:
            item.setBackground(bg)
        self._table.setItem(row, col, item)

    def _set_score(
        self, row: int, col: int, val: Optional[float], bg, bold: bool = False
    ) -> None:
        text = f"{val:.0f}" if val is not None else "—"
        item = _NumericItem(text)
        item.setData(Qt.ItemDataRole.UserRole, val if val is not None else 0)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if val is not None:
            item.setForeground(_C_GREEN if val >= 55 else _C_RED if val <= 45 else _C_MUTED)
        else:
            item.setForeground(_C_MUTED)
        if bold:
            f = item.font()
            f.setBold(True)
            item.setFont(f)
        if bg:
            item.setBackground(bg)
        self._table.setItem(row, col, item)

    def _set_rscore(self, row: int, val: Optional[float], bg) -> None:
        if val is None:
            item = QTableWidgetItem("—")
            item.setForeground(_C_MUTED)
        else:
            sign = "+" if val > 0 else ""
            item = _NumericItem(f"{sign}{val:.3f}")
            item.setData(Qt.ItemDataRole.UserRole, val)
            item.setForeground(_C_GREEN if val > 0 else _C_RED)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if bg:
            item.setBackground(bg)
        self._table.setItem(row, _COL_RSCORE, item)

    def _set_zconf(self, row: int, val: Optional[float], bg) -> None:
        if val is None:
            item = QTableWidgetItem("—")
            item.setForeground(_C_MUTED)
        else:
            sign = "+" if val > 0 else ""
            item = _NumericItem(f"{sign}{val:.2f}")
            item.setData(Qt.ItemDataRole.UserRole, val)
            if val >= 1.5:
                item.setForeground(_C_GREEN)
            elif val >= 0.5:
                item.setForeground(QColor(100, 200, 100))
            elif val <= -1.5:
                item.setForeground(_C_RED)
            elif val <= -0.5:
                item.setForeground(QColor(200, 100, 100))
            else:
                item.setForeground(_C_MUTED)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if bg:
            item.setBackground(bg)
        self._table.setItem(row, _COL_ZCONF, item)

    def _set_vs_btc(self, row: int, val: Optional[float], bg) -> None:
        if val is None:
            item = QTableWidgetItem("—")
            item.setForeground(_C_MUTED)
        else:
            sign = "+" if val > 0 else ""
            item = _NumericItem(f"{sign}{val:.1f}")
            item.setData(Qt.ItemDataRole.UserRole, val)
            item.setForeground(_C_GREEN if val > 0 else _C_RED if val < 0 else _C_MUTED)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if bg:
            item.setBackground(bg)
        self._table.setItem(row, _COL_VSBTC, item)

    def closeEvent(self, event) -> None:
        self._worker.stop()
        super().closeEvent(event)
