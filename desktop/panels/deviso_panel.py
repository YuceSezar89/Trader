"""
DevisoPanel — aktif sinyalleri devisso_score'a göre sıralayan tablo.

Kolonlar: # | Sembol | TF | Yön | Score | Δ | Ratio | Zaman
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
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

_COL_RANK   = 0
_COL_SYMBOL = 1
_COL_TF     = 2
_COL_DIR    = 3
_COL_SCORE  = 4
_COL_DELTA  = 5
_COL_RATIO  = 6
_COL_TIME   = 7
_HEADERS = ["#", "Sembol", "TF", "Yön", "Score", "Δ", "Ratio", "Zaman"]

_TOOLTIPS = [
    "Sıralama — Devisso score'a göre yüksekten düşüğe",
    "İşlem çifti",
    "Zaman dilimi (timeframe)",
    "Sinyal yönü — Long (alım) veya Short (satım)",
    "Devisso Score (0-100) — RSI Verimliliği\nFiyat değişimi (%) / RSI değişimi oranının yüzdelik sırası (son 100 bar).\nYüksek → Az RSI ile çok fiyat hareketi — trend verimli ve sağlıklı\nDüşük → Aynı hareket için RSI çok yoruldu — trend zorlanıyor",
    "Delta — Önceki aynı yöndeki sinyale göre score farkı.\nPozitif → piyasa daha verimli hareket etti\nNegatif → piyasa daha çok zorlandı",
    "Efficiency Ratio — Mevcut score / Önceki sinyal score (aynı yön).\n>1.0 → piyasa öncekine göre daha verimli\n<1.0 → piyasa öncekine göre daha zorlandı",
    "Sinyalin açılma zamanı",
]

_C_GREEN  = QColor(COLORS["green"])
_C_RED    = QColor(COLORS["red"])
_C_MUTED  = QColor(COLORS["text_muted"])
_C_WHITE  = QColor(COLORS["text_primary"])

_BG_HIGH  = QColor(0, 120, 40, 100)
_BG_LOW   = QColor(180, 20, 20, 80)

_TF_OPTIONS = ["Tüm TF", "1m", "5m", "15m", "1h", "4h", "1d"]


class _NumItem(QTableWidgetItem):
    """Sayısal sıralama için UserRole verisini kullanan item."""
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        v1 = self.data(Qt.ItemDataRole.UserRole)
        v2 = other.data(Qt.ItemDataRole.UserRole)
        if v1 is None and v2 is None:
            return False
        if v1 is None:
            return True
        if v2 is None:
            return False
        return float(v1) < float(v2)


def _item(text: str, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    it.setTextAlignment(align)
    return it


def _num_item(text: str, value: Optional[float]) -> _NumItem:
    it = _NumItem(str(text))
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    it.setData(Qt.ItemDataRole.UserRole, value if value is not None else -999.0)
    return it


class DevisoPanel(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._signals: Dict[int, Dict[str, Any]] = {}
        self._dir_filter: str = "Tümü"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Sembol ara...")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedHeight(26)
        self._search.setStyleSheet(
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_primary']};"
            " border: 1px solid #444; border-radius: 4px; padding: 0 6px; font-size: 12px;"
        )
        self._search.textChanged.connect(self._refresh)
        toolbar.addWidget(self._search, stretch=2)

        self._tf_combo = QComboBox()
        self._tf_combo.addItems(_TF_OPTIONS)
        self._tf_combo.setFixedHeight(26)
        self._tf_combo.setStyleSheet(
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_primary']};"
            " border: 1px solid #444; border-radius: 4px; font-size: 12px;"
        )
        self._tf_combo.currentTextChanged.connect(self._refresh)
        toolbar.addWidget(self._tf_combo)

        for label in ("Tümü", "Long", "Short"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(label == "Tümü")
            btn.setFixedHeight(26)
            btn.setStyleSheet(self._btn_style(label == "Tümü"))
            btn.clicked.connect(lambda checked, l=label, b=btn: self._on_dir_btn(l, b))
            toolbar.addWidget(btn)
            setattr(self, f"_btn_{label.lower()}", btn)

        layout.addLayout(toolbar)

        self._status = QLabel("Devisso Sıralama")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        layout.addWidget(self._status)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        for _col, _tip in enumerate(_TOOLTIPS):
            self._table.horizontalHeaderItem(_col).setToolTip(_tip)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_SYMBOL, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setSortingEnabled(True)
        self._table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QHeaderView::section { font-size: 11px; font-weight: bold; cursor: pointer; }
            QHeaderView::section:hover { background: #2a2a2a; }
        """)

        bold = QFont()
        bold.setBold(True)
        self._table.horizontalHeader().setFont(bold)

        layout.addWidget(self._table)

    def _btn_style(self, active: bool) -> str:
        if active:
            return (
                f"background: {COLORS.get('accent', '#1a6bcc')}; color: white;"
                " border-radius: 4px; font-size: 11px; padding: 0 8px;"
            )
        return (
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_muted']};"
            " border: 1px solid #444; border-radius: 4px; font-size: 11px; padding: 0 8px;"
        )

    def _on_dir_btn(self, label: str, _btn: QPushButton) -> None:
        self._dir_filter = label
        for name in ("tümü", "long", "short"):
            btn = getattr(self, f"_btn_{name}", None)
            if btn:
                active = (name == label.lower())
                btn.setChecked(active)
                btn.setStyleSheet(self._btn_style(active))
        self._refresh()

    def _filtered_rows(self) -> List[Dict[str, Any]]:
        search = self._search.text().strip().upper()
        tf = self._tf_combo.currentText()
        direction = self._dir_filter

        rows = [r for r in self._signals.values() if r.get("status") == "active"]

        if search:
            rows = [r for r in rows if search in r.get("symbol", "").upper()]
        if tf and tf != "Tüm TF":
            rows = [r for r in rows if r.get("interval") == tf]
        if direction == "Long":
            rows = [r for r in rows if r.get("signal_type") == "Long"]
        elif direction == "Short":
            rows = [r for r in rows if r.get("signal_type") == "Short"]

        return sorted(rows, key=lambda r: (r.get("devisso_score") or -1), reverse=True)

    @pyqtSlot(list)
    def on_signals_loaded(self, rows: List[Dict[str, Any]]) -> None:
        self._signals = {r["id"]: r for r in rows}
        self._refresh()

    @pyqtSlot(dict)
    def on_new_signal(self, row: Dict[str, Any]) -> None:
        if row.get("status") == "active":
            self._signals[row["id"]] = row
        else:
            self._signals.pop(row["id"], None)
        self._refresh()

    @pyqtSlot(list)
    def on_signals_closed(self, ids: List[int]) -> None:
        for sid in ids:
            self._signals.pop(sid, None)
        self._refresh()

    def _refresh(self) -> None:
        rows = self._filtered_rows()
        self._table.setSortingEnabled(False)
        self._populate(rows)
        self._table.setSortingEnabled(True)

        total = sum(1 for r in self._signals.values() if r.get("status") == "active")
        with_score = sum(1 for r in rows if r.get("devisso_score") is not None)
        shown = len(rows)
        if shown < total:
            self._status.setText(f"{shown}/{total} sinyal gösteriliyor | {with_score} devisso hesaplı")
        else:
            self._status.setText(f"{total} aktif sinyal | {with_score} devisso hesaplı")

    def _populate(self, rows: List[Dict[str, Any]]) -> None:
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            score = r.get("devisso_score")
            delta = r.get("devisso_delta")
            ratio = r.get("devisso_ratio")
            sig_type = r.get("signal_type", "")
            opened_at = r.get("opened_at")

            rank_item = _item(str(i + 1))
            rank_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_RANK, rank_item)

            sym_item = _item(r.get("symbol", ""), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            sym_item.setForeground(_C_WHITE)
            self._table.setItem(i, _COL_SYMBOL, sym_item)

            self._table.setItem(i, _COL_TF, _item(r.get("interval", "")))

            dir_item = _item(sig_type)
            dir_item.setForeground(_C_GREEN if sig_type == "Long" else _C_RED)
            self._table.setItem(i, _COL_DIR, dir_item)

            score_item = _num_item(f"{score:.1f}" if score is not None else "-", score)
            if score is not None:
                if score >= 65:
                    score_item.setForeground(_C_GREEN)
                    score_item.setBackground(_BG_HIGH)
                elif score <= 35:
                    score_item.setForeground(_C_RED)
                    score_item.setBackground(_BG_LOW)
                else:
                    score_item.setForeground(_C_WHITE)
            else:
                score_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_SCORE, score_item)

            if delta is not None:
                prefix = "+" if delta >= 0 else ""
                delta_item = _num_item(f"{prefix}{delta:.1f}", delta)
                delta_item.setForeground(_C_GREEN if delta >= 0 else _C_RED)
            else:
                delta_item = _num_item("-", None)
                delta_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_DELTA, delta_item)

            if ratio is not None:
                ratio_item = _num_item(f"{ratio:.2f}x", ratio)
                if ratio > 1.0:
                    ratio_item.setForeground(_C_GREEN)
                elif ratio < 1.0:
                    ratio_item.setForeground(_C_RED)
                else:
                    ratio_item.setForeground(_C_WHITE)
            else:
                ratio_item = _num_item("-", None)
                ratio_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_RATIO, ratio_item)

            if isinstance(opened_at, datetime):
                time_str = opened_at.strftime("%m-%d %H:%M")
            elif isinstance(opened_at, str):
                time_str = opened_at[5:16]
            else:
                time_str = "-"
            time_item = _item(time_str)
            time_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_TIME, time_item)

        self._table.resizeColumnToContents(_COL_RANK)
        self._table.resizeColumnToContents(_COL_TF)
        self._table.resizeColumnToContents(_COL_SCORE)
        self._table.resizeColumnToContents(_COL_DELTA)
        self._table.resizeColumnToContents(_COL_RATIO)
