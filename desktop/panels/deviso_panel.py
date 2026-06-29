"""
DevisoPanel — aktif sinyalleri devisso_score'a göre sıralayan tablo.

Kolonlar: # | Sembol | TF | Yön | Score | Δ | Ratio | Zaman
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QHeaderView,
    QLabel,
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


def _item(text: str, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    it.setTextAlignment(align)
    return it


class DevisoPanel(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._signals: Dict[int, Dict[str, Any]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

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
        self._table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QHeaderView::section { font-size: 11px; font-weight: bold; }
        """)

        bold = QFont()
        bold.setBold(True)
        self._table.horizontalHeader().setFont(bold)

        layout.addWidget(self._table)

    @pyqtSlot(list)
    def on_signals_loaded(self, rows: List[Dict[str, Any]]) -> None:
        self._signals = {r["id"]: r for r in rows if r.get("status") == "active"}
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
        rows = sorted(
            self._signals.values(),
            key=lambda r: (r.get("devisso_score") or -1),
            reverse=True,
        )
        self._populate(rows)
        with_score = sum(1 for r in rows if r.get("devisso_score") is not None)
        self._status.setText(f"{len(rows)} aktif sinyal | {with_score} devisso hesaplı")

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

            score_item = _item(f"{score:.1f}" if score is not None else "-")
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
                delta_item = _item(f"{prefix}{delta:.1f}")
                delta_item.setForeground(_C_GREEN if delta >= 0 else _C_RED)
            else:
                delta_item = _item("-")
                delta_item.setForeground(_C_MUTED)
            self._table.setItem(i, _COL_DELTA, delta_item)

            if ratio is not None:
                ratio_item = _item(f"{ratio:.2f}x")
                if ratio > 1.0:
                    ratio_item.setForeground(_C_GREEN)
                elif ratio < 1.0:
                    ratio_item.setForeground(_C_RED)
                else:
                    ratio_item.setForeground(_C_WHITE)
            else:
                ratio_item = _item("-")
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
