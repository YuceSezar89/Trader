"""RankModel — Devisso Döngüsü / Totalamount Rank-1 tablosu için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 1 — rank_panel.py'nin
eski QTableWidget'ının yerini alır, bkz. desktop/models/base_table_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor, QFont

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt, sign_color
from desktop.theme import COLORS

COLUMNS = ["Sembol", "Totalamount", "Rank"]
COL_SYMBOL = 0
COL_VALUE = 1
COL_RANK = 2

_C_MUTED = QColor(COLORS["text_muted"])
_BG_TOP = QColor(0, 120, 40, 90)
_MONO = QFont("Courier New", 11)
_MONO_BOLD = QFont("Courier New", 11, QFont.Weight.Bold)


@dataclass
class RankRow:
    symbol: str
    value: float
    rank: int
    is_top: bool  # bu turun anlık sıralamasında Totalamount'u en yüksek satır


class RankModel(IdKeyedTableModel):
    COLUMNS = COLUMNS

    def _id_of_item(self, item: dict) -> str:
        return item["symbol"]

    def _id_of_row(self, row: RankRow) -> str:
        return row.symbol

    @staticmethod
    def build_row(item: dict) -> RankRow:
        return RankRow(
            symbol=item["symbol"], value=item["value"], rank=item["rank"], is_top=item["is_top"]
        )

    @staticmethod
    def update_row(row: RankRow, item: dict) -> None:
        row.value = item["value"]
        row.rank = item["rank"]
        row.is_top = item["is_top"]

    def _display(self, row: RankRow, col: int) -> str:
        if col == COL_SYMBOL:
            return row.symbol
        if col == COL_VALUE:
            return f"{row.value:+.2f}"
        if col == COL_RANK:
            return str(row.rank)
        return ""

    def _foreground(self, row: RankRow, col: int):
        if col == COL_SYMBOL:
            return sign_color(row.value) if row.is_top else _C_MUTED
        if col == COL_VALUE:
            return sign_color(row.value)
        if col == COL_RANK:
            return _C_MUTED
        return None

    def _background(self, row: RankRow, col: int):
        if col == COL_VALUE and row.is_top:
            return _BG_TOP
        return None

    def _font(self, row: RankRow, col: int):
        if col == COL_SYMBOL and row.is_top:
            return _MONO_BOLD
        return _MONO

    def _text_alignment(self, col: int) -> int:
        if col == COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if (
            role == Qt.ItemDataRole.FontRole
            and index.isValid()
            and 0 <= index.row() < len(self._rows)
        ):
            return self._font(self._rows[index.row()], index.column())
        return super().data(index, role)


class RankProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(COL_SYMBOL)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        col = left.column()
        if col in (COL_VALUE, COL_RANK):
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            attr = "value" if col == COL_VALUE else "rank"
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)
