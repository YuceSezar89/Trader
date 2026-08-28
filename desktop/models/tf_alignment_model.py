"""TFAlignmentModel — TF Hizalanma + Erken Ayrışma adayları tablosu için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 4 —
tf_alignment_panel.py'nin eski QTableWidget'ının yerini alır).

Long ve Short için AYRI model+proxy örnekleri kullanılır (backend zaten veriyi
yöne göre ayırıp panele öyle veriyor, tf_alignment_panel._render bkz.) — tek
paylaşımlı model üzerinde proxy ile bölme (Faz 5'in vpmv_divergence_panel için
çözeceği "pos/neg ikiz tablo" deseni) burada gerekmiyor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt
from desktop.theme import COLORS

COLUMNS = ["Sembol", "TF", "Gösterge", "Açılış", "Erken %", ""]
COL_SYMBOL = 0
COL_TF = 1
COL_INDICATOR = 2
COL_OPEN_PRICE = 3
COL_EARLY_PCT = 4
COL_ACTION = 5

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])


@dataclass
class TFAlignmentRow:
    symbol: str
    interval: str
    indicators: str
    open_price: Optional[float]
    early_pct: Optional[float]
    raw: dict  # ManualTradeDialog'u önceden doldurmak için ham veri saklanır


def _build_row(item: dict) -> TFAlignmentRow:
    return TFAlignmentRow(
        symbol=item.get("symbol", ""),
        interval=item.get("interval", ""),
        indicators=item.get("indicators", ""),
        open_price=item.get("open_price"),
        early_pct=item.get("early_pct"),
        raw=item,
    )


def _update_row(row: TFAlignmentRow, item: dict) -> None:
    new = _build_row(item)
    row.__dict__.update(new.__dict__)


class TFAlignmentModel(IdKeyedTableModel):
    COLUMNS = COLUMNS
    build_row = staticmethod(_build_row)
    update_row = staticmethod(_update_row)

    def _id_of_item(self, item: dict) -> str:
        return item.get("symbol", "")

    def _id_of_row(self, row: TFAlignmentRow) -> str:
        return row.symbol

    def _display(self, row: TFAlignmentRow, col: int) -> str:
        if col == COL_SYMBOL:
            return row.symbol
        if col == COL_TF:
            return row.interval
        if col == COL_INDICATOR:
            return row.indicators
        if col == COL_OPEN_PRICE:
            return f"{row.open_price:.6g}" if row.open_price is not None else "—"
        if col == COL_EARLY_PCT:
            if row.early_pct is None:
                return "—"
            sign = "+" if row.early_pct > 0 else ""
            return f"{sign}{row.early_pct:.3f}"
        return ""

    def _foreground(self, row: TFAlignmentRow, col: int):
        if col == COL_SYMBOL:
            return _C_WHITE
        if col in (COL_TF, COL_INDICATOR, COL_OPEN_PRICE):
            return _C_MUTED
        if col == COL_EARLY_PCT:
            if row.early_pct is None:
                return _C_MUTED
            return _C_GREEN if row.early_pct > 0 else _C_RED
        return None

    def _text_alignment(self, col: int) -> int:
        if col == COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)


class TFAlignmentProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ""

    def set_search(self, text: str) -> None:
        self._search = text.strip().upper()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._search and self._search not in row.symbol.upper():
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        col = left.column()
        if col in (COL_OPEN_PRICE, COL_EARLY_PCT):
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            attr = "open_price" if col == COL_OPEN_PRICE else "early_pct"
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)
