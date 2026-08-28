"""DivergenceModel — sinyal coinlerin fiyat Z-score ayrışma tablosu için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 6 —
divergence_panel.py'nin pos/neg ikiz QTableWidget'larının yerini alır).

Bu panelin bayatlık tespiti diğerlerinden FARKLI — worker-heartbeat değil,
SATIR BAZLI içerik zaman damgası (`divergence_live`'ın son bar zamanı, TF'ye
göre dinamik eşik). Panel her populate'te `is_stale`'i hesaplayıp satıra
gömüyor (rank/rank_delta/time_str gibi diğer türetilmiş alanlarla aynı desen)
— StalePanelMixin burada UYGULANMAZ, o worker'ın kendisinin sessiz kaldığı
global durumlar için (bkz. desktop/widgets/staleness.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor, QFont

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt, sign_color
from desktop.theme import COLORS

COLUMNS = ["Sembol", "Z-score", "VPMV", "Rank", "Zaman"]
COL_SYMBOL = 0
COL_ZSCORE = 1
COL_VPMV = 2
COL_RANK = 3
COL_TIME = 4

_C_VPMV_HIGH = QColor(80, 200, 80)
_C_VPMV_MID = QColor(200, 180, 60)
_C_VPMV_LOW = QColor(200, 80, 80)

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])

_BG_GREEN_STRONG = QColor(0, 120, 40, 150)
_BG_GREEN_SOFT = QColor(0, 80, 20, 80)
_BG_RED_STRONG = QColor(180, 20, 20, 150)
_BG_RED_SOFT = QColor(120, 10, 10, 80)
_BG_STALE = QColor(120, 70, 0, 140)

_BOLD = QFont("Courier New", 11, QFont.Weight.Bold)
_MONO = QFont("Courier New", 11)


@dataclass
class DivergenceRow:
    symbol: str
    z: float
    vpmv: float
    rank: Optional[int]
    time_str: str
    rank_delta: int
    is_stale: bool


def _build_row(item: dict) -> DivergenceRow:
    return DivergenceRow(
        symbol=item["symbol"],
        z=item["z"],
        vpmv=item.get("vpmv", 0.0),
        rank=item.get("rank"),
        time_str=item.get("time_str", "—"),
        rank_delta=item.get("rank_delta", 0),
        is_stale=item.get("is_stale", False),
    )


def _update_row(row: DivergenceRow, item: dict) -> None:
    new = _build_row(item)
    row.__dict__.update(new.__dict__)


class DivergenceModel(IdKeyedTableModel):
    COLUMNS = COLUMNS
    build_row = staticmethod(_build_row)
    update_row = staticmethod(_update_row)

    def __init__(self, positive: bool, parent=None):
        super().__init__(parent)
        self._positive = positive
        self._z_color = _C_GREEN if positive else _C_RED

    def _id_of_item(self, item: dict) -> str:
        return item["symbol"]

    def _id_of_row(self, row: DivergenceRow) -> str:
        return row.symbol

    def _display(self, row: DivergenceRow, col: int) -> str:
        if col == COL_SYMBOL:
            if row.rank_delta > 0:
                text = f"{row.symbol} ↑{row.rank_delta}"
            elif row.rank_delta < 0:
                text = f"{row.symbol} ↓{abs(row.rank_delta)}"
            else:
                text = row.symbol
            return f"{text} ⚠" if row.is_stale else text
        if col == COL_ZSCORE:
            return f"{row.z:+.2f}"
        if col == COL_VPMV:
            return f"{row.vpmv:.0f}"
        if col == COL_RANK:
            return str(row.rank) if row.rank is not None else "—"
        if col == COL_TIME:
            return row.time_str
        return ""

    def _foreground(self, row: DivergenceRow, col: int):
        if col == COL_SYMBOL:
            if row.rank_delta != 0:
                return sign_color(row.rank_delta)
            return self._z_color
        if col == COL_ZSCORE:
            return self._z_color
        if col == COL_VPMV:
            if row.vpmv >= 60:
                return _C_VPMV_HIGH
            if row.vpmv >= 45:
                return _C_VPMV_MID
            return _C_VPMV_LOW
        if col in (COL_RANK, COL_TIME):
            return _C_MUTED
        return None

    def _background(self, row: DivergenceRow, col: int):
        if row.is_stale:
            return _BG_STALE
        if col == COL_ZSCORE:
            abs_z = abs(row.z)
            if abs_z >= 2.0:
                return _BG_GREEN_STRONG if self._positive else _BG_RED_STRONG
            if abs_z >= 1.0:
                return _BG_GREEN_SOFT if self._positive else _BG_RED_SOFT
        return None

    def _font(self, row: DivergenceRow, col: int):
        return _BOLD if col == COL_SYMBOL else _MONO

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.FontRole
            and index.isValid()
            and 0 <= index.row() < len(self._rows)
        ):
            return self._font(self._rows[index.row()], index.column())
        return super().data(index, role)


class DivergenceProxyModel(QSortFilterProxyModel):
    _NUMERIC_ATTR = {COL_ZSCORE: "z", COL_VPMV: "vpmv", COL_RANK: "rank"}

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
        if self._search and self._search not in row.symbol:
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        attr = self._NUMERIC_ATTR.get(left.column())
        if attr is not None:
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)
