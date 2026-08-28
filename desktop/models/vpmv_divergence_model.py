"""VpmvDivergenceModel — sinyal sonrası VPMV momentum ayrışması tablosu için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 5 —
vpmv_divergence_panel.py'nin pos/neg ikiz QTableWidget'larının yerini alır).

Pos (momentum devam ediyor) ve neg (momentum söndü) tabloları AYRI birer
VpmvDivergenceModel örneği (`positive=True/False`) — asıl +/− ayrımı zaten
panel'de (`_populate`) ham veri üzerinde yapılıyor, tf_alignment_panel'in
Long/Short ayrımıyla aynı desen (backend/panel veriyi böler, iki ayrı model
örneği kendi payını tutar) — paylaşımlı tek model + proxy-ile-bölme burada
gerekmiyor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor, QFont

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt
from desktop.theme import COLORS

COLUMNS = ["Sembol", "Δ VPMV", "Şimdi", "vs Med", "Sinyal", "Pre", "Zaman"]
COL_SYMBOL = 0
COL_DELTA = 1
COL_NOW = 2
COL_VS_MED = 3
COL_SIG = 4
COL_PRE = 5
COL_TIME = 6

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)

_BG_POS_STRONG = QColor(0, 120, 40, 150)
_BG_POS_SOFT = QColor(0, 80, 20, 80)
_BG_NEG_STRONG = QColor(180, 20, 20, 150)
_BG_NEG_SOFT = QColor(120, 10, 10, 80)

_BOLD = QFont("Courier New", 11, QFont.Weight.Bold)
_MONO = QFont("Courier New", 11)


@dataclass
class VpmvDivergenceRow:
    symbol: str
    delta: float
    now: float
    sig: float
    pre: float
    vs_med: float
    time_str: str


def _build_row(item: dict) -> VpmvDivergenceRow:
    return VpmvDivergenceRow(
        symbol=item["symbol"],
        delta=item["delta"],
        now=item.get("now", 0.0),
        sig=item.get("sig", 0.0),
        pre=item.get("pre", 0.0),
        vs_med=item.get("vs_med", 0.0),
        time_str=item.get("time_str", "—"),
    )


def _update_row(row: VpmvDivergenceRow, item: dict) -> None:
    new = _build_row(item)
    row.__dict__.update(new.__dict__)


def _band_bg(value: float, positive: bool) -> Optional[QColor]:
    abs_v = abs(value)
    if abs_v >= 20:
        return _BG_POS_STRONG if positive else _BG_NEG_STRONG
    if abs_v >= 10:
        return _BG_POS_SOFT if positive else _BG_NEG_SOFT
    return _C_TRANSPARENT


class VpmvDivergenceModel(IdKeyedTableModel):
    COLUMNS = COLUMNS
    build_row = staticmethod(_build_row)
    update_row = staticmethod(_update_row)

    def __init__(self, positive: bool, parent=None):
        super().__init__(parent)
        self._positive = positive
        self._d_color = _C_GREEN if positive else _C_RED

    def _id_of_item(self, item: dict) -> str:
        return item["symbol"]

    def _id_of_row(self, row: VpmvDivergenceRow) -> str:
        return row.symbol

    def _display(self, row: VpmvDivergenceRow, col: int) -> str:
        if col == COL_SYMBOL:
            return row.symbol
        if col == COL_DELTA:
            return f"{row.delta:+.1f}"
        if col == COL_NOW:
            return f"{row.now:.0f}"
        if col == COL_VS_MED:
            return f"{row.vs_med:+.0f}"
        if col == COL_SIG:
            return f"{row.sig:.0f}"
        if col == COL_PRE:
            return f"{row.pre:.0f}"
        if col == COL_TIME:
            return row.time_str
        return ""

    def _foreground(self, row: VpmvDivergenceRow, col: int):
        if col == COL_SYMBOL:
            return self._d_color
        if col == COL_DELTA:
            return self._d_color
        if col in (COL_NOW, COL_SIG, COL_PRE, COL_TIME):
            return _C_MUTED
        if col == COL_VS_MED:
            return _C_GREEN if row.vs_med >= 0 else _C_RED
        return None

    def _background(self, row: VpmvDivergenceRow, col: int):
        if col == COL_DELTA:
            return _band_bg(row.delta, self._positive)
        if col == COL_VS_MED:
            return _band_bg(row.vs_med, row.vs_med >= 0)
        return None

    def _font(self, row: VpmvDivergenceRow, col: int):
        return _BOLD if col == COL_SYMBOL else _MONO

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.FontRole
            and index.isValid()
            and 0 <= index.row() < len(self._rows)
        ):
            return self._font(self._rows[index.row()], index.column())
        return super().data(index, role)


class VpmvDivergenceProxyModel(QSortFilterProxyModel):
    _NUMERIC_ATTR = {
        COL_DELTA: "delta",
        COL_NOW: "now",
        COL_VS_MED: "vs_med",
        COL_SIG: "sig",
        COL_PRE: "pre",
    }

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
