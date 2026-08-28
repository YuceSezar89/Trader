"""TradeXRayModel — açık/kapalı işlem listesi tablosu için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 7 —
trade_xray_panel.py'nin eski QTableWidget'ının yerini alır).

Arama/strateji/durum/yön filtreleri artık TradeXRayProxyModel.filterAcceptsRow'da
uygulanıyor (eskiden Python tarafında `_apply_search_filter()` ile elle
filtrelenip sadece filtrelenmiş liste tabloya yazılıyordu, deviso_panel.py'nin
Faz 2'deki göçüyle aynı desen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt
from desktop.theme import COLORS

COLUMNS = ["Sembol", "Strateji", "Yön", "Durum", "Açılış", "PnL%"]
COL_SYMBOL = 0
COL_STRATEGY = 1
COL_SIDE = 2
COL_STATUS = 3
COL_OPENED = 4
COL_PNL = 5

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])


@dataclass
class TradeXRayRow:
    id: int
    symbol: str
    strategy: str
    signal_type: str
    status: str
    opened_at_str: str
    pnl_pct: Optional[float]
    raw: dict  # TA bilgi etiketi (entry_features) için ham veri saklanır


def _build_row(item: dict) -> TradeXRayRow:
    return TradeXRayRow(
        id=item["id"],
        symbol=item.get("symbol", ""),
        strategy=item.get("strategy", ""),
        signal_type=item.get("signal_type", ""),
        status=item.get("status", ""),
        opened_at_str=item.get("opened_at_str") or "—",
        pnl_pct=item.get("pnl_pct"),
        raw=item,
    )


def _update_row(row: TradeXRayRow, item: dict) -> None:
    new = _build_row(item)
    row.__dict__.update(new.__dict__)


class TradeXRayModel(IdKeyedTableModel):
    COLUMNS = COLUMNS
    build_row = staticmethod(_build_row)
    update_row = staticmethod(_update_row)

    def _id_of_item(self, item: dict) -> int:
        return item["id"]

    def _id_of_row(self, row: TradeXRayRow) -> int:
        return row.id

    def _display(self, row: TradeXRayRow, col: int) -> str:
        if col == COL_SYMBOL:
            return row.symbol
        if col == COL_STRATEGY:
            return row.strategy
        if col == COL_SIDE:
            return row.signal_type
        if col == COL_STATUS:
            if row.status == "open":
                return "Açık"
            if row.status == "closed":
                return "Kapalı"
            return row.status
        if col == COL_OPENED:
            return row.opened_at_str
        if col == COL_PNL:
            if row.pnl_pct is None:
                return "—"
            sign = "+" if row.pnl_pct > 0 else ""
            return f"{sign}{row.pnl_pct:.2f}"
        return ""

    def _foreground(self, row: TradeXRayRow, col: int):
        if col == COL_SYMBOL:
            return _C_WHITE
        if col == COL_STRATEGY:
            return _C_MUTED
        if col == COL_SIDE:
            if row.signal_type == "Long":
                return _C_GREEN
            if row.signal_type == "Short":
                return _C_RED
            return _C_MUTED
        if col == COL_STATUS:
            return _C_GREEN if row.status == "open" else _C_MUTED
        if col == COL_OPENED:
            return _C_MUTED
        if col == COL_PNL:
            if row.pnl_pct is None:
                return _C_MUTED
            return _C_GREEN if row.pnl_pct > 0 else _C_RED
        return None

    def _text_alignment(self, col: int) -> int:
        if col == COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)


class TradeXRayProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ""
        self._strategy = "Tümü"
        self._status = "Tümü"
        self._side = "Tümü"

    def set_search(self, text: str) -> None:
        self._search = text.strip().upper()
        self.invalidateFilter()

    def set_strategy(self, strategy: str) -> None:
        self._strategy = strategy
        self.invalidateFilter()

    def set_status(self, status: str) -> None:
        self._status = status
        self.invalidateFilter()

    def set_side(self, side: str) -> None:
        self._side = side
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._search and self._search not in row.symbol.upper():
            return False
        if self._strategy != "Tümü" and row.strategy != self._strategy:
            return False
        if self._status != "Tümü" and row.status != self._status:
            return False
        if self._side != "Tümü" and row.signal_type != self._side:
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        if left.column() == COL_PNL:
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            return none_safe_lt(l_row.pnl_pct, r_row.pnl_pct)
        return super().lessThan(left, right)
