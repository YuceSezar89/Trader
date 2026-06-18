"""
WatchlistModel — 200 sembolü tutan, satır bazlı güncellenen tablo modeli.

QAbstractTableModel kullanılır; QTableWidget yerine bu tercih edilir çünkü
büyük veri setlerinde sadece değişen hücreyi yeniden çizer (dataChanged).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSlot,
)
from PyQt6.QtGui import QColor

from desktop.theme import COLORS

COLUMNS = ["Sembol", "Fiyat", "Değişim %", "Hacim", "Funding"]
COL_SYMBOL  = 0
COL_PRICE   = 1
COL_CHANGE  = 2
COL_VOLUME  = 3
COL_FUNDING = 4


@dataclass
class SymbolRow:
    symbol: str
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    prev_price: float = 0.0
    funding_rate: float = 0.0

    def update_price(self, price: float, change_pct: float, volume: float = 0.0, funding_rate: float = 0.0) -> None:
        self.prev_price = self.price
        self.price = price
        self.change_pct = change_pct
        if volume:
            self.volume = volume
        self.funding_rate = funding_rate


def _format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


def _format_volume(vol: float) -> str:
    if vol >= 1_000_000_000:
        return f"{vol/1_000_000_000:.1f}B"
    if vol >= 1_000_000:
        return f"{vol/1_000_000:.1f}M"
    if vol >= 1_000:
        return f"{vol/1_000:.1f}K"
    return f"{vol:.0f}"


class WatchlistModel(QAbstractTableModel):
    """
    Sembol listesini tutan tablo modeli.

    Sadece güncellenen satırda `dataChanged` yayınlanır;
    tüm tablo yeniden çizilmez.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[SymbolRow] = []
        self._symbol_index: dict[str, int] = {}  # symbol → satır numarası

    # ── QAbstractTableModel arayüzü ───────────────────────────────────────

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._rows):
            return None

        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)

        if role == Qt.ItemDataRole.ForegroundRole:
            return self._foreground(row, col)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == COL_SYMBOL:
                return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        if role == Qt.ItemDataRole.UserRole:
            return row.symbol

        return None

    def _display(self, row: SymbolRow, col: int) -> str:
        match col:
            case _ if col == COL_SYMBOL: return row.symbol
            case _ if col == COL_PRICE:  return _format_price(row.price) if row.price else "—"
            case _ if col == COL_CHANGE:
                if row.change_pct == 0:
                    return "—"
                sign = "+" if row.change_pct > 0 else ""
                return f"{sign}{row.change_pct:.2f}%"
            case _ if col == COL_VOLUME:  return _format_volume(row.volume) if row.volume else "—"
            case _ if col == COL_FUNDING:
                if row.funding_rate == 0:
                    return "—"
                pct = row.funding_rate * 100
                sign = "+" if pct > 0 else ""
                return f"{sign}{pct:.4f}%"
        return ""

    def _foreground(self, row: SymbolRow, col: int) -> Optional[QColor]:
        if col == COL_CHANGE:
            if row.change_pct > 0:
                return QColor(COLORS["green"])
            if row.change_pct < 0:
                return QColor(COLORS["red"])
            return QColor(COLORS["text_muted"])
        if col == COL_FUNDING:
            if row.funding_rate > 0.0003:      # >0.03% — yüksek long baskısı
                return QColor(COLORS["red"])
            if row.funding_rate > 0:
                return QColor(COLORS["text_muted"])
            if row.funding_rate < -0.0003:     # <-0.03% — yüksek short baskısı
                return QColor(COLORS["green"])
            if row.funding_rate < 0:
                return QColor(COLORS["text_muted"])
        return None

    # ── Veri yükleme / güncelleme ─────────────────────────────────────────

    def load_symbols(self, symbols: list[str]) -> None:
        """İlk sembol listesini yükler."""
        self.beginResetModel()
        self._rows = [SymbolRow(symbol=s) for s in sorted(symbols)]
        self._symbol_index = {r.symbol: i for i, r in enumerate(self._rows)}
        self.endResetModel()

    @pyqtSlot(str, float, float, float, float)
    def on_price_updated(self, symbol: str, price: float, change_pct: float, volume: float = 0.0, funding_rate: float = 0.0) -> None:
        idx = self._symbol_index.get(symbol)
        if idx is None:
            return
        self._rows[idx].update_price(price, change_pct, volume, funding_rate)
        top_left = self.index(idx, COL_PRICE)
        bottom_right = self.index(idx, COL_FUNDING)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole])

    def symbol_at(self, row: int) -> Optional[str]:
        if 0 <= row < len(self._rows):
            return self._rows[row].symbol
        return None


class WatchlistProxyModel(QSortFilterProxyModel):
    """Arama filtresi + sütun sıralama için proxy model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(COL_SYMBOL)
        self.setSortRole(Qt.ItemDataRole.UserRole + 1)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        src = self.sourceModel()
        col = left.column()
        l_row = src._rows[left.row()]   # noqa: SLF001
        r_row = src._rows[right.row()]  # noqa: SLF001

        match col:
            case _ if col == COL_PRICE:
                return l_row.price < r_row.price
            case _ if col == COL_CHANGE:
                return l_row.change_pct < r_row.change_pct
            case _ if col == COL_VOLUME:
                return l_row.volume < r_row.volume
            case _ if col == COL_FUNDING:
                return l_row.funding_rate < r_row.funding_rate
            case _:
                return super().lessThan(left, right)
