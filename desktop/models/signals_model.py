"""
SignalsModel — aktif sinyaller için QAbstractTableModel.

Sütunlar: Sembol | Tip | TF | VPM | MTF | α | β | Z-Score% | P&L% | Süre
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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

COLUMNS = ["Sembol", "Tip", "TF", "İndikatör", "VPMV", "MTF", "α", "β", "Z-Score%", "P&L%", "Süre"]

COL_SYMBOL    = 0
COL_TYPE      = 1
COL_TF        = 2
COL_INDICATOR = 3
COL_VPM       = 4
COL_MTF       = 5
COL_ALPHA     = 6
COL_BETA      = 7
COL_ZSCORE    = 8
COL_PNL       = 9
COL_AGE       = 10


def _fmt_score(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _fmt_ratio(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_age(ts: Optional[datetime]) -> str:
    if ts is None:
        return "—"
    now = datetime.now() if ts.tzinfo is None else datetime.now(tz=timezone.utc)
    secs = int((now - ts).total_seconds())
    if secs < 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}dk"
    hours = mins // 60
    rem_min = mins % 60
    if hours < 24:
        return f"{hours}sa {rem_min}dk"
    days = hours // 24
    rem_hr = hours % 24
    return f"{days}g {rem_hr}sa"


@dataclass
class SignalRow:
    id: int
    symbol: str
    signal_type: str          # "LONG" | "SHORT"
    interval: str
    entry_price: float
    vpm: Optional[float]
    mtf: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    zscore: Optional[float]
    timestamp: Optional[datetime]
    indicators: str = ""
    status: str = "active"
    current_price: float = 0.0
    st_confirmed: Optional[bool] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    omega: Optional[float] = None
    treynor: Optional[float] = None
    info_ratio: Optional[float] = None
    pnl_pct: Optional[float] = field(default=None, init=False)

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.entry_price and price:
            if self.signal_type == "LONG":
                self.pnl_pct = (price - self.entry_price) / self.entry_price * 100
            else:
                self.pnl_pct = (self.entry_price - price) / self.entry_price * 100


class SignalsModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[SignalRow] = []
        self._id_index: dict[int, int] = {}      # signal id → satır
        self._sym_rows: dict[str, list[int]] = {}  # symbol → [row indices]
        # (symbol, interval, signal_type) → aktif satır sayısı — 2+ ise çoklu sinyal var
        self._coincident: dict[tuple[str, str, str], int] = {}

    # ── QAbstractTableModel ───────────────────────────────────────────────────

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(row)
        if role == Qt.ItemDataRole.ForegroundRole:
            return self._foreground(row, col)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == COL_SYMBOL:
                return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        if role == Qt.ItemDataRole.UserRole:
            return row.symbol
        return None

    def _display(self, row: SignalRow, col: int) -> str:
        match col:
            case _ if col == COL_SYMBOL:    return row.symbol
            case _ if col == COL_TYPE:      return row.signal_type
            case _ if col == COL_TF:        return row.interval
            case _ if col == COL_INDICATOR: return row.indicators or "—"
            case _ if col == COL_VPM:       return _fmt_score(row.vpm)
            case _ if col == COL_MTF:       return f"{int(row.mtf)}" if row.mtf is not None else "—"
            case _ if col == COL_ALPHA:     return _fmt_ratio(row.alpha)
            case _ if col == COL_BETA:      return _fmt_ratio(row.beta)
            case _ if col == COL_ZSCORE:    return _fmt_score(row.zscore)
            case _ if col == COL_PNL:       return _fmt_pnl(row.pnl_pct)
            case _ if col == COL_AGE:       return _fmt_age(row.timestamp)
        return ""

    def _tooltip(self, row: SignalRow) -> str:
        def _r(v, fmt=".2f"): return f"{v:{fmt}}" if v is not None else "—"
        st = ("✓ Onaylı" if row.st_confirmed else "✗ Onaysız") if row.st_confirmed is not None else "—"
        mtf = f"{int(row.mtf)}" if row.mtf is not None else "—"
        return (
            f"{row.symbol}  {row.signal_type}  {row.interval}  |  {row.indicators}\n"
            f"{'─'*52}\n"
            f"  Alpha    {_r(row.alpha, '+.4f'):>10}    Beta     {_r(row.beta):>8}\n"
            f"  Sharpe   {_r(row.sharpe):>10}    Sortino  {_r(row.sortino):>8}\n"
            f"  Calmar   {_r(row.calmar):>10}    Omega    {_r(row.omega):>8}\n"
            f"  Treynor  {_r(row.treynor):>10}    Info R   {_r(row.info_ratio):>8}\n"
            f"{'─'*52}\n"
            f"  VPMV: {_r(row.vpm, '.1f')}   MTF: {mtf}   ST: {st}"
        )

    def _foreground(self, row: SignalRow, col: int) -> Optional[QColor]:
        if col == COL_TYPE:
            return QColor(COLORS["green"] if row.signal_type == "LONG" else COLORS["red"])
        if col == COL_INDICATOR:
            key = (row.symbol, row.interval, row.signal_type)
            if self._coincident.get(key, 0) >= 2:
                return QColor(COLORS["yellow"])
        if col == COL_PNL and row.pnl_pct is not None:
            if row.pnl_pct > 0:
                return QColor(COLORS["green"])
            if row.pnl_pct < 0:
                return QColor(COLORS["red"])
        if col == COL_MTF and row.mtf is not None:
            if row.mtf >= 100:
                return QColor(COLORS["green"])
            if row.mtf >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["red"])
        if col == COL_VPM and row.vpm is not None:
            if row.vpm >= 70:
                return QColor(COLORS["green"])
            if row.vpm >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col == COL_ALPHA and row.alpha is not None:
            return QColor(COLORS["green"] if row.alpha > 0 else COLORS["red"])
        return None

    # ── Veri yükleme / güncelleme ─────────────────────────────────────────────

    def load_signals(self, signals: list[dict]) -> None:
        self.beginResetModel()
        self._rows.clear()
        self._id_index.clear()
        self._sym_rows.clear()
        self._coincident.clear()
        for s in signals:
            self._append_row(s)
        self.endResetModel()

    def add_or_update(self, signal: dict) -> None:
        sid = signal.get("id")
        if sid in self._id_index:
            idx = self._id_index[sid]
            row = self._rows[idx]
            row.vpm = signal.get("vpms_score", row.vpm)
            row.mtf = signal.get("vpms_mtf_score", row.mtf)
            tl = self.index(idx, 0)
            br = self.index(idx, len(COLUMNS) - 1)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.DisplayRole])
        else:
            self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
            self._append_row(signal)
            self.endInsertRows()

    def _append_row(self, s: dict) -> None:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = None

        row = SignalRow(
            id=s.get("id", 0),
            symbol=s.get("symbol", ""),
            signal_type=s.get("signal_type", "").upper(),
            interval=s.get("interval", ""),
            entry_price=float(s.get("price") or 0),
            vpm=s.get("vpms_score"),
            mtf=s.get("vpms_mtf_score"),
            alpha=s.get("alpha"),
            beta=s.get("beta"),
            zscore=s.get("zscore_ratio_percent"),
            timestamp=ts,
            indicators=s.get("indicators") or "",
            status=s.get("status", "active"),
            st_confirmed=s.get("st_confirmed"),
            sharpe=s.get("sharpe_ratio"),
            sortino=s.get("sortino_ratio"),
            calmar=s.get("calmar_ratio"),
            omega=s.get("omega_ratio"),
            treynor=s.get("treynor_ratio"),
            info_ratio=s.get("information_ratio"),
        )
        idx = len(self._rows)
        self._rows.append(row)
        self._id_index[row.id] = idx
        self._sym_rows.setdefault(row.symbol, []).append(idx)
        key = (row.symbol, row.interval, row.signal_type)
        self._coincident[key] = self._coincident.get(key, 0) + 1

    @pyqtSlot(str, float, float)
    def on_price_updated(self, symbol: str, price: float, _change_pct: float) -> None:
        indices = self._sym_rows.get(symbol, [])
        for idx in indices:
            self._rows[idx].update_price(price)
            tl = self.index(idx, COL_PNL)
            br = self.index(idx, COL_PNL)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole])

    def signal_at(self, row: int) -> Optional[SignalRow]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


class SignalsProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._type_filter = ""
        self._tf_filter = ""
        self._indicator_filter = ""
        self._st_only = False
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(COL_SYMBOL)

    def set_type_filter(self, type_filter: str) -> None:
        self._type_filter = type_filter
        self.invalidateFilter()

    def set_tf_filter(self, tf: str) -> None:
        self._tf_filter = tf
        self.invalidateFilter()

    def set_indicator_filter(self, indicator: str) -> None:
        self._indicator_filter = indicator
        self.invalidateFilter()

    def set_st_filter(self, only_confirmed: bool) -> None:
        self._st_only = only_confirmed
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._type_filter and row.signal_type != self._type_filter:
            return False
        if self._tf_filter and row.interval != self._tf_filter:
            return False
        if self._indicator_filter and not row.indicators.startswith(self._indicator_filter):
            return False
        if self._st_only and row.st_confirmed is False:
            return False
        return super().filterAcceptsRow(source_row, source_parent)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        src = self.sourceModel()
        col = left.column()
        l_row = src._rows[left.row()]   # noqa: SLF001
        r_row = src._rows[right.row()]  # noqa: SLF001

        def _cmp(a, b):
            if a is None and b is None:
                return False
            if a is None:
                return True
            if b is None:
                return False
            return a < b

        match col:
            case _ if col == COL_VPM:       return _cmp(l_row.vpm, r_row.vpm)
            case _ if col == COL_MTF:       return _cmp(l_row.mtf, r_row.mtf)
            case _ if col == COL_ALPHA:     return _cmp(l_row.alpha, r_row.alpha)
            case _ if col == COL_BETA:      return _cmp(l_row.beta, r_row.beta)
            case _ if col == COL_ZSCORE:    return _cmp(l_row.zscore, r_row.zscore)
            case _ if col == COL_PNL:       return _cmp(l_row.pnl_pct, r_row.pnl_pct)
            case _ if col == COL_AGE:
                lt = l_row.timestamp or datetime.min.replace(tzinfo=timezone.utc)
                rt = r_row.timestamp or datetime.min.replace(tzinfo=timezone.utc)
                return lt < rt
            case _:
                return super().lessThan(left, right)
