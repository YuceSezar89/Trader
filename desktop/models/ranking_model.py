"""RankingModel — tüm coinleri VPMV güç skoruna göre sıralayan Güç Sıralaması
tablosu için QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 3).

Kolonlar: # | Sembol | 5m | 15m | 1h | 4h | Birleşik | RSI Cross | Z-Conf |
R-Score | TF Uyum | VS BTC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor, QFont

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt, sign_color
from desktop.theme import COLORS

COLUMNS = [
    "#",
    "Sembol",
    "5m",
    "15m",
    "1h",
    "4h",
    "Birleşik",
    "RSI Cross",
    "Z-Conf",
    "R-Score",
    "TF Uyum",
    "VS BTC",
]
COL_RANK = 0
COL_SYMBOL = 1
COL_5M = 2
COL_15M = 3
COL_1H = 4
COL_4H = 5
COL_COMBINED = 6
COL_RSICROSS = 7
COL_ZCONF = 8
COL_RSCORE = 9
COL_ALIGN = 10
COL_VSBTC = 11

_SCORE_COLS = (COL_5M, COL_15M, COL_1H, COL_4H, COL_COMBINED, COL_RSICROSS)

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_YELLOW = QColor(COLORS["yellow"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])

_BG_STRONG_BULL = QColor(0, 120, 40, 120)
_BG_SOFT_BULL = QColor(0, 80, 20, 60)
_BG_STRONG_BEAR = QColor(180, 20, 20, 120)
_BG_SOFT_BEAR = QColor(120, 10, 10, 60)

_BOLD = QFont()
_BOLD.setBold(True)

PINE_20 = {
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "LTCUSDT",
    "DOTUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "LINKUSDT",
    "VETUSDT",
    "XLMUSDT",
    "NEARUSDT",
    "WIFUSDT",
    "ZRXUSDT",
    "ATOMUSDT",
    "CAKEUSDT",
    "KSMUSDT",
}


def _row_bg(rank_score: float, direction: str) -> Optional[QColor]:
    if rank_score >= 80:
        return _BG_STRONG_BULL if direction == "long" else _BG_STRONG_BEAR
    if rank_score >= 60:
        return _BG_SOFT_BULL if direction == "long" else _BG_SOFT_BEAR
    return None


@dataclass
class RankingRow:  # pylint: disable=too-many-instance-attributes
    symbol: str
    rank: int
    rank_delta: int
    direction: str
    score_5m: Optional[float]
    score_15m: Optional[float]
    score_1h: Optional[float]
    score_4h: Optional[float]
    combined: Optional[float]
    rsi_cross_combined: Optional[float]
    z_confluence: Optional[float]
    r_score: Optional[float]
    alignment_count: int
    tf_count: int
    aligned: bool
    vs_btc: Optional[float]
    bg: Optional[QColor]


def _build_from_item(item: dict) -> RankingRow:
    rank_score = item.get("rank_score", 50)
    direction = item.get("direction", "long")
    return RankingRow(
        symbol=item["symbol"],
        rank=item["rank"],
        rank_delta=item.get("rank_delta", 0),
        direction=direction,
        score_5m=item.get("score_5m"),
        score_15m=item.get("score_15m"),
        score_1h=item.get("score_1h"),
        score_4h=item.get("score_4h"),
        combined=item.get("combined", 50),
        rsi_cross_combined=item.get("rsi_cross_combined"),
        z_confluence=item.get("z_confluence"),
        r_score=item.get("r_score"),
        alignment_count=item.get("alignment_count", 0),
        tf_count=item.get("tf_count", 0),
        aligned=item.get("aligned", False),
        vs_btc=item.get("vs_btc"),
        bg=_row_bg(rank_score, direction),
    )


def _update_from_item(row: RankingRow, item: dict) -> None:
    new = _build_from_item(item)
    row.__dict__.update(new.__dict__)


def _fmt_score(v: Optional[float]) -> str:
    return f"{v:.0f}" if v is not None else "—"


def _fmt_signed(v: Optional[float], prec: int) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{prec}f}"


class RankingModel(IdKeyedTableModel):
    COLUMNS = COLUMNS

    def _id_of_item(self, item: dict) -> str:
        return item["symbol"]

    def _id_of_row(self, row: RankingRow) -> str:
        return row.symbol

    build_row = staticmethod(_build_from_item)
    update_row = staticmethod(_update_from_item)

    def _display(self, row: RankingRow, col: int) -> str:
        if col == COL_RANK:
            return str(row.rank)
        if col == COL_SYMBOL:
            if row.rank_delta > 0:
                return f"{row.symbol} ↑{row.rank_delta}"
            if row.rank_delta < 0:
                return f"{row.symbol} ↓{abs(row.rank_delta)}"
            return row.symbol
        if col in _SCORE_COLS:
            val = {
                COL_5M: row.score_5m,
                COL_15M: row.score_15m,
                COL_1H: row.score_1h,
                COL_4H: row.score_4h,
                COL_COMBINED: row.combined,
                COL_RSICROSS: row.rsi_cross_combined,
            }[col]
            return _fmt_score(val)
        if col == COL_ZCONF:
            return _fmt_signed(row.z_confluence, 2)
        if col == COL_RSCORE:
            return _fmt_signed(row.r_score, 3)
        if col == COL_ALIGN:
            check = "✓" if row.aligned else "~"
            return f"{check} {row.alignment_count}/{row.tf_count}"
        if col == COL_VSBTC:
            return _fmt_signed(row.vs_btc, 1)
        return ""

    def _foreground(self, row: RankingRow, col: int):
        if col == COL_RANK:
            return _C_MUTED
        if col == COL_SYMBOL:
            return sign_color(row.rank_delta, zero_is_muted=False)
        if col in _SCORE_COLS:
            val = {
                COL_5M: row.score_5m,
                COL_15M: row.score_15m,
                COL_1H: row.score_1h,
                COL_4H: row.score_4h,
                COL_COMBINED: row.combined,
                COL_RSICROSS: row.rsi_cross_combined,
            }[col]
            if val is None:
                return _C_MUTED
            if val >= 55:
                return _C_GREEN
            if val <= 45:
                return _C_RED
            return _C_MUTED
        if col == COL_ZCONF:
            v = row.z_confluence
            if v is None:
                return _C_MUTED
            if v >= 1.5:
                return _C_GREEN
            if v >= 0.5:
                return QColor(100, 200, 100)
            if v <= -1.5:
                return _C_RED
            if v <= -0.5:
                return QColor(200, 100, 100)
            return _C_MUTED
        if col == COL_RSCORE:
            return sign_color(row.r_score)
        if col == COL_ALIGN:
            return _C_GREEN if row.aligned else _C_YELLOW
        if col == COL_VSBTC:
            return sign_color(row.vs_btc)
        return None

    def _background(self, row: RankingRow, col: int):
        return row.bg

    def _font(self, row: RankingRow, col: int):
        return _BOLD if col == COL_COMBINED else None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if (
            role == Qt.ItemDataRole.FontRole
            and index.isValid()
            and 0 <= index.row() < len(self._rows)
        ):
            return self._font(self._rows[index.row()], index.column())
        return super().data(index, role)


class RankingProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ""
        self._pine_only = False

    def set_search(self, text: str) -> None:
        self._search = text.strip().upper()
        self.invalidateFilter()

    def set_pine_filter(self, only_pine: bool) -> None:
        self._pine_only = only_pine
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._pine_only and row.symbol not in PINE_20:
            return False
        if self._search and self._search not in row.symbol:
            return False
        return True

    _NUMERIC_ATTR = {
        COL_RANK: "rank",
        COL_5M: "score_5m",
        COL_15M: "score_15m",
        COL_1H: "score_1h",
        COL_4H: "score_4h",
        COL_COMBINED: "combined",
        COL_RSICROSS: "rsi_cross_combined",
        COL_ZCONF: "z_confluence",
        COL_RSCORE: "r_score",
        COL_ALIGN: "alignment_count",
        COL_VSBTC: "vs_btc",
    }

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        col = left.column()
        attr = self._NUMERIC_ATTR.get(col)
        if attr is not None:
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)
