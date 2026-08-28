"""DevisoModel — aktif sinyalleri devisso_score'a göre sıralayan tablo için
QAbstractTableModel (28 Ağu 2026, Model/View göçü Faz 2 — deviso_panel.py'nin
eski QTableWidget'ının yerini alır).

deviso_panel.py diğer panellerden farklı olarak PERİYODİK POLL değil, OLAY
GÜDÜMLÜ (on_signals_loaded/on_new_signal/on_signals_closed) — model TÜM aktif
sinyalleri tutar, arama/TF/yön filtreleri DevisoProxyModel.filterAcceptsRow'da
uygulanır (eskiden Python tarafında `_filtered_rows()` ile elle filtrelenip
sadece filtrelenmiş satırlar tabloya yazılıyordu).

"#" (sıra) sütunu artık PROXY'nin kendi görünür satır pozisyonundan (canlı,
mevcut sıralama/filtreye göre) hesaplanıyor — eskiden populate anında
devisso_score'a göre DONDURULMUŞ bir sıra numarasıydı, kullanıcı başka bir
sütuna göre sıralasa bile değişmiyordu. Bu, kasıtlı küçük bir davranış
iyileştirmesi (bkz. DevisoProxyModel.data).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt, sign_color
from desktop.theme import COLORS

COLUMNS = ["#", "Sembol", "TF", "Yön", "Score", "Δ", "Ratio", "Zaman"]
COL_RANK = 0
COL_SYMBOL = 1
COL_TF = 2
COL_DIR = 3
COL_SCORE = 4
COL_DELTA = 5
COL_RATIO = 6
COL_TIME = 7

COLUMN_TOOLTIPS = [
    "Sıralama — Devisso score'a göre yüksekten düşüğe",
    "İşlem çifti",
    "Zaman dilimi (timeframe)",
    "Sinyal yönü — Long (alım) veya Short (satım)",
    "Devisso Score (0-100) — RSI Verimliliği\nFiyat değişimi (%) / RSI değişimi oranının yüzdelik sırası (son 100 bar).\nYüksek → Az RSI ile çok fiyat hareketi — trend verimli ve sağlıklı\nDüşük → Aynı hareket için RSI çok yoruldu — trend zorlanıyor",
    "Delta — Önceki aynı yöndeki sinyale göre score farkı.\nPozitif → piyasa daha verimli hareket etti\nNegatif → piyasa daha çok zorlandı",
    "Efficiency Ratio — Mevcut score / Önceki sinyal score (aynı yön).\n>1.0 → piyasa öncekine göre daha verimli\n<1.0 → piyasa öncekine göre daha zorlandı",
    "Sinyalin açılma zamanı",
]

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_WHITE = QColor(COLORS["text_primary"])
_BG_HIGH = QColor(0, 120, 40, 100)
_BG_LOW = QColor(180, 20, 20, 80)

TF_OPTIONS = ["Tüm TF", "1m", "5m", "15m", "1h", "4h", "1d"]


@dataclass
class DevisoRow:
    id: int
    symbol: str
    interval: str
    signal_type: str
    devisso_score: Optional[float]
    devisso_delta: Optional[float]
    devisso_ratio: Optional[float]
    opened_at: Any
    status: str = "active"


def _fmt_time(opened_at: Any) -> str:
    if isinstance(opened_at, datetime):
        return opened_at.strftime("%m-%d %H:%M")
    if isinstance(opened_at, str):
        return opened_at[5:16]
    return "-"


class DevisoModel(IdKeyedTableModel):
    COLUMNS = COLUMNS

    def _id_of_item(self, item: dict) -> int:
        return item["id"]

    def _id_of_row(self, row: DevisoRow) -> int:
        return row.id

    @staticmethod
    def build_row(item: dict) -> DevisoRow:
        return DevisoRow(
            id=item["id"],
            symbol=item.get("symbol", ""),
            interval=item.get("interval", ""),
            signal_type=item.get("signal_type", ""),
            devisso_score=item.get("devisso_score"),
            devisso_delta=item.get("devisso_delta"),
            devisso_ratio=item.get("devisso_ratio"),
            opened_at=item.get("opened_at"),
            status=item.get("status", "active"),
        )

    @staticmethod
    def update_row(row: DevisoRow, item: dict) -> None:
        row.symbol = item.get("symbol", row.symbol)
        row.interval = item.get("interval", row.interval)
        row.signal_type = item.get("signal_type", row.signal_type)
        row.devisso_score = item.get("devisso_score")
        row.devisso_delta = item.get("devisso_delta")
        row.devisso_ratio = item.get("devisso_ratio")
        row.opened_at = item.get("opened_at", row.opened_at)
        row.status = item.get("status", row.status)

    def _display(self, row: DevisoRow, col: int) -> str:
        if col == COL_RANK:
            return ""  # proxy hesaplar, bkz. DevisoProxyModel.data
        if col == COL_SYMBOL:
            return row.symbol
        if col == COL_TF:
            return row.interval
        if col == COL_DIR:
            return row.signal_type
        if col == COL_SCORE:
            return f"{row.devisso_score:.1f}" if row.devisso_score is not None else "-"
        if col == COL_DELTA:
            if row.devisso_delta is None:
                return "-"
            prefix = "+" if row.devisso_delta >= 0 else ""
            return f"{prefix}{row.devisso_delta:.1f}"
        if col == COL_RATIO:
            return f"{row.devisso_ratio:.2f}x" if row.devisso_ratio is not None else "-"
        if col == COL_TIME:
            return _fmt_time(row.opened_at)
        return ""

    def _foreground(self, row: DevisoRow, col: int):
        if col == COL_RANK or col == COL_TF or col == COL_TIME:
            return _C_MUTED
        if col == COL_SYMBOL:
            return _C_WHITE
        if col == COL_DIR:
            return _C_GREEN if row.signal_type == "Long" else _C_RED
        if col == COL_SCORE:
            s = row.devisso_score
            if s is None:
                return _C_MUTED
            if s >= 65:
                return _C_GREEN
            if s <= 35:
                return _C_RED
            return _C_WHITE
        if col == COL_DELTA:
            if row.devisso_delta is None:
                return _C_MUTED
            return _C_GREEN if row.devisso_delta >= 0 else _C_RED
        if col == COL_RATIO:
            if row.devisso_ratio is None:
                return _C_MUTED
            return sign_color(row.devisso_ratio - 1.0, zero_is_muted=False)
        return None

    def _background(self, row: DevisoRow, col: int):
        if col == COL_SCORE and row.devisso_score is not None:
            if row.devisso_score >= 65:
                return _BG_HIGH
            if row.devisso_score <= 35:
                return _BG_LOW
        return None

    def _text_alignment(self, col: int) -> int:
        if col == COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)

    def _tooltip(self, row: DevisoRow, col: int):
        return None  # sütun başlığı tooltip'i DevisoProxyModel/panel'de (header)


class DevisoProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ""
        self._tf = ""
        self._direction = "Tümü"
        self.setDynamicSortFilter(True)

    def set_search(self, text: str) -> None:
        self._search = text.strip().upper()
        self.invalidateFilter()

    def set_tf(self, tf: str) -> None:
        self._tf = tf
        self.invalidateFilter()

    def set_direction(self, direction: str) -> None:
        self._direction = direction
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if row.status != "active":
            return False
        if self._search and self._search not in row.symbol.upper():
            return False
        if self._tf and self._tf != "Tüm TF" and row.interval != self._tf:
            return False
        if self._direction == "Long" and row.signal_type != "Long":
            return False
        if self._direction == "Short" and row.signal_type != "Short":
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        col = left.column()
        if col in (COL_SCORE, COL_DELTA, COL_RATIO):
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            attr = {
                COL_SCORE: "devisso_score",
                COL_DELTA: "devisso_delta",
                COL_RATIO: "devisso_ratio",
            }[col]
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        # "#" sütunu: canlı görünür satır pozisyonu (mevcut sıralama/filtreye
        # göre) — sabit bir snapshot-anı sırası DEĞİL, bkz. modül docstring'i.
        if index.column() == COL_RANK and role == Qt.ItemDataRole.DisplayRole:
            return str(index.row() + 1)
        return super().data(index, role)
