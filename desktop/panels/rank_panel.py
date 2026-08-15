"""
RankPanel — Devisso Döngüsü / Totalamount Rank-1 sekmesi (13 Ağu 2026).

Aktif Supertrend Long adaylarının Totalamount tablosu (büyükten küçüğe).

13 Ağu 2026, ÜÇÜNCÜ düzeltme — çizgi grafik alt-sekmesi kaldırıldı. Kanıtlanan
davranış: uygulama hiç dokunulmadan (Rank sekmesi hiç açılmadan) 200sn+ boyunca
tamamen stabil (RSS doğrusal, ~110MB/200sn) — Rank sekmesi (özellikle Grafik alt-
sekmesi) her açıldığında ~70-95sn içinde RSS 900MB'den 7-14GB'a fırlayıp masaüstü
PC'yi kilitledi (2 kez gerçekleşti, canlıda). Debug loglamayla RankPanel'in kendi
_update_chart/_populate kodunun patlama anında ÇALIŞMADIĞI kanıtlandı (worker
cycle 2 hiç başlamamıştı) — yani Python tarafında görünür bir sebep yok, sorun
Qt/pyqtgraph render pipeline'ının (Python log'unun göremediği) bir yerinde,
kök neden tam izole edilemedi. Kullanıcı kararıyla: kök neden bulunana kadar
grafik tamamen kaldırıldı, sadece güvenli olduğu doğrulanmış Tablo kalıyor.
"""

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

_COLS = ["Sembol", "Totalamount", "Rank"]
_COL_SYMBOL = 0
_COL_VALUE = 1
_COL_RANK = 2

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


def _make_table() -> QTableWidget:
    t = QTableWidget(0, len(_COLS))
    t.setHorizontalHeaderLabels(_COLS)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(False)
    t.setSortingEnabled(False)
    t.setShowGrid(False)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.Interactive)
    hh.setSectionResizeMode(_COL_RANK, QHeaderView.ResizeMode.Interactive)
    return t


def _make_search_box(placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(24)
    box.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 3px; "
        f"padding: 0 4px; font-size: 11px;"
    )
    return box


class RankPanel(QWidget):
    """Totalamount Rank-1 aday tablosu (grafik yok — bkz. modül docstring'i)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ""
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(
            self._muted_label("Devisso Döngüsü — Totalamount Rank-1 (Supertrend Long adayları, 5m)")
        )
        ctrl.addStretch()
        self._status_label = QLabel("Aday bekleniyor…")
        self._status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        ctrl.addWidget(self._status_label)
        root.addLayout(ctrl)

        hdr = QHBoxLayout()
        title = QLabel("▲ TOTALAMOUNT SIRALAMASI")
        title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._search_box = _make_search_box("Ara…")
        self._search_box.textChanged.connect(self._on_search)
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._search_box)
        root.addLayout(hdr)

        self._table = _make_table()
        root.addWidget(self._table)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    # ── Slot'lar ──────────────────────────────────────────────────────────

    def _on_search(self, text: str) -> None:
        self._search = text.strip().upper()
        self._apply_filter()

    def _apply_filter(self) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_SYMBOL)
            if item is None:
                continue
            symbol = item.text()
            self._table.setRowHidden(row, bool(self._search) and self._search not in symbol)

    @pyqtSlot(object)
    def on_totalamount_updated(self, result: dict) -> None:
        current = result.get("current", {})
        ranking = result.get("ranking", {})
        self._status_label.setText(
            f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}  •  {len(current)} aday"
        )
        self._populate(current, ranking)

    @pyqtSlot(str)
    def on_status_updated(self, msg: str) -> None:
        self._status_label.setText(msg)

    # ── Tablo doldurma ────────────────────────────────────────────────────

    def _populate(self, current: dict, ranking: dict) -> None:
        rows = sorted(current.items(), key=lambda kv: kv[1], reverse=True)

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        mono = QFont("Courier New", 11)
        bold = QFont("Courier New", 11, QFont.Weight.Bold)

        for row_idx, (symbol, value) in enumerate(rows):
            color = _C_GREEN if value >= 0 else _C_RED

            sym_item = QTableWidgetItem(symbol)
            sym_item.setFont(bold if row_idx == 0 else mono)
            sym_item.setForeground(color if row_idx == 0 else _C_MUTED)
            self._table.setItem(row_idx, _COL_SYMBOL, sym_item)

            val_item = _NumericItem(f"{value:+.2f}")
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            val_item.setFont(mono)
            val_item.setData(Qt.ItemDataRole.UserRole, value)
            val_item.setForeground(color)
            val_item.setBackground(QColor(0, 120, 40, 90) if row_idx == 0 else _C_TRANSPARENT)
            self._table.setItem(row_idx, _COL_VALUE, val_item)

            rank = ranking.get(symbol, row_idx + 1)
            rank_item = _NumericItem(str(rank))
            rank_item.setData(Qt.ItemDataRole.UserRole, rank)
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            rank_item.setFont(mono)
            rank_item.setForeground(_C_MUTED)
            self._table.setItem(row_idx, _COL_RANK, rank_item)

        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        self._apply_filter()
