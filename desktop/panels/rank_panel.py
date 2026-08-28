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

28 Ağu 2026: QTableWidget → Model/View göçü (Faz 1, bkz. proje hafızası
"masaüstü panel mimari denetimi"). Eski elle yönetilen get-or-create/satır
haritası/sıralama-bayatlığı kodu kaldırıldı — RankModel (QAbstractTableModel)
bunu içeriden, tek doğru şekilde çözüyor.
"""

from datetime import datetime

from PyQt6.QtCore import pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from desktop.models.rank_model import COL_RANK, COL_SYMBOL, RankModel, RankProxyModel
from desktop.theme import COLORS


def _make_view(model: RankModel, proxy: RankProxyModel) -> QTableView:
    v = QTableView()
    v.setModel(proxy)
    v.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    v.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    v.setAlternatingRowColors(False)
    v.setSortingEnabled(True)
    v.setShowGrid(False)
    v.verticalHeader().setVisible(False)
    v.verticalHeader().setDefaultSectionSize(24)
    hh = v.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    hh.setSectionResizeMode(COL_SYMBOL, QHeaderView.ResizeMode.Interactive)
    hh.setSectionResizeMode(COL_RANK, QHeaderView.ResizeMode.Interactive)
    return v


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
        self._resized_once = False
        self._model = RankModel(self)
        self._proxy = RankProxyModel(self)
        self._proxy.setSourceModel(self._model)
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

        self._view = _make_view(self._model, self._proxy)
        root.addWidget(self._view)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    # ── Slot'lar ──────────────────────────────────────────────────────────

    def _on_search(self, text: str) -> None:
        self._proxy.setFilterFixedString(text.strip())

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
        items = [
            {
                "symbol": symbol,
                "value": value,
                "rank": ranking.get(symbol, idx + 1),
                "is_top": idx == 0,
            }
            for idx, (symbol, value) in enumerate(rows)
        ]
        self._model.bulk_upsert(items, self._model.build_row, self._model.update_row)
        self._model.prune_missing({d["symbol"] for d in items})

        # resizeColumnsToContents() satır başına font-shaping (CoreText) çağırıyor
        # — 90sn'de bir periyodik olarak CPU'yu tıkıyordu (27 Ağu 2026, sample ile
        # ölçüldü). İlk dolduruluşta bir kez yapılması yeterli.
        if not self._resized_once:
            self._view.resizeColumnsToContents()
            self._resized_once = True
