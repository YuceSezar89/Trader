"""IdKeyedTableModel — 27-28 Ağu 2026'da 8 QTableWidget panelinde ayrı ayrı
elle yamanan "get-or-create satır + iki render arası sıralanırsa harita
bayatlar" sorununun TEK, doğru çözümü (28 Ağu 2026, Faz 0).

QTableWidget panelleri periyodik bir tam-snapshot (`render(sonuc_listesi)`)
alıp her satırı id/sembole göre bulup güncelliyor, yoksa ekliyordu — bu
mantığı doğru yazmak (silme sonrası satır kayması, sıralama sonrası harita
bayatlığı) her panelde ayrı ayrı hataya açıktı. `QAbstractTableModel` +
`QSortFilterProxyModel` bunu zaten bedava çözer: model satırların GERÇEK
sırasını (giriş sırası) tutar, proxy görünürdeki sıralamayı yönetir, satır
haritası hiç "bayatlamaz" çünkü zaten yok — id→index eşlemesi tek yerde
(`_id_index`) ve sadece burada güncellenir.

Alt sınıflar şunları override eder:
    COLUMNS: sütun başlıkları listesi
    _display(row, col), _foreground(row, col), _background(row, col),
    _tooltip(row, col) — hücre içeriği/görünümü
    _id_of_item(item), _id_of_row(row) — ham veriden / satır nesnesinden id
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt


class IdKeyedTableModel(QAbstractTableModel):
    COLUMNS: list[str] = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[Any] = []
        self._id_index: dict[Any, int] = {}

    # ── QAbstractTableModel zorunlu arayüz ──────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return self.COLUMNS[section]
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignCenter)
        # Diğer roller (ör. ToolTipRole) için Qt'nin `setHeaderData()` ile
        # ayarlanmış varsayılan önbelleğine düş — burada override edilmeyen
        # her rol sessizce None dönmesin.
        return super().headerData(section, orientation, role)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)
        if role == Qt.ItemDataRole.ForegroundRole:
            return self._foreground(row, col)
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._background(row, col)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(row, col)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return self._text_alignment(col)
        return None

    # ── Alt sınıfların override ettiği kancalar ─────────────────────────────

    def _display(self, row: Any, col: int) -> str:  # pragma: no cover - override edilir
        return ""

    def _foreground(self, row: Any, col: int):  # pragma: no cover - override edilir
        return None

    def _background(self, row: Any, col: int):  # pragma: no cover - override edilir
        return None

    def _tooltip(self, row: Any, col: int):  # pragma: no cover - override edilir
        return None

    @staticmethod
    def _text_alignment(col: int) -> int:
        return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

    def _id_of_item(self, item: Any) -> Any:  # pragma: no cover - override edilir
        raise NotImplementedError

    def _id_of_row(self, row: Any) -> Any:  # pragma: no cover - override edilir
        raise NotImplementedError

    # ── Satır erişimi ────────────────────────────────────────────────────────

    def row_at(self, view_row: int) -> Optional[Any]:
        if 0 <= view_row < len(self._rows):
            return self._rows[view_row]
        return None

    def row_by_id(self, row_id: Any) -> Optional[Any]:
        idx = self._id_index.get(row_id)
        return self._rows[idx] if idx is not None else None

    # ── get-or-create çekirdeği ──────────────────────────────────────────────

    def replace_all(self, items: Iterable[Any], build_row: Callable[[Any], Any]) -> None:
        """Tam reset — sadece ilk yükleme için. Periyodik render'da KULLANMA,
        seçim/scroll konumunu sıfırlar; onun yerine bulk_upsert+prune_missing
        kullan (get-or-create, satır kimliği korunur)."""
        self.beginResetModel()
        self._rows = [build_row(item) for item in items]
        self._id_index = {self._id_of_row(row): i for i, row in enumerate(self._rows)}
        self.endResetModel()

    def bulk_upsert(
        self,
        items: Iterable[Any],
        build_row: Callable[[Any], Any],
        update_row: Callable[[Any, Any], None],
    ) -> None:
        """Periyodik tam-snapshot render'ı için get-or-create: `items`'teki her
        öğe id'sine göre var olan satırda GÜNCELLENİR (update_row) ya da yeni
        satır olarak eklenir (build_row). Var olan satırlar için TEK bir
        dataChanged (dokunulan min..max satır aralığı, tüm sütunlar) yayınlanır
        — periyodik-tam-snapshot zaten "bu turda gerçekten güncellenen"
        satırlardır, 15 Ağu dataChanged kuralını ihlal etmez (o kural
        koşulsuz TÜM satırların bildirilmesini yasaklar, gerçekten
        güncellenenin bildirilmesini değil)."""
        min_idx: Optional[int] = None
        max_idx: Optional[int] = None
        for item in items:
            rid = self._id_of_item(item)
            existing_idx = self._id_index.get(rid)
            if existing_idx is None:
                new_row = build_row(item)
                insert_at = len(self._rows)
                self.beginInsertRows(QModelIndex(), insert_at, insert_at)
                self._rows.append(new_row)
                self._id_index[rid] = insert_at
                self.endInsertRows()
            else:
                update_row(self._rows[existing_idx], item)
                if min_idx is None or existing_idx < min_idx:
                    min_idx = existing_idx
                if max_idx is None or existing_idx > max_idx:
                    max_idx = existing_idx
        if min_idx is not None:
            tl = self.index(min_idx, 0)
            br = self.index(max_idx, len(self.COLUMNS) - 1)
            self.dataChanged.emit(tl, br, [])

    def prune_missing(self, present_ids: set) -> None:
        """`present_ids`'te olmayan (bu turun snapshot'ında artık gelmeyen)
        satırları kaldırır. bulk_upsert'ten SONRA çağrılır — periyodik
        tam-snapshot akışı için (ör. rank_panel)."""
        to_remove = [
            i for i, row in enumerate(self._rows) if self._id_of_row(row) not in present_ids
        ]
        self._remove_rows_by_indices(to_remove)

    def remove_by_ids(self, ids: set) -> None:
        """Belirli id'lere sahip satırları kaldırır — olay-güdümlü akışlar için
        (ör. deviso_panel'in on_signals_closed'ı), tam-snapshot gerekmez."""
        to_remove = [i for i, row in enumerate(self._rows) if self._id_of_row(row) in ids]
        self._remove_rows_by_indices(to_remove)

    def _remove_rows_by_indices(self, indices: list[int]) -> None:
        if not indices:
            return
        for idx in sorted(indices, reverse=True):
            self.beginRemoveRows(QModelIndex(), idx, idx)
            self._rows.pop(idx)
            self.endRemoveRows()
        self._id_index = {self._id_of_row(row): i for i, row in enumerate(self._rows)}
