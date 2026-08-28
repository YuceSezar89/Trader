"""
DevisoPanel — aktif sinyalleri devisso_score'a göre sıralayan tablo.

Kolonlar: # | Sembol | TF | Yön | Score | Δ | Ratio | Zaman

28 Ağu 2026: QTableWidget → Model/View göçü (Faz 2, bkz. proje hafızası
"masaüstü panel mimari denetimi"). Eski elle yönetilen get-or-create/id-satır
haritası/sıralama-bayatlığı kodu ve Python tarafında elle filtreleme kaldırıldı
— DevisoModel tüm sinyalleri tutar, DevisoProxyModel arama/TF/yön filtrelerini
ve sıralamayı uygular (bkz. desktop/models/deviso_model.py).
"""

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from desktop.models.deviso_model import (
    COL_SCORE,
    COL_SYMBOL,
    COLUMN_TOOLTIPS,
    TF_OPTIONS,
    DevisoModel,
    DevisoProxyModel,
)
from desktop.theme import COLORS


class DevisoPanel(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = DevisoModel(self)
        self._proxy = DevisoProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._resized_once = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Sembol ara...")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedHeight(26)
        self._search.setStyleSheet(
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_primary']};"
            " border: 1px solid #444; border-radius: 4px; padding: 0 6px; font-size: 12px;"
        )
        self._search.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self._search, stretch=2)

        self._tf_combo = QComboBox()
        self._tf_combo.addItems(TF_OPTIONS)
        self._tf_combo.setFixedHeight(26)
        self._tf_combo.setStyleSheet(
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_primary']};"
            " border: 1px solid #444; border-radius: 4px; font-size: 12px;"
        )
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)
        toolbar.addWidget(self._tf_combo)

        for label in ("Tümü", "Long", "Short"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(label == "Tümü")
            btn.setFixedHeight(26)
            btn.setStyleSheet(self._btn_style(label == "Tümü"))
            btn.clicked.connect(lambda checked, l=label: self._on_dir_btn(l))
            toolbar.addWidget(btn)
            setattr(self, f"_btn_{label.lower()}", btn)

        layout.addLayout(toolbar)

        self._status = QLabel("Devisso Sıralama")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        layout.addWidget(self._status)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().setVisible(False)
        self._view.horizontalHeader().setStretchLastSection(True)
        self._view.horizontalHeader().setSectionResizeMode(
            COL_SYMBOL, QHeaderView.ResizeMode.ResizeToContents
        )
        self._view.setSortingEnabled(True)
        self._view.setStyleSheet(
            """
            QTableView { font-size: 12px; }
            QHeaderView::section { font-size: 11px; font-weight: bold; }
            QHeaderView::section:hover { background: #2a2a2a; }
        """
        )
        bold = QFont()
        bold.setBold(True)
        self._view.horizontalHeader().setFont(bold)
        self._install_header_tooltips()

        layout.addWidget(self._view)

        self._proxy.sort(COL_SCORE, Qt.SortOrder.DescendingOrder)
        self._model.dataChanged.connect(lambda *_: self._update_status())
        self._model.rowsInserted.connect(lambda *_: self._update_status())
        self._model.rowsRemoved.connect(lambda *_: self._update_status())
        self._proxy.layoutChanged.connect(lambda *_: self._update_status())

    def _install_header_tooltips(self) -> None:
        for col, tip in enumerate(COLUMN_TOOLTIPS):
            self._proxy.setHeaderData(
                col, Qt.Orientation.Horizontal, tip, Qt.ItemDataRole.ToolTipRole
            )

    def _btn_style(self, active: bool) -> str:
        if active:
            return (
                f"background: {COLORS.get('accent', '#1a6bcc')}; color: white;"
                " border-radius: 4px; font-size: 11px; padding: 0 8px;"
            )
        return (
            f"background: {COLORS['bg_secondary']}; color: {COLORS['text_muted']};"
            " border: 1px solid #444; border-radius: 4px; font-size: 11px; padding: 0 8px;"
        )

    def _on_dir_btn(self, label: str) -> None:
        for name in ("tümü", "long", "short"):
            btn = getattr(self, f"_btn_{name}", None)
            if btn:
                active = name == label.lower()
                btn.setChecked(active)
                btn.setStyleSheet(self._btn_style(active))
        self._proxy.set_direction(label)
        self._update_status()

    def _on_search_changed(self, text: str) -> None:
        self._proxy.set_search(text)
        self._update_status()

    def _on_tf_changed(self, tf: str) -> None:
        self._proxy.set_tf(tf)
        self._update_status()

    # ── Veri slot'ları ───────────────────────────────────────────────────────

    @pyqtSlot(list)
    def on_signals_loaded(self, rows: List[Dict[str, Any]]) -> None:
        self._model.replace_all(rows, self._model.build_row)
        self._update_status()
        self._resize_once()

    @pyqtSlot(dict)
    def on_new_signal(self, row: Dict[str, Any]) -> None:
        if row.get("status") == "active":
            self._model.bulk_upsert([row], self._model.build_row, self._model.update_row)
        else:
            self._model.remove_by_ids({row["id"]})
        self._update_status()

    @pyqtSlot(list)
    def on_signals_closed(self, ids: List[int]) -> None:
        self._model.remove_by_ids(set(ids))
        self._update_status()

    def _resize_once(self) -> None:
        if self._resized_once:
            return
        self._resized_once = True
        # resizeColumnToContents() satır başına font-shaping (CoreText) çağırıyor
        # — event-bazlı sık tetiklenen bu panelde periyodik olarak CPU'yu
        # tıkıyordu (27 Ağu 2026, sample ile ölçüldü). İlk dolduruluşta bir kez
        # yapılması yeterli.
        self._view.resizeColumnsToContents()

    def _update_status(self) -> None:
        total = sum(
            1 for r in range(self._model.rowCount()) if self._model.row_at(r).status == "active"
        )
        shown = self._proxy.rowCount()
        with_score = 0
        for r in range(shown):
            src_row = self._model.row_at(self._proxy.mapToSource(self._proxy.index(r, 0)).row())
            if src_row is not None and src_row.devisso_score is not None:
                with_score += 1
        if shown < total:
            self._status.setText(
                f"{shown}/{total} sinyal gösteriliyor | {with_score} devisso hesaplı"
            )
        else:
            self._status.setText(f"{total} aktif sinyal | {with_score} devisso hesaplı")
