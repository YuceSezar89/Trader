"""
TFAlignmentPanel — TF Hizalanma + Erken Ayrışma sisteminin süresiz izlenen
adaylarını gösterir (bkz. signals/tf_alignment_gate.py, 20 Tem 2026 dinamik-
eşik mimarisi).

Long ve Short ayrı tablolarda (kendi popülasyonları içinde percentile'a göre
sıralanıyor, bkz. backend). Her satırda "Aç" butonu — ManualTradeDialog'u
sembol/yön/TF/güncel fiyat önceden doldurulmuş açar.

28 Ağu 2026: QTableWidget → Model/View göçü (Faz 4, bkz. proje hafızası
"masaüstü panel mimari denetimi"). Gömülü "Aç" butonu artık setCellWidget
DEĞİL, ButtonColumnDelegate (bkz. desktop/widgets/button_column_delegate.py)
— sıralanabilir bir QSortFilterProxyModel altında setCellWidget/setIndexWidget
satır eşlemesi her sıralamada bozulurdu, ayrıca 26 Tem 2026'da bulunan "her
render'da yeni QPushButton = tüm QSS'in yeniden ayrıştırılması" CPU maliyeti
de bu şekilde ortadan kalkıyor (buton hiç widget olarak yaratılmıyor, sadece
çiziliyor).
"""

from PyQt6.QtCore import pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from desktop.models.tf_alignment_model import (
    COL_ACTION,
    COL_SYMBOL,
    TFAlignmentModel,
    TFAlignmentProxyModel,
)
from desktop.theme import COLORS
from desktop.widgets.button_column_delegate import ButtonColumnDelegate
from desktop.widgets.staleness import StalePanelMixin
from desktop.workers.tf_alignment_worker import TFAlignmentWorker

# tf_alignment_worker 10sn'de bir yayınlıyor (bkz. o worker'ın _UPDATE_SEC'i).
# 3 tur (30sn) hiç veri gelmezse worker/Redis bağlantısı kopmuş olabilir —
# panel eski veriyi sessizce göstermeye devam etmesin diye uyarı gösterilir
# (27 Ağu 2026, MOVR'daki "sessizce bayat veri" olayına karşı genel önlem).
_STALE_THRESHOLD_SEC = 30
_STALE_CHECK_INTERVAL_MS = 10_000


class _DirectionTableView(QTableView):
    """Long veya Short adaylarını gösteren tek bir tablo."""

    def __init__(self, direction: str, on_open, parent=None):
        super().__init__(parent)
        self._direction = direction
        self._on_open = on_open
        self._model = TFAlignmentModel(self)
        self._proxy = TFAlignmentProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self.setModel(self._proxy)
        self._resized_once = False

        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(False)
        self.setSortingEnabled(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(
            f"""
            QTableView {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: none;
                font-size: 12px;
            }}
            QHeaderView::section {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_muted']};
                border: none;
                padding: 4px;
                font-size: 11px;
            }}
            QTableView::item:selected {{
                background: {COLORS['bg_tertiary']};
            }}
            """
        )
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(COL_SYMBOL, QHeaderView.ResizeMode.Stretch)

        self._delegate = ButtonColumnDelegate("Aç")
        self._delegate.clicked.connect(self._on_action_clicked)
        self.setItemDelegateForColumn(COL_ACTION, self._delegate)

    def _on_action_clicked(self, proxy_index) -> None:
        source_index = self._proxy.mapToSource(proxy_index)
        row = self._model.row_at(source_index.row())
        if row is not None:
            self._on_open(row.raw)

    def render(self, rows: list, search_text: str) -> None:
        self._model.bulk_upsert(rows, self._model.build_row, self._model.update_row)
        self._model.prune_missing({r.get("symbol", "") for r in rows})
        # resizeColumnsToContents() satır başına font-shaping (CoreText) çağırıyor
        # — 10sn'de bir periyodik olarak CPU'yu tıkıyordu (27 Ağu 2026, sample ile
        # ölçüldü). İlk dolduruluşta bir kez yapılması yeterli.
        if not self._resized_once:
            self.resizeColumnsToContents()
            self._resized_once = True
        self._proxy.set_search(search_text)


class TFAlignmentPanel(QWidget, StalePanelMixin):
    def __init__(self, db_config: dict, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._db_config = db_config or {}
        self._worker = TFAlignmentWorker(redis_url, parent=self)
        self._search_text = ""
        self._last_rows: list = []
        self._setup_ui()
        self._connect_worker()
        self._worker.start()
        self._init_staleness(
            self._status,
            threshold_sec=_STALE_THRESHOLD_SEC,
            check_interval_ms=_STALE_CHECK_INTERVAL_MS,
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        title = QLabel("TF Hizalanma Adayları")
        title.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")

        self._status = QLabel("Yükleniyor…")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Ara…")
        self._search_box.setFixedHeight(24)
        self._search_box.setFixedWidth(110)
        self._search_box.setStyleSheet(
            f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 0 4px; font-size: 11px;"
        )
        self._search_box.textChanged.connect(self._on_search_changed)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setStyleSheet(
            f"color: {COLORS['accent']}; background: transparent; border: none; font-size: 14px;"
        )
        refresh_btn.clicked.connect(self._worker.refresh)

        top.addWidget(title)
        top.addStretch()
        top.addWidget(self._search_box)
        top.addWidget(self._status)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        tables_row = QHBoxLayout()
        tables_row.setSpacing(6)

        long_col = QVBoxLayout()
        long_label = QLabel("Long")
        long_label.setStyleSheet(f"color: {COLORS['green']}; font-weight: bold; font-size: 11px;")
        self._long_table = _DirectionTableView("Long", self._open_manual_trade)
        long_col.addWidget(long_label)
        long_col.addWidget(self._long_table)

        short_col = QVBoxLayout()
        short_label = QLabel("Short")
        short_label.setStyleSheet(f"color: {COLORS['red']}; font-weight: bold; font-size: 11px;")
        self._short_table = _DirectionTableView("Short", self._open_manual_trade)
        short_col.addWidget(short_label)
        short_col.addWidget(self._short_table)

        tables_row.addLayout(long_col)
        tables_row.addLayout(short_col)
        layout.addLayout(tables_row)

    def _connect_worker(self) -> None:
        self._worker.candidates_updated.connect(self._on_updated)
        self._worker.status_updated.connect(self._on_status)

    @pyqtSlot(object)
    def _on_updated(self, rows: list) -> None:
        self._mark_fresh()
        self._last_rows = rows
        self._render(rows)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        if not self._is_stale:
            self._status.setText(msg)

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip().upper()
        self._render(self._last_rows)

    def _render(self, rows: list) -> None:
        long_rows = [r for r in rows if r.get("signal_type") == "Long"]
        short_rows = [r for r in rows if r.get("signal_type") == "Short"]
        self._long_table.render(long_rows, self._search_text)
        self._short_table.render(short_rows, self._search_text)

    def _open_manual_trade(self, row_data: dict) -> None:
        from desktop.dialogs.manual_trade_dialog import ManualTradeDialog

        dlg = ManualTradeDialog(
            self._db_config,
            self._redis_url,
            parent=self,
            prefill_symbol=row_data.get("symbol", ""),
            prefill_direction=row_data.get("signal_type", "Long"),
            prefill_interval=row_data.get("interval", ""),
        )
        dlg.exec()
