"""
RankingPanel — tüm coinleri VPMV güç skoruna göre sıralayan panel.

Kolonlar: Rank | Sembol | 5m | 15m | 1h | Birleşik | TF Uyum | VS BTC

28 Ağu 2026: QTableWidget → Model/View göçü (Faz 3, bkz. proje hafızası
"masaüstü panel mimari denetimi"). Eski elle yönetilen get-or-create/satır
haritası/sıralama-bayatlığı kodu ve elle yazılmış staleness-timer kaldırıldı
— RankingModel/RankingProxyModel (desktop/models/ranking_model.py) ve
StalePanelMixin (desktop/widgets/staleness.py) bunları tek doğru şekilde
çözüyor.
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

from desktop.models.ranking_model import COL_SYMBOL, RankingModel, RankingProxyModel
from desktop.theme import COLORS
from desktop.widgets.staleness import StalePanelMixin
from desktop.workers.ranking_worker import RankingWorker


class RankingPanel(QWidget, StalePanelMixin):
    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._worker = RankingWorker(redis_url, parent=self)
        self._pine_filter = False
        self._prev_ranks: dict[str, int] = {}
        self._resized_once = False
        self._model = RankingModel(self)
        self._proxy = RankingProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._setup_ui()
        self._connect_worker()
        self._worker.start()
        self._init_staleness(self._status)

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Üst bar
        top = QHBoxLayout()
        title = QLabel("Güç Sıralaması")
        title.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")

        self._status = QLabel("Yükleniyor…")
        self._status.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self._pine_btn = QPushButton("Pine 20")
        self._pine_btn.setCheckable(True)
        self._pine_btn.setFixedHeight(24)
        self._pine_btn.setFixedWidth(60)
        self._pine_btn.setStyleSheet(self._filter_btn_style(False))
        self._pine_btn.clicked.connect(self._on_pine_toggled)

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
        top.addWidget(self._pine_btn)
        top.addWidget(self._status)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        # Tablo
        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setAlternatingRowColors(False)
        self._view.setSortingEnabled(True)
        self._view.setShowGrid(False)
        self._view.verticalHeader().setVisible(False)
        self._view.setStyleSheet(
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

        hh = self._view.horizontalHeader()
        # ResizeToContents sürekli modda HER veri değişikliğinde tüm sütunu yeniden
        # ölçüyor (O(satır) maliyet × N güncelleme = O(satır²)) — 550 sembolle bu, ana
        # thread'i kilitleyip panel kasmasına yol açıyordu. Interactive + tek seferlik
        # resizeColumnsToContents() aynı görünümü verir, sürekli yeniden ölçüm olmadan.
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(COL_SYMBOL, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self._view)

    @staticmethod
    def _filter_btn_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background: {COLORS['accent']}; color: #fff; "
                f"border: none; border-radius: 3px; font-size: 10px; }}"
            )
        return (
            f"QPushButton {{ background: {COLORS['bg_tertiary']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px; font-size: 10px; }}"
        )

    def _on_pine_toggled(self, checked: bool) -> None:
        self._pine_filter = checked
        self._pine_btn.setStyleSheet(self._filter_btn_style(checked))
        self._proxy.set_pine_filter(checked)

    def _on_search_changed(self, text: str) -> None:
        self._proxy.set_search(text)

    def _connect_worker(self) -> None:
        self._worker.ranking_updated.connect(self._on_updated)
        self._worker.status_updated.connect(self._on_status)

    # ------------------------------------------------------------------
    @pyqtSlot(object)
    def _on_updated(self, result: list) -> None:
        self._mark_fresh()

        rank_deltas: dict[str, int] = {}
        for row_data in result:
            sym = row_data["symbol"]
            if sym in self._prev_ranks:
                rank_deltas[sym] = self._prev_ranks[sym] - row_data["rank"]
        self._prev_ranks = {r["symbol"]: r["rank"] for r in result}

        items = [{**r, "rank_delta": rank_deltas.get(r["symbol"], 0)} for r in result]
        self._model.bulk_upsert(items, self._model.build_row, self._model.update_row)
        self._model.prune_missing({r["symbol"] for r in result})

        # resizeColumnsToContents() satır başına font-shaping (CoreText) çağırıyor
        # — 550 satır × 12 sütunda birkaç saniye sürüp CPU'yu periyodik olarak
        # tıkıyordu (27 Ağu 2026, sample ile ölçüldü). İlk dolduruluşta bir kez
        # yapılması yeterli; sütunlar zaten Interactive/Stretch, kullanıcı
        # istediğinde elle genişletebilir.
        if not self._resized_once:
            self._view.resizeColumnsToContents()
            self._resized_once = True

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status.setText(msg)

    def closeEvent(self, event) -> None:
        self._worker.stop()
