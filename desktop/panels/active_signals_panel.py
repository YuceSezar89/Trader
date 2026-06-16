"""
ActiveSignalsPanel — aktif sinyaller tablosu.

Sütunlar: Sembol | Tip | TF | VPM | MTF | α | β | Z-Score% | P&L% | Süre
Filtreler: LONG / SHORT / Hepsi toggle + sembol arama
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QHeaderView
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from config import Config
from desktop.models.signals_model import SignalsModel, SignalsProxyModel
from desktop.theme import COLORS


class ActiveSignalsPanel(QWidget):
    """
    Aktif sinyaller paneli.

    Sinyaller:
        symbol_selected(str, str): Satıra tıklandığında (symbol, interval).
    """

    symbol_selected = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = SignalsModel(self)
        self._proxy = SignalsProxyModel(self)
        self._proxy.setSourceModel(self._model)

        self._setup_ui()
        self._setup_age_timer()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Başlık
        header = QHBoxLayout()
        title = QLabel("AKTİF SİNYALLER")
        title.setObjectName("section_title")
        self._count_label = QLabel("0 sinyal")
        self._count_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._count_label)
        root.addLayout(header)

        # Filtre çubuğu
        filter_row = QHBoxLayout()

        self._btn_all   = self._make_filter_btn("Hepsi",  True)
        self._btn_long  = self._make_filter_btn("LONG",   False)
        self._btn_short = self._make_filter_btn("SHORT",  False)

        self._btn_all.clicked.connect(lambda: self._set_type_filter(""))
        self._btn_long.clicked.connect(lambda: self._set_type_filter("LONG"))
        self._btn_short.clicked.connect(lambda: self._set_type_filter("SHORT"))

        filter_row.addWidget(self._btn_all)
        filter_row.addWidget(self._btn_long)
        filter_row.addWidget(self._btn_short)
        filter_row.addStretch()

        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["TF: Hepsi", "1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setFixedWidth(90)
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)
        filter_row.addWidget(self._tf_combo)

        self._ind_combo = QComboBox()
        self._ind_combo.addItems(["İnd: Hepsi", "RSI_Cross", "MA200_Cross", "Supertrend"])
        self._ind_combo.setFixedWidth(110)
        self._ind_combo.currentTextChanged.connect(self._on_indicator_changed)
        filter_row.addWidget(self._ind_combo)

        self._btn_st_filter = QPushButton("ST Filtre: Açık")
        self._btn_st_filter.setCheckable(True)
        self._btn_st_filter.setChecked(Config.ST_FILTER_ENABLED)
        self._btn_st_filter.setFixedHeight(24)
        self._btn_st_filter.setFixedWidth(110)
        self._btn_st_filter.setStyleSheet(self._st_filter_style(Config.ST_FILTER_ENABLED))
        self._btn_st_filter.clicked.connect(self._on_st_filter_toggled)
        filter_row.addWidget(self._btn_st_filter)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Sembol ara…")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedWidth(130)
        self._search.textChanged.connect(self._proxy.setFilterFixedString)
        filter_row.addWidget(self._search)
        root.addLayout(filter_row)

        # Tablo
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setDefaultSectionSize(26)

        # Varsayılan sıralama: VPM azalan
        self._table.sortByColumn(4, Qt.SortOrder.DescendingOrder)

        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._table.clicked.connect(self._on_row_clicked)
        self._table.doubleClicked.connect(self._on_row_double_clicked)
        root.addWidget(self._table)

        # İstatistik satırı
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; padding: 2px;"
        )
        root.addWidget(self._stats_label)

        # Detay şeridi — seçili sinyal metrikleri
        self._detail_bar = QLabel("")
        self._detail_bar.setStyleSheet(
            f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
            f"font-size: 11px; font-family: monospace; padding: 4px 8px; "
            f"border-top: 1px solid {COLORS['border']};"
        )
        self._detail_bar.setWordWrap(False)
        self._detail_bar.hide()
        root.addWidget(self._detail_bar)

    def _make_filter_btn(self, text: str, active: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setChecked(active)
        btn.setFixedHeight(24)
        btn.setStyleSheet(self._filter_btn_style(active))
        return btn

    @staticmethod
    def _filter_btn_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; color: #fff; "
                f"border: none; border-radius: 3px; font-size: 11px; padding: 0 10px; }}"
            )
        return (
            f"QPushButton {{ background-color: {COLORS['bg_tertiary']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px; font-size: 11px; padding: 0 10px; }}"
            f"QPushButton:hover {{ color: {COLORS['text_primary']}; }}"
        )

    @staticmethod
    def _st_filter_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; color: #fff; "
                f"border: none; border-radius: 3px; font-size: 10px; padding: 0 6px; }}"
            )
        return (
            f"QPushButton {{ background-color: {COLORS['bg_tertiary']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px; font-size: 10px; padding: 0 6px; }}"
            f"QPushButton:hover {{ color: {COLORS['text_primary']}; }}"
        )

    def _on_st_filter_toggled(self) -> None:
        enabled = self._btn_st_filter.isChecked()
        Config.ST_FILTER_ENABLED = enabled
        self._proxy.set_st_filter(enabled)
        self._btn_st_filter.setText("ST Filtre: Açık" if enabled else "ST Filtre: Kapalı")
        self._btn_st_filter.setStyleSheet(self._st_filter_style(enabled))

    def _on_tf_changed(self, text: str) -> None:
        self._proxy.set_tf_filter("" if text.startswith("TF:") else text)
        self._update_stats()

    def _on_indicator_changed(self, text: str) -> None:
        self._proxy.set_indicator_filter("" if text.startswith("İnd:") else text)
        self._update_stats()

    # ── Age timer — Süre kolonunu dakikada bir yenile ─────────────────────────

    def _setup_age_timer(self) -> None:
        timer = QTimer(self)
        timer.setInterval(60_000)
        timer.timeout.connect(self._refresh_age_column)
        timer.start()

    def _refresh_age_column(self) -> None:
        if self._model.rowCount() == 0:
            return
        tl = self._model.index(0, 10)
        br = self._model.index(self._model.rowCount() - 1, 10)
        self._model.dataChanged.emit(tl, br, [0])  # DisplayRole = 0

    # ── Filtre ────────────────────────────────────────────────────────────────

    def _set_type_filter(self, type_filter: str) -> None:
        self._proxy.set_type_filter(type_filter)
        for btn, tf in [
            (self._btn_all, ""),
            (self._btn_long, "LONG"),
            (self._btn_short, "SHORT"),
        ]:
            active = (tf == type_filter)
            btn.setChecked(active)
            btn.setStyleSheet(self._filter_btn_style(active))
        self._update_stats()

    # ── Worker slot'ları ──────────────────────────────────────────────────────

    @pyqtSlot(list)
    def on_signals_loaded(self, signals: list) -> None:
        self._model.load_signals(signals)
        self._update_stats()
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    @pyqtSlot(dict)
    def on_new_signal(self, signal: dict) -> None:
        self._model.add_or_update(signal)
        self._update_stats()

    @pyqtSlot(str, float, float)
    def on_price_updated(self, symbol: str, price: float, change_pct: float) -> None:
        self._model.on_price_updated(symbol, price, change_pct)

    # ── Seçim ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_row_clicked(self) -> None:
        self._update_stats()
        self._update_detail_bar()

    def _update_detail_bar(self) -> None:
        row = self._selected_row()
        if row is None:
            self._detail_bar.hide()
            return

        def _r(v, fmt=".2f"): return f"{v:{fmt}}" if v is not None else "—"
        st = ("✓" if row.st_confirmed else "✗") if row.st_confirmed is not None else "—"
        mtf = f"{int(row.mtf)}" if row.mtf is not None else "—"
        pnl = (f"{row.pnl_pct:+.2f}%" if row.pnl_pct is not None else "—")

        text = (
            f"  α {_r(row.alpha, '+.4f')}  "
            f"β {_r(row.beta)}  │  "
            f"Sharpe {_r(row.sharpe)}  "
            f"Sortino {_r(row.sortino)}  "
            f"Calmar {_r(row.calmar)}  "
            f"Omega {_r(row.omega)}  │  "
            f"VPMV {_r(row.vpm, '.1f')}  "
            f"MTF {mtf}  "
            f"ST {st}  │  "
            f"P&L {pnl}"
        )
        self._detail_bar.setText(text)
        self._detail_bar.show()

    @pyqtSlot()
    def _on_row_double_clicked(self) -> None:
        row = self._selected_row()
        if row:
            self.symbol_selected.emit(row.symbol, row.interval)

    def _selected_row(self):
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        src_idx = self._proxy.mapToSource(indexes[0])
        return self._model.signal_at(src_idx.row())

    # ── İstatistikler ─────────────────────────────────────────────────────────

    def _update_stats(self) -> None:
        rows = self._model._rows  # noqa: SLF001
        total = len(rows)
        longs  = sum(1 for r in rows if r.signal_type == "LONG")
        shorts = total - longs
        pos_pnl = sum(1 for r in rows if r.pnl_pct is not None and r.pnl_pct > 0)
        self._count_label.setText(f"{total} sinyal")
        self._stats_label.setText(
            f"↑ {longs} LONG  ↓ {shorts} SHORT  ✓ {pos_pnl} pozitif"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def model(self) -> SignalsModel:
        return self._model
