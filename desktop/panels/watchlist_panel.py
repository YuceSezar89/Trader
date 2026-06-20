"""
WatchlistPanel — 200 sembol, canlı fiyat, değişim %, VPM skoru.

MarketWorker'dan gelen Qt sinyalleriyle her saniye güncellenir.
Sembol seçiminde `symbol_selected` sinyali yayınlanır.
"""

import json

import redis
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QHeaderView
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from desktop.models.watchlist_model import (
    COL_SYMBOL,
    WatchlistModel,
    WatchlistProxyModel,
)
from desktop.theme import COLORS


class WatchlistPanel(QWidget):
    """
    Watchlist paneli içeriği (QDockWidget'a yerleştirilen widget).

    Sinyaller:
        symbol_selected(str): Kullanıcı bir sembole tıkladığında.
    """

    symbol_selected = pyqtSignal(str)
    symbols_changed = pyqtSignal(list)   # yeni sembol listesi yüklendiğinde

    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url

        self._model = WatchlistModel(self)
        self._proxy = WatchlistProxyModel(self)
        self._proxy.setSourceModel(self._model)

        self._setup_ui()
        self._load_symbols()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(60_000)
        self._refresh_timer.timeout.connect(self._auto_refresh)
        self._refresh_timer.start()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Başlık satırı
        header = QHBoxLayout()
        title = QLabel("WATCHLIST")
        title.setObjectName("section_title")
        self._count_label = QLabel("0 sembol")
        self._count_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._count_label)
        root.addLayout(header)

        # Arama kutusu
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Sembol ara… (BTC, ETH…)")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._proxy.setFilterFixedString)
        search_row.addWidget(self._search)

        refresh_btn = QPushButton("↺")
        refresh_btn.setFixedWidth(32)
        refresh_btn.setToolTip("Sembolleri yenile")
        refresh_btn.clicked.connect(self._load_symbols)
        search_row.addWidget(refresh_btn)
        root.addLayout(search_row)

        # Tablo
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setWordWrap(False)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(True)

        # Satır yüksekliği
        self._table.verticalHeader().setDefaultSectionSize(24)

        # Varsayılan sıralama: değişim% azalan
        self._table.sortByColumn(2, Qt.SortOrder.DescendingOrder)

        self._table.clicked.connect(self._on_row_clicked)
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        root.addWidget(self._table)

        # Alt istatistik satırı
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; padding: 2px;")
        root.addWidget(self._stats_label)

    # ── Sembol Yükleme ────────────────────────────────────────────────────

    def _load_symbols(self) -> None:
        """Redis'ten sembol listesini ve mevcut ticker verisini yükler."""
        ticker_data: dict[str, dict] = {}
        symbols: list[str] = []
        try:
            r = redis.Redis.from_url(self._redis_url, decode_responses=True, socket_connect_timeout=2)
            keys = r.keys("ticker:*")
            if keys:
                pipe = r.pipeline()
                for k in keys:
                    pipe.get(k)
                values = pipe.execute()
                for k, v in zip(keys, values):
                    sym = k.split(":", 1)[1]
                    if v:
                        try:
                            ticker_data[sym] = json.loads(v)
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass
                symbols = sorted(ticker_data.keys())
            if not symbols:
                kline_keys = r.keys("live_kline_data:*:1m")
                symbols = sorted(
                    k.split(":")[1] for k in kline_keys if k.count(":") == 2
                )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

        self._model.load_symbols(symbols)

        for sym, d in ticker_data.items():
            self._model.on_price_updated(
                sym,
                float(d.get("price", 0)),
                float(d.get("change_pct", 0)),
                float(d.get("volume", 0)),
                float(d.get("funding_rate", 0)),
            )

        self._count_label.setText(f"{len(symbols)} sembol")
        self._update_stats()
        self.symbols_changed.emit(symbols)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(True)

    def _auto_refresh(self) -> None:
        """Her 60s'de bir sembol listesini ve fiyatları yeniler."""
        self._load_symbols()

    # ── Worker Sinyalleri ─────────────────────────────────────────────────

    @pyqtSlot(str, float, float, float, float)
    def on_price_updated(self, symbol: str, price: float, change_pct: float, volume: float = 0.0, funding_rate: float = 0.0) -> None:
        self._model.on_price_updated(symbol, price, change_pct, volume, funding_rate)

    # ── Seçim ─────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_row_clicked(self) -> None:
        self._update_stats()

    @pyqtSlot()
    def _on_row_double_clicked(self) -> None:
        symbol = self._selected_symbol()
        if symbol:
            self.symbol_selected.emit(symbol)

    def _selected_symbol(self) -> str:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return ""
        src_idx = self._proxy.mapToSource(indexes[0])
        return self._model.symbol_at(src_idx.row()) or ""

    def _update_stats(self) -> None:
        rows = self._model._rows  # noqa: SLF001
        up   = sum(1 for r in rows if r.change_pct > 0)
        down = sum(1 for r in rows if r.change_pct < 0)
        self._stats_label.setText(f"↑ {up}  ↓ {down}")

    # ── Public API ────────────────────────────────────────────────────────

    def model(self) -> WatchlistModel:
        return self._model
