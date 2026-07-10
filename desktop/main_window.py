"""
MainWindow — TRader Desktop Terminal ana penceresi.
Dockable panel layout, menü/toolbar/status bar barındırır.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QToolBar,
    QWidget,
)

from desktop.panels.active_signals_panel import ActiveSignalsPanel
from desktop.panels.chart_panel import ChartPanel
from desktop.panels.divergence_panel import DivergencePanel
from desktop.panels.paper_trade_panel import PaperTradePanel
from desktop.panels.vpmv_divergence_panel import VpmvDivergencePanel
from desktop.panels.deviso_panel import DevisoPanel
from desktop.panels.ranking_panel import RankingPanel
from desktop.panels.watchlist_panel import WatchlistPanel
from desktop.theme import COLORS
from desktop.widgets.log_panel import LogPanel
from desktop.workers.divergence_worker import DivergenceWorker
from desktop.workers.health_worker import HealthWorker, ServiceStatus
from desktop.workers.vpmv_worker import VpmvWorker
from desktop.workers.market_worker import MarketWorker
from desktop.workers.signal_worker import SignalWorker


class StatusIndicator(QLabel):
    """Küçük renkli servis durumu etiketi."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        self.set_unknown()

    def set_ok(self) -> None:
        self.setText(f"● {self._name}")
        self.setStyleSheet(f"color: {COLORS['green']}; font-size: 11px;")

    def set_error(self) -> None:
        self.setText(f"● {self._name}")
        self.setStyleSheet(f"color: {COLORS['red']}; font-size: 11px;")

    def set_unknown(self) -> None:
        self.setText(f"● {self._name}")
        self.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")


class MainWindow(QMainWindow):
    """
    TRader Desktop Terminal ana penceresi.

    Panel yerleşimi QDockWidget ile yönetilir; kullanıcı panelleri
    sürükleyip bırakabilir, tabifikasyonu açabilir veya yüzdürebilir.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._workers: list = []

        self._setup_window()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_status_bar()
        self._setup_placeholder_panels()
        self._restore_layout()
        self._start_workers()

    # ── Pencere ───────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("TRader Terminal")
        self.setMinimumSize(1280, 720)
        self.resize(1600, 900)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

    # ── Menü Bar ──────────────────────────────────────────────────────────

    def _setup_menu(self) -> None:
        mb = self.menuBar()

        # Dosya
        file_menu = mb.addMenu("Dosya")
        act_save_layout = QAction("Layout Kaydet", self)
        act_save_layout.setShortcut(QKeySequence("Ctrl+S"))
        act_save_layout.triggered.connect(self._save_layout)
        file_menu.addAction(act_save_layout)

        act_reset_layout = QAction("Layout Sıfırla", self)
        act_reset_layout.triggered.connect(self._reset_layout)
        file_menu.addAction(act_reset_layout)

        file_menu.addSeparator()
        act_exit = QAction("Çıkış", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Görünüm
        self._view_menu = mb.addMenu("Görünüm")

        # Araçlar
        tools_menu = mb.addMenu("Araçlar")
        act_settings = QAction("Ayarlar", self)
        act_settings.setShortcut(QKeySequence("Ctrl+,"))
        tools_menu.addAction(act_settings)

    def _add_panel_toggle(self, dock: QDockWidget) -> None:
        """Görünüm menüsüne panel aç/kapat eylemi ekler."""
        action = dock.toggleViewAction()
        self._view_menu.addAction(action)

    # ── Toolbar ───────────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        tb = QToolBar("Ana Araç Çubuğu", self)
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Sembol arama
        self._symbol_search = QLineEdit()
        self._symbol_search.setPlaceholderText("Sembol ara…")
        self._symbol_search.setFixedWidth(140)
        self._symbol_search.setObjectName("symbol_search")
        self._symbol_search.textChanged.connect(self._on_toolbar_search)
        tb.addWidget(self._symbol_search)

        tb.addSeparator()

        # Timeframe seçici
        tf_label = QLabel("TF:")
        tf_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 0 4px;")
        tb.addWidget(tf_label)

        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setCurrentText("1h")
        self._tf_combo.setFixedWidth(70)
        self._tf_combo.currentTextChanged.connect(self._on_toolbar_tf_changed)
        tb.addWidget(self._tf_combo)

        tb.addSeparator()

        # Servis başlat/durdur butonu
        self._service_btn = QPushButton("▶  Başlat")
        self._service_btn.setFixedWidth(110)
        self._service_btn.clicked.connect(self._toggle_service)
        tb.addWidget(self._service_btn)
        self._update_service_btn()

        self._service_poll = QTimer(self)
        self._service_poll.timeout.connect(self._update_service_btn)
        self._service_poll.start(10_000)

        tb.addSeparator()

        # Bağlantı durumu butonu (görsel amaçlı)
        self._connect_btn = tb.addAction("⬤  Bağlanıyor…")
        self._connect_btn.setEnabled(False)

    # ── Status Bar ────────────────────────────────────────────────────────

    def _setup_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setObjectName("status_bar")

        self._status_redis = StatusIndicator("Redis")
        self._status_db = StatusIndicator("DB")
        self._status_symbols = QLabel("0 sembol")
        self._status_symbols.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self._status_signals = QLabel("0 aktif sinyal")
        self._status_signals.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self._status_time = QLabel("")
        self._status_time.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        sep = lambda: QLabel("  │  ")  # noqa: E731
        for sep in [
            self._status_redis, sep(),
            self._status_db, sep(),
            self._status_symbols, sep(),
            self._status_signals,
        ]:
            sb.addWidget(sep)

        sb.addPermanentWidget(self._status_time)

        # Saati güncelle
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    # ── Placeholder Paneller (Faz 1) ──────────────────────────────────────

    def _setup_placeholder_panels(self) -> None:
        """
        Faz 1: Gerçek içerik yokken placeholder paneller kur.
        Sonraki fazlarda her panel kendi modülüne taşınır.
        """
        self._docks: dict[str, QDockWidget] = {}

        redis_url = self._config.get("redis_url", "redis://localhost:6379/0")
        db_cfg = {
            "host":     self._config.get("db_host", "localhost"),
            "port":     self._config.get("db_port", 5432),
            "dbname":   self._config.get("db_name", "trader_panel"),
            "user":     self._config.get("db_user", "yusuf"),
            "password": self._config.get("db_password", ""),
        }

        # ── Watchlist (sol, gerçek panel) ─────────────────────────────────
        self._watchlist_panel = WatchlistPanel(self)
        watchlist_dock = QDockWidget("Watchlist", self)
        watchlist_dock.setObjectName("dock_watchlist")
        watchlist_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        watchlist_dock.setWidget(self._watchlist_panel)
        watchlist_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, watchlist_dock)
        self._add_panel_toggle(watchlist_dock)
        self._docks["watchlist"] = watchlist_dock

        # ── Ana Grafik (sağ, gerçek panel) ────────────────────────────────
        self._chart_panel = ChartPanel(redis_url, db_cfg, self)
        self._watchlist_panel.symbol_selected.connect(self._chart_panel.load_symbol)
        chart_dock = QDockWidget("Ana Grafik", self)
        chart_dock.setObjectName("dock_chart")
        chart_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        chart_dock.setWidget(self._chart_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, chart_dock)
        self._add_panel_toggle(chart_dock)
        self._docks["chart"] = chart_dock

        # ── Aktif Sinyaller (alt, gerçek panel) ───────────────────────────
        self._active_signals_panel = ActiveSignalsPanel(self)
        self._active_signals_panel.symbol_selected.connect(self._chart_panel.load_symbol)
        self._active_signals_panel.signal_data_selected.connect(self._chart_panel.set_signal_marker)
        active_sig_dock = QDockWidget("Aktif Sinyaller", self)
        active_sig_dock.setObjectName("dock_active_sig")
        active_sig_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        active_sig_dock.setWidget(self._active_signals_panel)
        active_sig_dock.setMinimumHeight(180)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, active_sig_dock)
        self._add_panel_toggle(active_sig_dock)
        self._docks["active_sig"] = active_sig_dock

        # ── Ayrışma paneli (sağ tabified) ─────────────────────────────────
        self._divergence_panel = DivergencePanel(self)
        divergence_dock = QDockWidget("Ayrışma", self)
        divergence_dock.setObjectName("dock_divergence")
        divergence_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        divergence_dock.setWidget(self._divergence_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, divergence_dock)
        self.tabifyDockWidget(chart_dock, divergence_dock)
        self._add_panel_toggle(divergence_dock)
        self._docks["divergence"] = divergence_dock

        # ── VPMV Ayrışma paneli (sağ tabified) ────────────────────────────
        self._vpmv_panel = VpmvDivergencePanel(self)
        vpmv_dock = QDockWidget("VPMV Ayrışma", self)
        vpmv_dock.setObjectName("dock_vpmv")
        vpmv_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        vpmv_dock.setWidget(self._vpmv_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, vpmv_dock)
        self.tabifyDockWidget(chart_dock, vpmv_dock)
        self._add_panel_toggle(vpmv_dock)
        self._docks["vpmv"] = vpmv_dock

        # ── Güç Sıralaması paneli (sağ tabified) ──────────────────────────
        self._ranking_panel = RankingPanel(redis_url, self)
        ranking_dock = QDockWidget("Güç Sıralaması", self)
        ranking_dock.setObjectName("dock_ranking")
        ranking_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        ranking_dock.setWidget(self._ranking_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ranking_dock)
        self.tabifyDockWidget(chart_dock, ranking_dock)
        self._add_panel_toggle(ranking_dock)
        self._docks["ranking"] = ranking_dock

        # ── Devisso Sıralama paneli (sağ tabified) ────────────────────────
        self._deviso_panel = DevisoPanel(self)
        deviso_dock = QDockWidget("Devisso Sırala", self)
        deviso_dock.setObjectName("dock_deviso")
        deviso_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        deviso_dock.setWidget(self._deviso_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, deviso_dock)
        self.tabifyDockWidget(chart_dock, deviso_dock)
        self._add_panel_toggle(deviso_dock)
        self._docks["deviso"] = deviso_dock

        # ── Paper Trade paneli (alt, tabified) ────────────────────────────
        self._paper_trade_panel = PaperTradePanel(db_cfg, redis_url, self)
        self._paper_trade_panel.symbol_selected.connect(self._chart_panel.load_symbol)
        paper_dock = QDockWidget("★ Paper Trade", self)
        paper_dock.setObjectName("dock_paper_trade")
        paper_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        paper_dock.setWidget(self._paper_trade_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, paper_dock)
        self.tabifyDockWidget(active_sig_dock, paper_dock)
        self._add_panel_toggle(paper_dock)
        self._docks["paper_trade"] = paper_dock

        # ── Placeholder paneller (sağ tabified + alt) ─────────────────────
        placeholder_panels = [
            ("signal_feed", "Sinyal Feed",      Qt.DockWidgetArea.RightDockWidgetArea),
            ("backtest",    "Backtest Stüdyo",  Qt.DockWidgetArea.BottomDockWidgetArea),
        ]

        for name, title, area in placeholder_panels:
            dock = self._make_placeholder_dock(title, name)
            self.addDockWidget(area, dock)
            self._add_panel_toggle(dock)
            self._docks[name] = dock

            if area == Qt.DockWidgetArea.RightDockWidgetArea:
                self.tabifyDockWidget(chart_dock, dock)
            if area == Qt.DockWidgetArea.BottomDockWidgetArea:
                self.tabifyDockWidget(active_sig_dock, dock)

        # ── Sistem Durumu → Log paneli ────────────────────────────────────
        self._log_panel = LogPanel(self)
        log_dock = QDockWidget("Sistem Durumu", self)
        log_dock.setObjectName("dock_system")
        log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        log_dock.setWidget(self._log_panel)
        log_dock.setMinimumHeight(120)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        self.tabifyDockWidget(active_sig_dock, log_dock)
        self._add_panel_toggle(log_dock)
        self._docks["system"] = log_dock

        chart_dock.raise_()
        active_sig_dock.raise_()

    def _make_placeholder_dock(self, title: str, name: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{name}")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        placeholder = QWidget()
        placeholder.setStyleSheet(f"background-color: {COLORS['bg_secondary']};")
        label = QLabel(f"{title}\n(Yakında)", placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
        label.setGeometry(0, 0, 300, 200)

        dock.setWidget(placeholder)
        return dock

    # ── Workers ───────────────────────────────────────────────────────────

    def _start_workers(self) -> None:
        redis_url = self._config.get("redis_url", "redis://localhost:6379/0")
        db_cfg = {
            "host":     self._config.get("db_host", "localhost"),
            "port":     self._config.get("db_port", 5432),
            "dbname":   self._config.get("db_name", "trader_panel"),
            "user":     self._config.get("db_user", "yusuf"),
            "password": self._config.get("db_password", ""),
        }

        # Health worker
        self._health_worker = HealthWorker(redis_url, db_cfg, parent=self)
        self._health_worker.health_updated.connect(self._on_health_updated)
        self._health_worker.start()
        self._workers.append(self._health_worker)

        # Market worker — watchlist + grafik paneline bağlı
        self._market_worker = MarketWorker(redis_url, parent=self)
        self._market_worker.connection_changed.connect(self._on_market_connection)
        self._market_worker.price_updated.connect(self._watchlist_panel.on_price_updated)
        self._market_worker.price_updated.connect(self._on_price_updated)
        self._market_worker.prices_updated.connect(self._paper_trade_panel.on_prices_updated)
        self._market_worker.klines_updated.connect(self._chart_panel.on_klines_updated)
        self._market_worker.symbols_discovered.connect(self._watchlist_panel.on_symbols_discovered)
        self._watchlist_panel.refresh_requested.connect(self._market_worker.request_symbol_refresh)
        self._chart_panel.symbol_changed.connect(self._market_worker.set_chart_watch)
        self._market_worker.set_chart_watch(
            self._chart_panel.current_symbol(), self._chart_panel.current_tf()
        )
        self._market_worker.start()
        self._workers.append(self._market_worker)

        # Signal worker — watchlist + aktif sinyaller paneline bağlı
        self._signal_worker = SignalWorker(
            db_cfg, redis_url=self._config.get("redis_url", ""), parent=self
        )
        self._signal_worker.connection_changed.connect(self._on_signal_connection)
        self._signal_worker.signals_loaded.connect(self._active_signals_panel.on_signals_loaded)
        self._signal_worker.new_signal.connect(self._active_signals_panel.on_new_signal)
        self._signal_worker.signals_closed.connect(self._active_signals_panel.on_signals_closed)
        self._market_worker.prices_updated.connect(self._active_signals_panel.on_prices_updated)
        self._active_signals_panel.vpmv_breakdown_requested.connect(
            self._market_worker.request_vpmv_breakdown
        )
        self._market_worker.vpmv_breakdown_ready.connect(
            self._active_signals_panel.on_vpmv_breakdown_ready
        )
        self._signal_worker.start()
        self._workers.append(self._signal_worker)

        # Divergence worker — sinyal coinlerin z-score ayrışması
        self._divergence_worker = DivergenceWorker(redis_url, parent=self)
        self._divergence_worker.divergence_updated.connect(
            self._divergence_panel.on_divergence_updated
        )
        self._ranking_panel._worker.ranking_updated.connect(
            self._divergence_panel.on_ranking_updated
        )
        self._divergence_worker.status_updated.connect(
            self._divergence_panel.on_status_updated
        )
        self._divergence_panel.tf_combo().currentTextChanged.connect(
            self._divergence_worker.set_timeframe
        )
        self._signal_worker.signals_loaded.connect(
            self._divergence_worker.set_symbols
        )
        self._signal_worker.new_signal.connect(
            self._divergence_worker.add_symbol
        )
        self._divergence_worker.start()
        self._workers.append(self._divergence_worker)

        # VPMV worker — sinyal sonrası VPMV momentum ayrışması
        self._vpmv_worker = VpmvWorker(redis_url, parent=self)
        self._vpmv_worker.vpmv_updated.connect(self._vpmv_panel.on_vpmv_updated)
        self._vpmv_worker.status_updated.connect(self._vpmv_panel.on_status_updated)
        self._signal_worker.signals_loaded.connect(self._vpmv_worker.set_symbols)
        self._signal_worker.new_signal.connect(self._vpmv_worker.add_symbol)
        self._vpmv_worker.start()
        self._workers.append(self._vpmv_worker)

        # Deviso panel — signals_loaded/new_signal/signals_closed ile beslenir
        self._signal_worker.signals_loaded.connect(self._deviso_panel.on_signals_loaded)
        self._signal_worker.new_signal.connect(self._deviso_panel.on_new_signal)
        self._signal_worker.signals_closed.connect(self._deviso_panel.on_signals_closed)

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_health_updated(self, status: dict) -> None:
        redis_ok = status["redis"] == ServiceStatus.OK
        db_ok = status["db"] == ServiceStatus.OK

        if redis_ok:
            self._status_redis.set_ok()
        else:
            self._status_redis.set_error()

        if db_ok:
            self._status_db.set_ok()
        else:
            self._status_db.set_error()

        self._status_symbols.setText(f"{status['symbol_count']} sembol")
        self._status_signals.setText(f"{status['active_signals']} aktif sinyal")

        if redis_ok and db_ok:
            self._connect_btn.setText("⬤  Bağlı")
        else:
            self._connect_btn.setText("⬤  Bağlantı Hatası")

    def _is_service_running(self) -> bool:
        try:
            out = subprocess.run(
                "launchctl list | grep com.trader.runner",
                shell=True, capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            if out:
                return out.split()[0] != "-"
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return False

    def _update_service_btn(self) -> None:
        running = self._is_service_running()
        if running:
            self._service_btn.setText("■  Durdur")
            self._service_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['bg_secondary']}; color: {COLORS['red']};"
                f" border: 1px solid {COLORS['border']}; border-radius: 4px;"
                f" padding: 2px 8px; font-size: 11px; }}"
                f" QPushButton:hover {{ border-color: {COLORS['red']}; }}"
            )
        else:
            self._service_btn.setText("▶  Başlat")
            self._service_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['bg_secondary']}; color: {COLORS['green']};"
                f" border: 1px solid {COLORS['border']}; border-radius: 4px;"
                f" padding: 2px 8px; font-size: 11px; }}"
                f" QPushButton:hover {{ border-color: {COLORS['green']}; }}"
            )

    @pyqtSlot()
    def _toggle_service(self) -> None:
        import os
        uid = os.getuid()
        if self._is_service_running():
            cmd = ["launchctl", "kill", "SIGTERM", f"gui/{uid}/com.trader.runner"]
        else:
            cmd = ["launchctl", "kickstart", f"gui/{uid}/com.trader.runner"]
        subprocess.run(cmd, timeout=5, check=False)
        QTimer.singleShot(2000, self._update_service_btn)

    @pyqtSlot(str)
    def _on_toolbar_search(self, text: str) -> None:
        """Toolbar arama kutusunu watchlist filtresine yönlendirir."""
        self._watchlist_panel._search.setText(text)  # noqa: SLF001

    @pyqtSlot(str)
    def _on_toolbar_tf_changed(self, tf: str) -> None:
        self._chart_panel.set_tf(tf)

    @pyqtSlot(str, float, float, float, float)
    def _on_price_updated(self, symbol: str, price: float, change_pct: float, _volume: float = 0.0, _funding: float = 0.0) -> None:
        # Toolbar sembol arama kutusundaki sembol seçiliyse başlık güncelle
        if self._symbol_search.text().upper() == symbol:
            sign = "+" if change_pct > 0 else ""
            self.setWindowTitle(
                f"TRader Terminal  —  {symbol}  {price:,.4f}  ({sign}{change_pct:.2f}%)"
            )

    @pyqtSlot(bool, str)
    def _on_market_connection(self, ok: bool, msg: str) -> None:
        if ok:
            self._status_redis.set_ok()
        else:
            self._status_redis.set_error()

    @pyqtSlot(bool, str)
    def _on_signal_connection(self, ok: bool, msg: str) -> None:
        if ok:
            self._status_db.set_ok()
        else:
            self._status_db.set_error()

    @pyqtSlot()
    def _update_clock(self) -> None:
        from datetime import datetime
        self._status_time.setText(
            datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
        )

    # ── Layout Kaydet / Yükle ─────────────────────────────────────────────

    def _save_layout(self) -> None:
        from PyQt6.QtCore import QSettings
        s = QSettings("TRader", "Terminal")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())

    def _restore_layout(self) -> None:
        from PyQt6.QtCore import QSettings
        s = QSettings("TRader", "Terminal")
        geom = s.value("geometry")
        state = s.value("state")
        if geom:
            self.restoreGeometry(geom)
        if state:
            self.restoreState(state)
        else:
            QTimer.singleShot(200, self._apply_default_sizes)

    def _apply_default_sizes(self) -> None:
        w = self.width()
        h = self.height()
        watchlist_w = 300
        chart_w = w - watchlist_w
        chart_h = int(h * 0.68)
        bottom_h = h - chart_h

        self.resizeDocks(
            [self._docks["watchlist"], self._docks["chart"]],
            [watchlist_w, chart_w],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._docks["chart"], self._docks["active_sig"]],
            [chart_h, bottom_h],
            Qt.Orientation.Vertical,
        )

    def _reset_layout(self) -> None:
        from PyQt6.QtCore import QSettings
        QSettings("TRader", "Terminal").clear()
        self.statusBar().showMessage("Layout sıfırlandı — yeniden başlatın.", 3000)

    # ── Kapat ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._save_layout()
        self._log_panel.cleanup()
        for w in self._workers:
            w.stop()
        event.accept()
