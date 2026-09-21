"""
PaperTradePanel — sanal portföy takip paneli.
Üst: bakiye / PnL / win rate / drawdown özeti
Alt: açık pozisyonlar tablosu + kapalı işlem geçmişi
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras
import pyarrow as _pa
import redis as _redis_lib
from PyQt6.QtCore import (  # pylint: disable=no-name-in-module
    QPoint,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.models.paper_trade_model import (
    PaperHistModel,
    PaperHistProxyModel,
    PaperOpenModel,
    PaperOpenProxyModel,
)
from desktop.theme import COLORS
from signals.paper_trade_manager import LEVERAGE_BY_STRATEGY

# Açık pozisyon fiyatı/PnL'i için TEK canlı kaynak: live_kline_data:{symbol}:1m
# (WS beslemeli, bkz. _poll_prices). Bu bardan bu kadar süre yeni veri
# gelmemişse (ör. o sembolün WS alt-akışı donmuşsa — 27 Ağu 2026, MOVR'da
# ~1.5 saat fark edilmeden yaşandı) fiyat/PnL hücreleri "bayat" işaretlenir,
# sessizce yanlış bir sayı gösterilmez.
_STALE_THRESHOLD_SEC = 300


class _FetchWorker(QThread):
    fetched = pyqtSignal(list, list, list)  # (summary_rows, open_rows, hist_rows)

    def __init__(self, db_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self._db_config = db_config

    def run(self) -> None:
        conn = None
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Özet bar (Bakiye/Win Rate/Max DD) artık seçilen stratejiye göre
                # dinamik hesaplanıyor (21 Eyl 2026) — paper_portfolio tablosu
                # totalamount_rank1/manual için hiç satır içermiyor (_apply_close
                # portfolio=None geldiğinde bakiye/drawdown güncellemiyor), bu
                # yüzden kapanmış işlemler limitsiz çekilip equity-curve max
                # drawdown'u burada, Python tarafında hesaplanıyor. Aşağıdaki
                # open/hist sorgularındaki source='manual' dahiliyeti (1 Ağu
                # 2026 — ManualTradeDialog'un eski hatalı "tf_alignment_live"
                # varsayılanı) korunuyor.
                cur.execute(
                    """
                    SELECT strategy, pnl_usd, closed_at
                    FROM paper_trades
                    WHERE status = 'closed'
                          AND (strategy IN ('ta_kovalama_live', 'totalamount_rank1') OR source = 'manual')
                    ORDER BY closed_at ASC
                """
                )
                summary_rows = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT id, symbol, signal_type, interval, strategy, source,
                           entry_price, stop_loss_price, take_profit_price,
                           trailing_stop_price, opened_at, vpms_score, z_score_entry,
                           position_usd, devisso_score
                    FROM paper_trades
                    WHERE status = 'open'
                          AND (strategy IN ('ta_kovalama_live', 'totalamount_rank1') OR source = 'manual')
                    ORDER BY opened_at DESC
                """
                )
                open_rows = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """
                    SELECT id, symbol, signal_type, interval, strategy, source,
                           entry_price, exit_price, pnl_usd, pnl_pct,
                           close_reason, closed_at, opened_at,
                           stop_loss_price, take_profit_price, trailing_stop_price
                    FROM paper_trades
                    WHERE status = 'closed'
                          AND (strategy IN ('ta_kovalama_live', 'totalamount_rank1') OR source = 'manual')
                    ORDER BY closed_at DESC
                    LIMIT 200
                """
                )
                hist_rows = [dict(r) for r in cur.fetchall()]

            self.fetched.emit(summary_rows, open_rows, hist_rows)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).error(
                "[PaperTradePanel] fetch hatası: %s", exc, exc_info=True
            )
        finally:
            if conn is not None:
                conn.close()


class PaperTradePanel(QWidget):

    symbol_selected = pyqtSignal(str, str)  # (symbol, interval)
    signal_data_selected = pyqtSignal(dict)  # grafikte giriş/çıkış marker'ı için

    def __init__(self, db_config: dict[str, Any], redis_url: str = "", parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._redis_url = redis_url
        self._open_prices: dict[str, float] = {}
        self._summary_rows: list[dict] = []
        self._summary_strategy_initialized = False
        self._open_model = PaperOpenModel(self)
        self._open_proxy = PaperOpenProxyModel(self)
        self._open_proxy.setSourceModel(self._open_model)
        self._hist_model = PaperHistModel(self)
        self._hist_proxy = PaperHistProxyModel(self)
        self._hist_proxy.setSourceModel(self._hist_model)

        # Redis bağlantısı — paper trade sembolleri için direkt polling (binary, Arrow)
        self._redis: _redis_lib.Redis | None = None
        if redis_url:
            try:
                self._redis = _redis_lib.Redis.from_url(
                    redis_url,
                    decode_responses=False,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._redis.ping()
            except Exception:  # pylint: disable=broad-exception-caught
                self._redis = None

        self._setup_ui()

        self._worker = _FetchWorker(db_config, parent=self)
        self._worker.fetched.connect(self._on_fetched)

        self._open_view.clicked.connect(self._on_table_clicked)
        self._hist_view.clicked.connect(self._on_table_clicked)
        self._open_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._open_view.customContextMenuRequested.connect(self._on_open_context_menu)

        # DB fetch: 5 saniyede bir
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._trigger_fetch)
        self._timer.start(5000)

        # Redis price poll: 2 saniyede bir
        self._price_timer = QTimer(self)
        self._price_timer.timeout.connect(self._poll_prices)
        self._price_timer.start(2000)

        self._trigger_fetch()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Özet bar ──
        summary = QHBoxLayout()
        self._lbl_balance = self._stat_label("Bakiye", "$0.00")
        self._lbl_pnl = self._stat_label("Toplam P&L", "$0.00")
        self._lbl_winrate = self._stat_label("Win Rate", "—")
        self._lbl_drawdown = self._stat_label("Max DD", "-$0.00")
        self._lbl_open = self._stat_label("Açık", "0")
        for w in [
            self._lbl_balance,
            self._lbl_pnl,
            self._lbl_winrate,
            self._lbl_drawdown,
            self._lbl_open,
        ]:
            summary.addWidget(w)
        summary.addStretch()

        summary.addWidget(QLabel("Strateji:"))
        self._summary_cb_strategy = self._make_combo(["Tümü"])
        self._summary_cb_strategy.currentTextChanged.connect(lambda _v: self._recompute_summary())
        summary.addWidget(self._summary_cb_strategy)

        self._pt_toggle_btn = QPushButton("● PT: Aktif")
        self._pt_toggle_btn.setFixedWidth(110)
        self._pt_toggle_btn.setCheckable(False)
        self._pt_toggle_btn.clicked.connect(self._toggle_paper_trade)
        summary.addWidget(self._pt_toggle_btn)
        self._sync_pt_button()

        root.addLayout(summary)

        # ── Tab widget ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['bg_tertiary']};
                background: {COLORS['bg_secondary']};
            }}
            QTabBar::tab {{
                background: {COLORS['bg_tertiary']};
                color: {COLORS['text_muted']};
                padding: 5px 16px;
                border: none;
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border-bottom: 2px solid {COLORS['accent']};
            }}
            QTabBar::tab:hover {{
                color: {COLORS['text_primary']};
            }}
        """
        )

        # ── Açık pozisyonlar tab ──
        open_widget = QWidget()
        open_layout = QVBoxLayout(open_widget)
        open_layout.setContentsMargins(4, 4, 4, 0)
        open_layout.setSpacing(4)

        open_bar = QHBoxLayout()
        open_bar.setSpacing(6)
        self._btn_manual = QPushButton("+ Manuel İşlem")
        self._btn_manual.setFixedHeight(24)
        self._btn_manual.setStyleSheet(
            f"QPushButton {{ background: {COLORS['bg_tertiary']}; color: {COLORS['green']};"
            f" border: 1px solid {COLORS['green']}; border-radius: 3px; font-size: 11px; padding: 0 8px; }}"
            f"QPushButton:hover {{ background: #1a5c2a; }}"
        )
        self._btn_manual.clicked.connect(self._on_manual_trade)
        self._open_cb_side = self._make_combo(["Tümü", "Long", "Short"])
        self._open_cb_tf = self._make_combo(["Tümü", "5m", "15m", "1h", "4h"])
        self._open_cb_strategy = self._make_combo(["Tümü"])
        self._open_search = self._make_search("Sembol ara...")
        self._open_cb_side.currentTextChanged.connect(lambda v: self._set_open_filter("side", v))
        self._open_cb_tf.currentTextChanged.connect(lambda v: self._set_open_filter("tf", v))
        self._open_cb_strategy.currentTextChanged.connect(
            lambda v: self._set_open_filter("strategy", v)
        )
        self._open_search.textChanged.connect(lambda v: self._set_open_filter("search", v))
        open_bar.addWidget(self._btn_manual)
        open_bar.addSpacing(8)
        open_bar.addWidget(QLabel("Yön:"))
        open_bar.addWidget(self._open_cb_side)
        open_bar.addWidget(QLabel("TF:"))
        open_bar.addWidget(self._open_cb_tf)
        open_bar.addWidget(QLabel("Strateji:"))
        open_bar.addWidget(self._open_cb_strategy)
        open_bar.addWidget(self._open_search)
        open_bar.addStretch()
        open_layout.addLayout(open_bar)

        self._open_view = self._make_view(self._open_model, self._open_proxy)
        open_layout.addWidget(self._open_view)

        # ── Kapalı işlemler tab ──
        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)
        hist_layout.setContentsMargins(4, 4, 4, 0)
        hist_layout.setSpacing(4)

        hist_bar = QHBoxLayout()
        hist_bar.setSpacing(6)
        self._hist_cb_side = self._make_combo(["Tümü", "Long", "Short"])
        self._hist_cb_tf = self._make_combo(["Tümü", "5m", "15m", "1h", "4h"])
        self._hist_cb_reason = self._make_combo(["Tümü"])
        self._hist_cb_strategy = self._make_combo(["Tümü"])
        self._hist_search = self._make_search("Sembol ara...")
        self._hist_cb_side.currentTextChanged.connect(lambda v: self._set_hist_filter("side", v))
        self._hist_cb_tf.currentTextChanged.connect(lambda v: self._set_hist_filter("tf", v))
        self._hist_cb_reason.currentTextChanged.connect(
            lambda v: self._set_hist_filter("reason", v)
        )
        self._hist_cb_strategy.currentTextChanged.connect(
            lambda v: self._set_hist_filter("strategy", v)
        )
        self._hist_search.textChanged.connect(lambda v: self._set_hist_filter("search", v))
        hist_bar.addWidget(QLabel("Yön:"))
        hist_bar.addWidget(self._hist_cb_side)
        hist_bar.addWidget(QLabel("TF:"))
        hist_bar.addWidget(self._hist_cb_tf)
        hist_bar.addWidget(QLabel("Neden:"))
        hist_bar.addWidget(self._hist_cb_reason)
        hist_bar.addWidget(QLabel("Strateji:"))
        hist_bar.addWidget(self._hist_cb_strategy)
        hist_bar.addWidget(self._hist_search)
        hist_bar.addStretch()
        hist_layout.addLayout(hist_bar)

        self._hist_view = self._make_view(self._hist_model, self._hist_proxy)
        hist_layout.addWidget(self._hist_view)

        self._tabs.addTab(open_widget, "Açık Pozisyonlar")
        self._tabs.addTab(hist_widget, "Kapalı İşlemler")
        root.addWidget(self._tabs, stretch=1)

    def _is_pt_enabled(self) -> bool:
        if self._redis is None:
            return True
        try:
            val = self._redis.get("settings:paper_trade_enabled")
            return val != b"0"
        except Exception:  # pylint: disable=broad-exception-caught
            return True

    def _sync_pt_button(self) -> None:
        enabled = self._is_pt_enabled()
        if enabled:
            self._pt_toggle_btn.setText("⏸ PT Durdur")
            self._pt_toggle_btn.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-weight: bold;"
            )
        else:
            self._pt_toggle_btn.setText("▶ PT Başlat")
            self._pt_toggle_btn.setStyleSheet(f"color: {COLORS['green']}; font-weight: bold;")

    def _toggle_paper_trade(self) -> None:
        if self._redis is None:
            return
        try:
            enabled = self._is_pt_enabled()
            self._redis.set("settings:paper_trade_enabled", "0" if enabled else "1")
            self._sync_pt_button()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def _stat_label(self, title: str, value: str) -> QLabel:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:10px;")
        v = QLabel(value)
        v.setObjectName(f"stat_{title}")
        v.setStyleSheet(f"color:{COLORS['text_primary']}; font-size:13px; font-weight:bold;")
        lay.addWidget(t)
        lay.addWidget(v)
        box.setFixedWidth(130)
        return box

    @staticmethod
    def _stat_value(box: QWidget, text: str, color: str | None = None) -> None:
        labels = box.findChildren(QLabel)
        if len(labels) >= 2:
            labels[1].setText(text)
            c = color or COLORS["text_primary"]
            labels[1].setStyleSheet(f"color:{c}; font-size:13px; font-weight:bold;")

    @staticmethod
    def _make_combo(options: list[str]) -> QComboBox:
        cb = QComboBox()
        cb.addItems(options)
        cb.setFixedHeight(24)
        cb.setStyleSheet(
            f"""
            QComboBox {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                border: 1px solid #444; border-radius: 3px;
                font-size: 11px; padding: 0 6px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']};
            }}
        """
        )
        return cb

    @staticmethod
    def _make_search(placeholder: str) -> QLineEdit:
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        le.setFixedHeight(24)
        le.setFixedWidth(130)
        le.setStyleSheet(
            f"""
            QLineEdit {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                border: 1px solid #444; border-radius: 3px;
                font-size: 11px; padding: 0 6px;
            }}
        """
        )
        return le

    @staticmethod
    def _make_view(model, proxy) -> QTableView:
        v = QTableView()
        v.setModel(proxy)
        v.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        v.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.setAlternatingRowColors(True)
        v.verticalHeader().setVisible(False)
        v.horizontalHeader().setStretchLastSection(True)
        v.setSortingEnabled(True)
        v.setStyleSheet(
            f"""
            QTableView {{
                background-color: {COLORS['bg_secondary']};
                gridline-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                font-size: 12px;
            }}
            QTableView::item:selected {{
                background-color: {COLORS['accent']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_muted']};
                padding: 3px;
                border: none;
                font-size: 11px;
            }}
            QTableView::item:alternate {{
                background-color: {COLORS['bg_primary']};
            }}
        """
        )
        return v

    # ── Fiyat güncellemesi (TEK kaynak: live_kline_data, bkz. yukarıdaki not) ──

    def _poll_prices(self) -> None:
        symbols = self._open_model.symbols()
        if not self._redis or not symbols:
            return
        try:
            syms = list(symbols)
            pipe = self._redis.pipeline(transaction=False)
            for sym in syms:
                pipe.get(f"live_kline_data:{sym}:1m".encode())
            results = pipe.execute()
            now_ms = datetime.now().timestamp() * 1000
            price_updates: dict[str, tuple[float, bool]] = {}
            for sym, raw in zip(syms, results):
                price, open_time_ms = self._extract_close_and_time(raw)
                if price:
                    self._open_prices[sym] = price
                    stale = (
                        open_time_ms is None
                        or (now_ms - open_time_ms) / 1000 > _STALE_THRESHOLD_SEC
                    )
                    price_updates[sym] = (price, stale)
            if price_updates:
                self._open_model.update_prices(price_updates)
                self._recompute_summary()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    @staticmethod
    def _extract_close_and_time(raw: bytes | None) -> tuple[float | None, float | None]:
        if not raw:
            return None, None
        try:
            if raw[:4] == b"ARDF":
                reader = _pa.ipc.open_stream(raw[4:])
                try:
                    df = reader.read_pandas()
                finally:
                    reader.close()
                if "close" in df.columns and not df.empty:
                    price = float(df["close"].iloc[-1])
                    open_time_ms = (
                        float(df["open_time"].iloc[-1]) if "open_time" in df.columns else None
                    )
                    return price, open_time_ms
            else:
                d = json.loads(raw.decode("utf-8"))
                price = float(d.get("price") or d.get("last_price") or 0) or None
                return price, None
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return None, None

    # ── Veri yükleme ──────────────────────────────────────────────────────

    def _trigger_fetch(self) -> None:
        if not self._worker.isRunning():
            self._worker.start()

    def _on_fetched(self, summary_rows: list, open_rows: list, hist_rows: list) -> None:
        self._summary_rows = summary_rows
        self._fill_open(open_rows)
        self._populate_summary_strategy_options(summary_rows, open_rows)
        self._recompute_summary()
        self._fill_hist(hist_rows)
        self._tabs.setTabText(0, f"Açık Pozisyonlar ({len(open_rows)})")
        self._tabs.setTabText(1, f"Kapalı İşlemler ({len(hist_rows)})")

    def _populate_summary_strategy_options(self, summary_rows: list, open_rows: list) -> None:
        strategies: set[str] = set()
        latest_ts: dict[str, datetime] = {}
        for r in summary_rows:
            s = r["strategy"]
            strategies.add(s)
            ts = r["closed_at"]
            if ts and (s not in latest_ts or ts > latest_ts[s]):
                latest_ts[s] = ts
        for r in open_rows:
            s = r["strategy"]
            strategies.add(s)
            ts = r["opened_at"]
            if ts and (s not in latest_ts or ts > latest_ts[s]):
                latest_ts[s] = ts

        cur_selection = self._summary_cb_strategy.currentText()
        self._summary_cb_strategy.blockSignals(True)
        self._summary_cb_strategy.clear()
        self._summary_cb_strategy.addItems(["Tümü"] + sorted(strategies))
        if not self._summary_strategy_initialized and latest_ts:
            # İlk yüklemede en son aktif stratejiyi seçili getir (ör. totalamount_rank1);
            # sonraki fetch'lerde kullanıcının seçimi korunur.
            default_strategy = max(latest_ts, key=latest_ts.get)
            idx = self._summary_cb_strategy.findText(default_strategy)
            self._summary_cb_strategy.setCurrentIndex(max(0, idx))
            self._summary_strategy_initialized = True
        else:
            idx = self._summary_cb_strategy.findText(cur_selection)
            self._summary_cb_strategy.setCurrentIndex(max(0, idx))
        self._summary_cb_strategy.blockSignals(False)

    # Her iki strateji de $2000 sermayeyle başlatıldı (ta_kovalama_live:
    # paper_portfolio.initial_balance=2000; totalamount_rank1: kullanıcı
    # teyidi, 21 Eyl 2026 — "sermaye 2000 dolardı strateji başlarken").
    # manual: ad-hoc elle girilen işlemler, bir "başlangıç sermayesi"
    # kavramı yok — bakiye yerine sadece PnL toplamı gösterilir.
    _STRATEGY_INITIAL_BALANCE: dict[str, float] = {
        "totalamount_rank1": 2000.0,
        "ta_kovalama_live": 2000.0,
    }

    def _initial_balance_for(self, strategy: str) -> float:
        if strategy == "Tümü":
            return sum(self._STRATEGY_INITIAL_BALANCE.values())
        return self._STRATEGY_INITIAL_BALANCE.get(strategy, 0.0)

    def _compute_realized_stats(self, strategy: str) -> tuple[int, int, float, float, float]:
        """(toplam_islem, kazanan, gerceklesen_pnl, max_dd_usd, max_dd_pct).

        Equity, bilinen başlangıç sermayesinden başlatılır — bu sayede peak
        hiç sıfıra yakın kalmıyor ve % drawdown istikrarlı çıkıyor (sermayesiz
        stratejilerde erken küçük bir tepe paydaya bölününce %400+ gibi
        anlamsız değerler üretiyordu, 21 Eyl 2026).
        """
        rows = [r for r in self._summary_rows if strategy == "Tümü" or r["strategy"] == strategy]
        total = len(rows)
        wins = sum(1 for r in rows if float(r["pnl_usd"] or 0) > 0)
        initial_balance = self._initial_balance_for(strategy)
        total_pnl = 0.0
        equity = initial_balance
        peak = initial_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        for r in rows:  # closed_at ASC sıralı geldi (SQL) — filtre sırayı bozmaz
            pnl = float(r["pnl_usd"] or 0)
            total_pnl += pnl
            equity += pnl
            if equity > peak:
                peak = equity
            dd_usd = peak - equity
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if peak > 0:
                dd_pct = dd_usd / peak * 100
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
        return total, wins, total_pnl, max_dd_usd, max_dd_pct

    def _compute_open_stats_by_strategy(self) -> dict[str, tuple[int, float]]:
        """strateji -> (acik_pozisyon_sayisi, gerceklesmemis_pnl_usd)."""
        result: dict[str, tuple[int, float]] = {}
        for i in range(self._open_model.rowCount()):
            row = self._open_model.row_at(i)
            strat = row.raw.get("strategy", "")
            cnt, pnl = result.get(strat, (0, 0.0))
            result[strat] = (cnt + 1, pnl + row.pnl_usd)
        return result

    def _recompute_summary(self) -> None:
        strategy = self._summary_cb_strategy.currentText() or "Tümü"
        total, wins, realized, max_dd_usd, max_dd_pct = self._compute_realized_stats(strategy)
        open_stats = self._compute_open_stats_by_strategy()
        if strategy == "Tümü":
            open_count = sum(c for c, _ in open_stats.values())
            unrealized = sum(p for _, p in open_stats.values())
        else:
            open_count, unrealized = open_stats.get(strategy, (0, 0.0))
        total_pnl = realized + unrealized
        initial_balance = self._initial_balance_for(strategy)
        has_capital = initial_balance > 0
        balance = initial_balance + total_pnl if has_capital else total_pnl

        pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
        wr_color = COLORS["green"] if total > 0 and wins / total >= 0.5 else COLORS["red"]
        wr_str = f"{wins}/{total} ({wins/total*100:.0f}%)" if total > 0 else "—"

        unr_str = f" ({unrealized:+.2f}$ açık)" if unrealized != 0 else ""
        pnl_str = f"${total_pnl:+,.2f}{unr_str}"

        self._stat_value(self._lbl_balance, f"${balance:,.2f}", None if has_capital else pnl_color)
        self._stat_value(self._lbl_pnl, pnl_str, pnl_color)
        self._stat_value(self._lbl_winrate, wr_str, wr_color)
        if has_capital:
            dd_text = f"{max_dd_pct:.2f}%"
            dd_color = COLORS["red"] if max_dd_pct > 5 else COLORS["text_primary"]
        else:
            dd_text = f"-${max_dd_usd:,.2f}"
            dd_color = COLORS["red"] if max_dd_usd > 0 else COLORS["text_primary"]
        self._stat_value(self._lbl_drawdown, dd_text, dd_color)
        self._stat_value(self._lbl_open, str(open_count))

    def _fill_open(self, rows: list[dict]) -> None:
        items = []
        for row in rows:
            item = dict(row)
            item["live_price"] = self._open_prices.get(row["symbol"], float(row["entry_price"]))
            items.append(item)
        self._open_model.bulk_upsert(items, self._open_model.build_row, self._open_model.update_row)
        self._open_model.prune_missing({r["id"] for r in rows})

        strategies: set[str] = set()
        for r_idx in range(self._open_model.rowCount()):
            row_obj = self._open_model.row_at(r_idx)
            strategies.add(row_obj.strategy_label)

        cur_strategy = self._open_cb_strategy.currentText()
        self._open_cb_strategy.blockSignals(True)
        self._open_cb_strategy.clear()
        self._open_cb_strategy.addItems(["Tümü"] + sorted(strategies))
        idx = self._open_cb_strategy.findText(cur_strategy)
        self._open_cb_strategy.setCurrentIndex(max(0, idx))
        self._open_cb_strategy.blockSignals(False)

    def _fill_hist(self, rows: list[dict]) -> None:
        self._hist_model.bulk_upsert(rows, self._hist_model.build_row, self._hist_model.update_row)
        self._hist_model.prune_missing({r["id"] for r in rows})

        reasons: set[str] = set()
        strategies: set[str] = set()
        for r_idx in range(self._hist_model.rowCount()):
            row_obj = self._hist_model.row_at(r_idx)
            reasons.add(row_obj.reason)
            strategies.add(row_obj.strategy_label)

        cur_reason = self._hist_cb_reason.currentText()
        self._hist_cb_reason.blockSignals(True)
        self._hist_cb_reason.clear()
        self._hist_cb_reason.addItems(["Tümü"] + sorted(reasons))
        idx = self._hist_cb_reason.findText(cur_reason)
        self._hist_cb_reason.setCurrentIndex(max(0, idx))
        self._hist_cb_reason.blockSignals(False)

        cur_strategy = self._hist_cb_strategy.currentText()
        self._hist_cb_strategy.blockSignals(True)
        self._hist_cb_strategy.clear()
        self._hist_cb_strategy.addItems(["Tümü"] + sorted(strategies))
        idx = self._hist_cb_strategy.findText(cur_strategy)
        self._hist_cb_strategy.setCurrentIndex(max(0, idx))
        self._hist_cb_strategy.blockSignals(False)

    # ── Filtre ────────────────────────────────────────────────────────────

    def _set_open_filter(self, key: str, val: str) -> None:
        {
            "side": self._open_proxy.set_side,
            "tf": self._open_proxy.set_tf,
            "strategy": self._open_proxy.set_strategy,
            "search": self._open_proxy.set_search,
        }[key](val)
        self._tabs.setTabText(0, f"Açık Pozisyonlar ({self._open_proxy.rowCount()})")

    def _set_hist_filter(self, key: str, val: str) -> None:
        {
            "side": self._hist_proxy.set_side,
            "tf": self._hist_proxy.set_tf,
            "reason": self._hist_proxy.set_reason,
            "strategy": self._hist_proxy.set_strategy,
            "search": self._hist_proxy.set_search,
        }[key](val)
        self._tabs.setTabText(1, f"Kapalı İşlemler ({self._hist_proxy.rowCount()})")

    def _on_table_clicked(self, index) -> None:
        view = self.sender()
        if view is self._open_view:
            proxy, model = self._open_proxy, self._open_model
        elif view is self._hist_view:
            proxy, model = self._hist_proxy, self._hist_model
        else:
            return
        row = model.row_at(proxy.mapToSource(index).row())
        if row is None:
            return
        self.symbol_selected.emit(row.symbol, row.interval)
        self.signal_data_selected.emit(dict(row.raw))

    def _on_open_context_menu(self, pos: QPoint) -> None:
        index = self._open_view.indexAt(pos)
        if not index.isValid():
            return
        row = self._open_model.row_at(self._open_proxy.mapToSource(index).row())
        if row is None:
            return
        menu = QMenu(self)
        act_close = menu.addAction(f"Manuel Kapat — {row.symbol} {row.signal_type}")
        action = menu.exec(self._open_view.viewport().mapToGlobal(pos))
        if action == act_close:
            self._manual_close(row.id, row.symbol)

    def _manual_close(self, trade_id: int, symbol: str) -> None:
        price = self._open_prices.get(symbol, 0.0)
        if price == 0.0:
            r = None
            try:
                import json

                import redis as _redis

                r = _redis.Redis.from_url(
                    self._redis_url, socket_connect_timeout=2, decode_responses=True
                )
                raw = r.get(f"ticker:{symbol}")
                if raw:
                    d = json.loads(raw)
                    price = float(d.get("price") or d.get("last_price") or 0)
            except Exception:
                pass
            finally:
                if r is not None:
                    r.close()

        conn = None
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT signal_type, entry_price, position_usd, strategy "
                    "FROM paper_trades WHERE id = %s",
                    (trade_id,),
                )
                rec = cur.fetchone()
                if not rec:
                    return
                side = rec["signal_type"]
                entry = float(rec["entry_price"])
                position_usd = float(rec["position_usd"] or 100.0)
                leverage = LEVERAGE_BY_STRATEGY.get(rec["strategy"] or "", 1.0)
                if price == 0.0:
                    price = entry
                pnl_pct = (
                    (price - entry) / entry * 100
                    if side == "Long"
                    else (entry - price) / entry * 100
                )
                fee_usd = position_usd * 0.0005 * 2
                pnl_usd = pnl_pct / 100 * position_usd * leverage - fee_usd
                cur.execute(
                    """
                    UPDATE paper_trades SET
                        status = 'closed', closed_at = NOW(),
                        exit_price = %s, close_reason = 'manual',
                        pnl_pct = %s, fee_usd = %s, pnl_usd = %s
                    WHERE id = %s
                """,
                    (price, round(pnl_pct, 4), fee_usd, round(pnl_usd, 4), trade_id),
                )
            conn.commit()
            self._trigger_fetch()
        except Exception as exc:
            import logging

            logging.getLogger(__name__).error("[PaperTrade] manuel kapat hatası: %s", exc)
        finally:
            if conn is not None:
                conn.close()

    def _on_manual_trade(self) -> None:
        from desktop.dialogs.manual_trade_dialog import ManualTradeDialog

        dlg = ManualTradeDialog(self._db_config, self._redis_url, parent=self)
        if dlg.exec():
            self._trigger_fetch()
