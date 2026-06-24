"""
PaperTradePanel — sanal portföy takip paneli.
Üst: bakiye / PnL / win rate / drawdown özeti
Alt: açık pozisyonlar tablosu + kapalı işlem geçmişi
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import json

import psycopg2
import psycopg2.extras
import pyarrow as _pa
import redis as _redis_lib
from PyQt6.QtCore import Qt, QPoint, QThread, QTimer, pyqtSignal  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS


class _FetchWorker(QThread):
    fetched = pyqtSignal(object, list, list)  # (portfolio_dict, open_rows, hist_rows)

    def __init__(self, db_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self._db_config = db_config

    def run(self) -> None:
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM paper_portfolio WHERE strategy = 'conf_100'")
                pf = dict(cur.fetchone()) if cur.rowcount else None

                cur.execute("""
                    SELECT id, symbol, signal_type, interval, strategy, source,
                           entry_price, stop_loss_price, take_profit_price,
                           trailing_stop_price, opened_at, vpms_score, z_score_entry
                    FROM paper_trades
                    WHERE status = 'open'
                    ORDER BY opened_at DESC
                """)
                open_rows = [dict(r) for r in cur.fetchall()]

                cur.execute("""
                    SELECT symbol, signal_type, interval, strategy, source,
                           entry_price, exit_price, pnl_usd, pnl_pct,
                           close_reason, closed_at
                    FROM paper_trades
                    WHERE status = 'closed'
                    ORDER BY closed_at DESC
                    LIMIT 200
                """)
                hist_rows = [dict(r) for r in cur.fetchall()]

            conn.close()
            self.fetched.emit(pf, open_rows, hist_rows)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("[PaperTradePanel] fetch hatası: %s", exc, exc_info=True)

OPEN_COLS  = ["Sembol", "Yön", "TF", "Strateji", "Giriş$", "Fiyat$", "P&L$", "P&L%", "SL%", "TP%", "Trail$", "VPMS", "Süre"]
HIST_COLS  = ["Sembol", "Yön", "TF", "Strateji", "Giriş$", "Çıkış$", "P&L$", "P&L%", "Neden", "Kapatma"]


def _age(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = int((datetime.now(timezone.utc) - dt).total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}dk"
    return f"{secs // 3600}s{(secs % 3600) // 60}dk"


def _pnl_color(val: float | None) -> QColor:
    if val is None:
        return QColor(COLORS["text_muted"])
    return QColor(COLORS["green"] if val >= 0 else COLORS["red"])


def _item(text: str, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    it.setTextAlignment(align)
    return it


class _NumItem(QTableWidgetItem):
    """Numerik değerleri doğru sıralayan QTableWidgetItem."""
    def __init__(self, text: str, sort_val: float):
        super().__init__(text)
        self._sort_val = sort_val
        self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _NumItem):
            return self._sort_val < other._sort_val
        return super().__lt__(other)


class PaperTradePanel(QWidget):

    symbol_selected = pyqtSignal(str, str)  # (symbol, interval)

    def __init__(self, db_config: dict[str, Any], redis_url: str = "", parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._redis_url = redis_url
        self._open_prices: dict[str, float] = {}
        self._open_ids: list[int] = []
        self._open_rows: list[dict] = []
        self._open_filter: dict[str, str] = {"side": "Tümü", "tf": "Tümü", "search": ""}
        self._hist_filter: dict[str, str] = {"side": "Tümü", "tf": "Tümü", "reason": "Tümü", "search": ""}

        # Redis bağlantısı — paper trade sembolleri için direkt polling (binary, Arrow)
        self._redis: _redis_lib.Redis | None = None
        if redis_url:
            try:
                self._redis = _redis_lib.Redis.from_url(
                    redis_url, decode_responses=False,
                    socket_connect_timeout=2, socket_timeout=2,
                )
                self._redis.ping()
            except Exception:  # pylint: disable=broad-exception-caught
                self._redis = None

        self._setup_ui()

        self._worker = _FetchWorker(db_config, parent=self)
        self._worker.fetched.connect(self._on_fetched)

        self._open_table.clicked.connect(self._on_table_clicked)
        self._hist_table.clicked.connect(self._on_table_clicked)
        self._open_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._open_table.customContextMenuRequested.connect(self._on_open_context_menu)

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
        self._lbl_balance   = self._stat_label("Bakiye", "$10,000.00")
        self._lbl_pnl       = self._stat_label("Toplam P&L", "$0.00")
        self._lbl_winrate   = self._stat_label("Win Rate", "—")
        self._lbl_drawdown  = self._stat_label("Max DD", "0.00%")
        self._lbl_open      = self._stat_label("Açık", "0")
        for w in [self._lbl_balance, self._lbl_pnl, self._lbl_winrate,
                  self._lbl_drawdown, self._lbl_open]:
            summary.addWidget(w)
        summary.addStretch()
        root.addLayout(summary)

        # ── Tab widget ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
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
        """)

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
        self._open_cb_side   = self._make_combo(["Tümü", "Long", "Short"])
        self._open_cb_tf     = self._make_combo(["Tümü", "5m", "15m", "1h", "4h"])
        self._open_search    = self._make_search("Sembol ara...")
        self._open_cb_side.currentTextChanged.connect(lambda v: self._set_open_filter("side", v))
        self._open_cb_tf.currentTextChanged.connect(lambda v: self._set_open_filter("tf", v))
        self._open_search.textChanged.connect(lambda v: self._set_open_filter("search", v))
        open_bar.addWidget(self._btn_manual)
        open_bar.addSpacing(8)
        open_bar.addWidget(QLabel("Yön:")); open_bar.addWidget(self._open_cb_side)
        open_bar.addWidget(QLabel("TF:"));  open_bar.addWidget(self._open_cb_tf)
        open_bar.addWidget(self._open_search)
        open_bar.addStretch()
        open_layout.addLayout(open_bar)

        self._open_table = self._make_table(OPEN_COLS)
        open_layout.addWidget(self._open_table)

        # ── Kapalı işlemler tab ──
        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)
        hist_layout.setContentsMargins(4, 4, 4, 0)
        hist_layout.setSpacing(4)

        hist_bar = QHBoxLayout()
        hist_bar.setSpacing(6)
        self._hist_cb_side   = self._make_combo(["Tümü", "Long", "Short"])
        self._hist_cb_tf     = self._make_combo(["Tümü", "5m", "15m", "1h", "4h"])
        self._hist_cb_reason = self._make_combo(["Tümü"])
        self._hist_search    = self._make_search("Sembol ara...")
        self._hist_cb_side.currentTextChanged.connect(lambda v: self._set_hist_filter("side", v))
        self._hist_cb_tf.currentTextChanged.connect(lambda v: self._set_hist_filter("tf", v))
        self._hist_cb_reason.currentTextChanged.connect(lambda v: self._set_hist_filter("reason", v))
        self._hist_search.textChanged.connect(lambda v: self._set_hist_filter("search", v))
        hist_bar.addWidget(QLabel("Yön:"));   hist_bar.addWidget(self._hist_cb_side)
        hist_bar.addWidget(QLabel("TF:"));    hist_bar.addWidget(self._hist_cb_tf)
        hist_bar.addWidget(QLabel("Neden:")); hist_bar.addWidget(self._hist_cb_reason)
        hist_bar.addWidget(self._hist_search)
        hist_bar.addStretch()
        hist_layout.addLayout(hist_bar)

        self._hist_table = self._make_table(HIST_COLS)
        hist_layout.addWidget(self._hist_table)

        self._tabs.addTab(open_widget, "Açık Pozisyonlar")
        self._tabs.addTab(hist_widget, "Kapalı İşlemler")
        root.addWidget(self._tabs, stretch=1)

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
        cb.setStyleSheet(f"""
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
        """)
        return cb

    @staticmethod
    def _make_search(placeholder: str) -> QLineEdit:
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        le.setFixedHeight(24)
        le.setFixedWidth(130)
        le.setStyleSheet(f"""
            QLineEdit {{
                background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};
                border: 1px solid #444; border-radius: 3px;
                font-size: 11px; padding: 0 6px;
            }}
        """)
        return le

    @staticmethod
    def _make_table(cols: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.setSortingEnabled(True)
        t.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                gridline-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                font-size: 12px;
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['accent']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_muted']};
                padding: 3px;
                border: none;
                font-size: 11px;
            }}
            QTableWidget::item:alternate {{
                background-color: {COLORS['bg_primary']};
            }}
        """)
        return t

    # ── Fiyat güncellemesi (market worker'dan) ────────────────────────────

    def on_price_updated(self, symbol: str, price: float, *_) -> None:
        self._open_prices[symbol] = price
        self._refresh_price_cells(symbol, price)

    def _poll_prices(self) -> None:
        if not self._redis or not self._open_rows:
            return
        try:
            syms = list({r["symbol"] for r in self._open_rows})
            pipe = self._redis.pipeline(transaction=False)
            for sym in syms:
                pipe.get(f"live_kline_data:{sym}:1m".encode())
            results = pipe.execute()
            for sym, raw in zip(syms, results):
                price = self._extract_close(raw)
                if price:
                    self._open_prices[sym] = price
                    self._refresh_price_cells(sym, price)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    @staticmethod
    def _extract_close(raw: bytes | None) -> float | None:
        if not raw:
            return None
        try:
            if raw[:4] == b"ARDF":
                reader = _pa.ipc.open_stream(raw[4:])
                df = reader.read_pandas()
                if "close" in df.columns and not df.empty:
                    return float(df["close"].iloc[-1])
            else:
                d = json.loads(raw.decode("utf-8"))
                return float(d.get("price") or d.get("last_price") or 0) or None
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return None

    def _refresh_price_cells(self, symbol: str, live: float) -> None:
        # Sembol + TF ile eşleşen tüm satırları bul (sıralama sonrası doğru index için)
        row_data = {row["interval"]: row for row in self._open_rows if row["symbol"] == symbol}
        if not row_data:
            return

        for t_idx in range(self._open_table.rowCount()):
            sym_item = self._open_table.item(t_idx, 0)
            tf_item  = self._open_table.item(t_idx, 2)
            if not sym_item or sym_item.text() != symbol:
                continue
            tf  = tf_item.text() if tf_item else ""
            row = row_data.get(tf) or next(iter(row_data.values()))

            side  = row["signal_type"]
            entry = float(row["entry_price"])
            sl    = row["stop_loss_price"]
            tp    = row["take_profit_price"]

            pnl_pct = (live - entry) / entry * 100 if side == "Long" else (entry - live) / entry * 100
            pnl_usd = pnl_pct / 100 * 100

            if sl and live:
                sl_dist = (float(sl) - live) / live * 100 if side == "Long" else (live - float(sl)) / live * 100
                sl_str  = f"{sl_dist:+.2f}%"
            else:
                sl_dist, sl_str = None, "—"

            tp_str = (
                f"{((float(tp) - live) / live * 100 if side == 'Long' else (live - float(tp)) / live * 100):+.2f}%"
                if tp and live else "—"
            )

            sl_danger = sl_dist is not None and abs(sl_dist) < 1.0
            danger_bg = QColor("#3d1515")

            updates: dict[int, tuple[str, float, QColor | None]] = {
                5: (f"{live:.5g}",       live,    None),
                6: (f"{pnl_usd:+.2f}$", pnl_usd, _pnl_color(pnl_usd)),
                7: (f"{pnl_pct:+.2f}%", pnl_pct, _pnl_color(pnl_usd)),
                8: (sl_str, sl_dist if sl_dist is not None else 0,
                    QColor("#ff4444") if sl_danger else QColor(COLORS["red"])),
                9: (tp_str, 0, QColor(COLORS["green"])),
            }
            for c_idx, (text, sort_val, fg) in updates.items():
                it = self._open_table.item(t_idx, c_idx)
                if it is None:
                    continue
                it.setText(text)
                if isinstance(it, _NumItem):
                    it._sort_val = sort_val
                if fg:
                    it.setForeground(fg)
                it.setBackground(danger_bg if sl_danger else QColor(0, 0, 0, 0))

    # ── Veri yükleme ──────────────────────────────────────────────────────

    def _trigger_fetch(self) -> None:
        if not self._worker.isRunning():
            self._worker.start()

    def _on_fetched(self, pf: dict | None, open_rows: list, hist_rows: list) -> None:
        unrealized = self._fill_open(open_rows)
        if pf:
            self._update_summary(pf, unrealized)
        self._fill_hist(hist_rows)
        self._tabs.setTabText(0, f"Açık Pozisyonlar ({len(open_rows)})")
        self._tabs.setTabText(1, f"Kapalı İşlemler ({len(hist_rows)})")

    def _update_summary(self, pf: dict, unrealized: float = 0.0) -> None:
        balance    = float(pf["balance"])
        realized   = float(pf["total_pnl_usd"])
        total      = int(pf["total_trades"])
        wins       = int(pf["winning_trades"])
        dd         = float(pf["max_drawdown_pct"])
        total_pnl  = realized + unrealized

        pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
        wr_color  = COLORS["green"] if total > 0 and wins / total >= 0.5 else COLORS["red"]
        wr_str    = f"{wins}/{total} ({wins/total*100:.0f}%)" if total > 0 else "—"

        unr_str   = f" ({unrealized:+.2f}$ açık)" if unrealized != 0 else ""
        pnl_str   = f"${total_pnl:+,.2f}{unr_str}"

        self._stat_value(self._lbl_balance,  f"${balance + unrealized:,.2f}")
        self._stat_value(self._lbl_pnl,      pnl_str, pnl_color)
        self._stat_value(self._lbl_winrate,  wr_str, wr_color)
        self._stat_value(self._lbl_drawdown, f"{dd:.2f}%",
                         COLORS["red"] if dd > 5 else COLORS["text_primary"])

    # OPEN_COLS = [Sembol0, Yön1, TF2, Strateji3, Giriş$4, Fiyat$5, P&L$6, P&L%7, SL%8, TP%9, Trail$10, VPMS11, Süre12]
    _OPEN_NUM_COLS = {4, 5, 6, 7, 8, 9, 10, 11}  # numerik sıralama gereken kolonlar

    def _fill_open(self, rows: list[dict]) -> float:
        self._open_table.setSortingEnabled(False)
        self._open_table.setRowCount(len(rows))
        self._stat_value(self._lbl_open, str(len(rows)))
        self._open_ids = [r["id"] for r in rows]
        self._open_rows = rows

        total_unrealized = 0.0
        for r_idx, row in enumerate(rows):
            sym    = row["symbol"]
            side   = row["signal_type"]
            tf     = row["interval"]
            strat  = row.get("strategy", "")
            src    = row.get("source", "signal")
            entry  = float(row["entry_price"])
            trail  = row["trailing_stop_price"]
            vpms   = row.get("vpms_score")

            live   = self._open_prices.get(sym, entry)
            pnl_pct = (live - entry) / entry * 100 if side == "Long" else (entry - live) / entry * 100
            pnl_usd = pnl_pct / 100 * 100
            total_unrealized += pnl_usd

            sl = row["stop_loss_price"]
            tp = row["take_profit_price"]

            if sl and live:
                sl_dist = (float(sl) - live) / live * 100 if side == "Long" else (live - float(sl)) / live * 100
                sl_str  = f"{sl_dist:+.2f}%"
            else:
                sl_dist, sl_str = None, "—"

            if tp and live:
                tp_dist = (float(tp) - live) / live * 100 if side == "Long" else (live - float(tp)) / live * 100
                tp_str  = f"{tp_dist:+.2f}%"
            else:
                tp_dist, tp_str = None, "—"

            trail_str   = f"{float(trail):.5g}" if trail else "—"
            vpms_val    = float(vpms) if vpms is not None else 0.0
            vpms_str    = f"{vpms_val:.1f}" if vpms is not None else "—"
            strat_label = "✋ " + strat if src == "manual" else strat
            sl_danger   = sl_dist is not None and abs(sl_dist) < 1.0

            # (text, sort_value)
            cell_data = [
                (sym,                  None),
                (side,                 None),
                (tf,                   None),
                (strat_label,          None),
                (f"{entry:.5g}",       entry),
                (f"{live:.5g}",        live),
                (f"{pnl_usd:+.2f}$",  pnl_usd),
                (f"{pnl_pct:+.2f}%",  pnl_pct),
                (sl_str,               sl_dist if sl_dist is not None else 0.0),
                (tp_str,               tp_dist if tp_dist is not None else 0.0),
                (trail_str,            float(trail) if trail else 0.0),
                (vpms_str,             vpms_val),
                (_age(row["opened_at"]), None),
            ]

            for c_idx, (text, sort_val) in enumerate(cell_data):
                it = _NumItem(text, sort_val) if sort_val is not None else _item(text)
                if sl_danger:
                    it.setBackground(QColor("#3d1515"))
                if c_idx == 1:
                    it.setForeground(QColor(COLORS["green"]) if side == "Long" else QColor(COLORS["red"]))
                if c_idx in (6, 7):
                    it.setForeground(_pnl_color(pnl_usd))
                if c_idx == 8:
                    it.setForeground(QColor("#ff4444") if sl_danger else QColor(COLORS["red"]))
                if c_idx == 9:
                    it.setForeground(QColor(COLORS["green"]))
                if c_idx == 10 and trail:
                    it.setForeground(QColor(COLORS["accent"]))
                self._open_table.setItem(r_idx, c_idx, it)

        self._open_table.setSortingEnabled(True)
        self._apply_open_filter()
        return total_unrealized

    # HIST_COLS = [Sembol0, Yön1, TF2, Strateji3, Giriş$4, Çıkış$5, P&L$6, P&L%7, Neden8, Kapatma9]

    def _fill_hist(self, rows: list[dict]) -> None:
        self._hist_table.setSortingEnabled(False)
        self._hist_table.setRowCount(len(rows))

        reasons: set[str] = set()
        for r_idx, row in enumerate(rows):
            pnl_usd = float(row["pnl_usd"]) if row["pnl_usd"] else 0.0
            pnl_pct = float(row["pnl_pct"]) if row["pnl_pct"] else 0.0
            closed  = row["closed_at"]
            if isinstance(closed, datetime) and closed.tzinfo is None:
                closed = closed.replace(tzinfo=timezone.utc)
            src         = row.get("source", "signal")
            strat       = row.get("strategy", "")
            strat_label = "✋ " + strat if src == "manual" else strat
            reason      = row["close_reason"] or "—"
            reasons.add(reason)

            cell_data = [
                (row["symbol"],                                              None),
                (row["signal_type"],                                         None),
                (row["interval"],                                            None),
                (strat_label,                                                None),
                (f"{float(row['entry_price']):.5g}",                        float(row["entry_price"])),
                (f"{float(row['exit_price']):.5g}" if row["exit_price"] else "—",
                 float(row["exit_price"]) if row["exit_price"] else 0.0),
                (f"{pnl_usd:+.2f}$",  pnl_usd),
                (f"{pnl_pct:+.2f}%",  pnl_pct),
                (reason,               None),
                (closed.strftime("%d/%m %H:%M") if closed else "—", None),
            ]
            for c_idx, (text, sort_val) in enumerate(cell_data):
                it = _NumItem(text, sort_val) if sort_val is not None else _item(text)
                if c_idx == 1:
                    it.setForeground(
                        QColor(COLORS["green"]) if row["signal_type"] == "Long"
                        else QColor(COLORS["red"])
                    )
                if c_idx in (6, 7):
                    it.setForeground(_pnl_color(pnl_usd))
                self._hist_table.setItem(r_idx, c_idx, it)

        # Neden dropdown'ını güncelle
        cur_reason = self._hist_cb_reason.currentText()
        self._hist_cb_reason.blockSignals(True)
        self._hist_cb_reason.clear()
        self._hist_cb_reason.addItems(["Tümü"] + sorted(reasons))
        idx = self._hist_cb_reason.findText(cur_reason)
        self._hist_cb_reason.setCurrentIndex(max(0, idx))
        self._hist_cb_reason.blockSignals(False)

        self._hist_table.setSortingEnabled(True)
        self._apply_hist_filter()

    # ── Filtre ────────────────────────────────────────────────────────────

    def _set_open_filter(self, key: str, val: str) -> None:
        self._open_filter[key] = val
        self._apply_open_filter()

    def _set_hist_filter(self, key: str, val: str) -> None:
        self._hist_filter[key] = val
        self._apply_hist_filter()

    def _apply_open_filter(self) -> None:
        side   = self._open_filter["side"]
        tf     = self._open_filter["tf"]
        search = self._open_filter["search"].upper()
        visible = 0
        for r in range(self._open_table.rowCount()):
            sym_it  = self._open_table.item(r, 0)
            side_it = self._open_table.item(r, 1)
            tf_it   = self._open_table.item(r, 2)
            hide = bool(
                (side != "Tümü" and side_it and side_it.text() != side) or
                (tf   != "Tümü" and tf_it   and tf_it.text()   != tf)   or
                (search and sym_it and search not in sym_it.text().upper())
            )
            self._open_table.setRowHidden(r, hide)
            if not hide:
                visible += 1
        self._tabs.setTabText(0, f"Açık Pozisyonlar ({visible})")

    def _apply_hist_filter(self) -> None:
        side   = self._hist_filter["side"]
        tf     = self._hist_filter["tf"]
        reason = self._hist_filter["reason"]
        search = self._hist_filter["search"].upper()
        visible = 0
        for r in range(self._hist_table.rowCount()):
            sym_it    = self._hist_table.item(r, 0)
            side_it   = self._hist_table.item(r, 1)
            tf_it     = self._hist_table.item(r, 2)
            reason_it = self._hist_table.item(r, 8)
            hide = bool(
                (side   != "Tümü" and side_it   and side_it.text()   != side)   or
                (tf     != "Tümü" and tf_it     and tf_it.text()     != tf)     or
                (reason != "Tümü" and reason_it and reason_it.text() != reason) or
                (search and sym_it and search not in sym_it.text().upper())
            )
            self._hist_table.setRowHidden(r, hide)
            if not hide:
                visible += 1
        self._tabs.setTabText(1, f"Kapalı İşlemler ({visible})")

    def _on_table_clicked(self, index) -> None:
        table = self.sender()
        row = index.row()
        sym_item = table.item(row, 0)
        tf_item  = table.item(row, 2)
        if sym_item and tf_item:
            self.symbol_selected.emit(sym_item.text(), tf_item.text())

    def _on_open_context_menu(self, pos: QPoint) -> None:
        row = self._open_table.rowAt(pos.y())
        if row < 0:
            return
        sym_item = self._open_table.item(row, 0)
        tf_item  = self._open_table.item(row, 2)
        if not sym_item or not tf_item:
            return
        sym_txt = sym_item.text()
        tf_txt  = tf_item.text()
        trade_id = next(
            (r["id"] for r in self._open_rows
             if r["symbol"] == sym_txt and r["interval"] == tf_txt),
            None,
        )
        if trade_id is None:
            return
        sym_item  = self._open_table.item(row, 0)
        side_item = self._open_table.item(row, 1)
        sym  = sym_item.text() if sym_item else "?"
        side = side_item.text() if side_item else "?"

        menu = QMenu(self)
        act_close = menu.addAction(f"Manuel Kapat — {sym} {side}")
        action = menu.exec(self._open_table.viewport().mapToGlobal(pos))
        if action == act_close:
            self._manual_close(trade_id, sym)

    def _manual_close(self, trade_id: int, symbol: str) -> None:
        price = self._open_prices.get(symbol, 0.0)
        if price == 0.0:
            try:
                import redis as _redis, json
                r = _redis.Redis.from_url(self._redis_url, socket_connect_timeout=2, decode_responses=True)
                raw = r.get(f"ticker:{symbol}")
                if raw:
                    d = json.loads(raw)
                    price = float(d.get("price") or d.get("last_price") or 0)
            except Exception:
                pass

        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT signal_type, entry_price FROM paper_trades WHERE id = %s", (trade_id,))
                rec = cur.fetchone()
                if not rec:
                    conn.close()
                    return
                side  = rec["signal_type"]
                entry = float(rec["entry_price"])
                if price == 0.0:
                    price = entry
                pnl_pct = (price - entry) / entry * 100 if side == "Long" else (entry - price) / entry * 100
                fee_usd = 100.0 * 0.0005 * 2
                pnl_usd = pnl_pct / 100 * 100 - fee_usd
                cur.execute("""
                    UPDATE paper_trades SET
                        status = 'closed', closed_at = NOW(),
                        exit_price = %s, close_reason = 'manual',
                        pnl_pct = %s, fee_usd = %s, pnl_usd = %s
                    WHERE id = %s
                """, (price, round(pnl_pct, 4), fee_usd, round(pnl_usd, 4), trade_id))
            conn.commit()
            conn.close()
            self._trigger_fetch()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("[PaperTrade] manuel kapat hatası: %s", exc)

    def _on_manual_trade(self) -> None:
        from desktop.dialogs.manual_trade_dialog import ManualTradeDialog
        dlg = ManualTradeDialog(self._db_config, self._redis_url, parent=self)
        if dlg.exec():
            self._trigger_fetch()
