"""
PaperTradePanel — sanal portföy takip paneli.
Üst: bakiye / PnL / win rate / drawdown özeti
Alt: açık pozisyonlar tablosu + kapalı işlem geçmişi
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

STRATEGY = "conf_100"


class _FetchWorker(QThread):
    fetched = pyqtSignal(object, list, list)  # (portfolio_dict, open_rows, hist_rows)

    def __init__(self, db_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self._db_config = db_config

    def run(self) -> None:
        try:
            conn = psycopg2.connect(**self._db_config)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM paper_portfolio WHERE strategy = %s", (STRATEGY,))
                pf = dict(cur.fetchone()) if cur.rowcount else None

                cur.execute("""
                    SELECT id, symbol, signal_type, interval,
                           entry_price, stop_loss_price, take_profit_price,
                           trailing_stop_price, opened_at, vpms_score, z_score_entry
                    FROM paper_trades
                    WHERE strategy = %s AND status = 'open'
                    ORDER BY opened_at DESC
                """, (STRATEGY,))
                open_rows = [dict(r) for r in cur.fetchall()]

                cur.execute("""
                    SELECT symbol, signal_type, interval,
                           entry_price, exit_price, pnl_usd, pnl_pct,
                           close_reason, closed_at
                    FROM paper_trades
                    WHERE strategy = %s AND status = 'closed'
                    ORDER BY closed_at DESC
                    LIMIT 200
                """, (STRATEGY,))
                hist_rows = [dict(r) for r in cur.fetchall()]

            conn.close()
            self.fetched.emit(pf, open_rows, hist_rows)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("[PaperTradePanel] fetch hatası: %s", exc, exc_info=True)

OPEN_COLS  = ["Sembol", "Yön", "TF", "Giriş$", "Fiyat$", "P&L$", "P&L%", "SL$", "TP$", "Süre", "Trail"]
HIST_COLS  = ["Sembol", "Yön", "TF", "Giriş$", "Çıkış$", "P&L$", "P&L%", "Neden", "Kapatma"]


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


class PaperTradePanel(QWidget):

    symbol_selected = pyqtSignal(str, str)  # (symbol, interval)

    def __init__(self, db_config: dict[str, Any], parent=None):
        super().__init__(parent)
        self._db_config = db_config
        self._open_prices: dict[str, float] = {}

        self._setup_ui()

        self._worker = _FetchWorker(db_config, parent=self)
        self._worker.fetched.connect(self._on_fetched)

        self._open_table.clicked.connect(self._on_table_clicked)
        self._hist_table.clicked.connect(self._on_table_clicked)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._trigger_fetch)
        self._timer.start(5000)
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

        # ── Açık pozisyonlar başlığı ──
        open_hdr = QLabel("Açık Pozisyonlar")
        open_hdr.setObjectName("section_title")
        root.addWidget(open_hdr)

        self._open_table = self._make_table(OPEN_COLS)
        self._open_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        root.addWidget(self._open_table, stretch=2)

        # ── Geçmiş başlığı ──
        hist_hdr = QLabel("Son Kapalı İşlemler")
        hist_hdr.setObjectName("section_title")
        root.addWidget(hist_hdr)

        self._hist_table = self._make_table(HIST_COLS)
        self._hist_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        root.addWidget(self._hist_table, stretch=3)

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
    def _make_table(cols: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
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

    # ── Veri yükleme ──────────────────────────────────────────────────────

    def _trigger_fetch(self) -> None:
        if not self._worker.isRunning():
            self._worker.start()

    def _on_fetched(self, pf: dict | None, open_rows: list, hist_rows: list) -> None:
        unrealized = self._fill_open(open_rows)
        if pf:
            self._update_summary(pf, unrealized)
        self._fill_hist(hist_rows)

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

    def _fill_open(self, rows: list[dict]) -> float:
        self._open_table.setRowCount(len(rows))
        self._stat_value(self._lbl_open, str(len(rows)))

        total_unrealized = 0.0
        for r_idx, row in enumerate(rows):
            sym    = row["symbol"]
            side   = row["signal_type"]
            tf     = row["interval"]
            entry  = float(row["entry_price"])
            trail  = row["trailing_stop_price"]

            live   = self._open_prices.get(sym, entry)
            if side == "Long":
                pnl_pct = (live - entry) / entry * 100
            else:
                pnl_pct = (entry - live) / entry * 100
            pnl_usd = pnl_pct / 100 * 100  # $100 pozisyon
            total_unrealized += pnl_usd

            sl    = row["stop_loss_price"]
            tp    = row["take_profit_price"]
            sl_str = f"{float(sl):.5g}" if sl else "—"
            tp_str = f"{float(tp):.5g}" if tp else "—"
            trail_str = f"{float(trail):.5g}" if trail else "—"

            cells = [
                sym, side, tf,
                f"{entry:.5g}", f"{live:.5g}",
                f"{pnl_usd:+.2f}$", f"{pnl_pct:+.2f}%",
                sl_str, tp_str,
                _age(row["opened_at"]),
                "✓" if trail else "—",
            ]
            for c_idx, text in enumerate(cells):
                it = _item(text)
                if c_idx == 1:
                    it.setForeground(
                        QColor(COLORS["green"]) if side == "Long" else QColor(COLORS["red"])
                    )
                if c_idx in (5, 6):
                    it.setForeground(_pnl_color(pnl_usd))
                if c_idx == 7:  # SL
                    it.setForeground(QColor(COLORS["red"]))
                if c_idx == 8:  # TP
                    it.setForeground(QColor(COLORS["green"]))
                self._open_table.setItem(r_idx, c_idx, it)

        return total_unrealized

    def _fill_hist(self, rows: list[dict]) -> None:
        self._hist_table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            pnl_usd = float(row["pnl_usd"]) if row["pnl_usd"] else 0.0
            pnl_pct = float(row["pnl_pct"]) if row["pnl_pct"] else 0.0
            closed  = row["closed_at"]
            if isinstance(closed, datetime) and closed.tzinfo is None:
                closed = closed.replace(tzinfo=timezone.utc)

            cells = [
                row["symbol"],
                row["signal_type"],
                row["interval"],
                f"{float(row['entry_price']):.5g}",
                f"{float(row['exit_price']):.5g}" if row["exit_price"] else "—",
                f"{pnl_usd:+.2f}$",
                f"{pnl_pct:+.2f}%",
                row["close_reason"] or "—",
                closed.strftime("%d/%m %H:%M") if closed else "—",
            ]
            for c_idx, text in enumerate(cells):
                it = _item(text)
                if c_idx == 1:
                    it.setForeground(
                        QColor(COLORS["green"]) if row["signal_type"] == "Long"
                        else QColor(COLORS["red"])
                    )
                if c_idx in (5, 6):
                    it.setForeground(_pnl_color(pnl_usd))
                self._hist_table.setItem(r_idx, c_idx, it)

    def _on_table_clicked(self, index) -> None:
        table = self.sender()
        row = index.row()
        sym_item = table.item(row, 0)
        tf_item  = table.item(row, 2)
        if sym_item and tf_item:
            self.symbol_selected.emit(sym_item.text(), tf_item.text())
