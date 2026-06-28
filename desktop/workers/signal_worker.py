"""
SignalWorker — Veritabanından aktif sinyalleri ve yeni sinyal gelişlerini izler.
"""

import threading
from typing import Any

import psycopg2
import psycopg2.extras
from PyQt6.QtCore import QThread, pyqtSignal


class SignalWorker(QThread):
    """
    Her `interval_ms` milisaniyede bir DB'yi sorgular.
    Yeni sinyal ve durum değişikliklerini Qt sinyalleriyle bildirir.
    """

    signals_loaded = pyqtSignal(list)           # aktif sinyal listesi
    new_signal = pyqtSignal(dict)               # tek yeni sinyal
    signals_closed = pyqtSignal(list)           # kapanan sinyal ID listesi
    connection_changed = pyqtSignal(bool, str)  # connected, message

    def __init__(
        self,
        db_config: dict[str, Any],
        interval_ms: int = 5000,
        parent=None,
    ):
        super().__init__(parent)
        self._db_config = db_config
        self._interval_ms = interval_ms
        self._running = False
        self._last_signal_id: int = 0
        self._active_ids: set[int] = set()
        self._wake = threading.Event()

    def run(self) -> None:
        self._running = True
        try:
            conn = self._connect()
            conn.close()
            self.connection_changed.emit(True, "DB bağlantısı kuruldu")
        except Exception as exc:
            self.connection_changed.emit(False, f"DB bağlanamadı: {exc}")
            return

        self._load_all()
        while self._running:
            self._wake.wait(timeout=self._interval_ms / 1000)
            self._wake.clear()
            if not self._running:
                break
            self._check_new()

    def _connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(**self._db_config)

    def _load_all(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, symbol, signal_type, opened_at,
                           open_price, interval, vpms_score, mtf_score,
                           alpha, beta, status, indicators,
                           st_confirmed, sharpe_ratio, oi_data,
                           stop_loss_price, take_profit_price,
                           z_score_entry, is_confluence, trailing_stop_price,
                           sortino_ratio, calmar_ratio,
                           vpmv_pre_avg, vpmv_slope, vpmv_ratio,
                           cvd_slope, vp_buy_avg, vp_sell_avg, vp_score,
                           deviso_score, deviso_delta
                    FROM signals
                    WHERE status = 'active'
                    ORDER BY opened_at DESC
                    LIMIT 500
                """)
                rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    self._last_signal_id = max(r["id"] for r in rows)
                self._active_ids = {r["id"] for r in rows}
                self.signals_loaded.emit(rows)
            conn.close()
        except Exception:
            pass

    def _check_new(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Yeni sinyaller
                cur.execute("""
                    SELECT id, symbol, signal_type, opened_at,
                           open_price, interval, vpms_score, mtf_score,
                           alpha, beta, status, indicators,
                           st_confirmed, sharpe_ratio, oi_data,
                           stop_loss_price, take_profit_price,
                           z_score_entry, is_confluence, trailing_stop_price,
                           sortino_ratio, calmar_ratio,
                           vpmv_pre_avg, vpmv_slope, vpmv_ratio,
                           cvd_slope, vp_buy_avg, vp_sell_avg, vp_score,
                           deviso_score, deviso_delta
                    FROM signals
                    WHERE id > %s
                    ORDER BY id ASC
                """, (self._last_signal_id,))
                for row in cur.fetchall():
                    d = dict(row)
                    self._last_signal_id = max(self._last_signal_id, d["id"])
                    if d["status"] == "active":
                        self._active_ids.add(d["id"])
                    self.new_signal.emit(d)

                # Kapanan sinyaller
                if self._active_ids:
                    cur.execute("""
                        SELECT id FROM signals
                        WHERE id = ANY(%s) AND status != 'active'
                    """, (list(self._active_ids),))
                    closed = [r["id"] for r in cur.fetchall()]
                    if closed:
                        self._active_ids -= set(closed)
                        self.signals_closed.emit(closed)

            conn.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()
