"""
SignalWorker — Veritabanından aktif sinyalleri ve yeni sinyal gelişlerini izler.
"""

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
            self.msleep(self._interval_ms)
            self._check_new()

    def _connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(**self._db_config)

    def _load_all(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, symbol, signal_type, timestamp,
                           price, interval, vpms_score, vpms_mtf_score,
                           alpha, beta, zscore_ratio_percent, status, indicators
                    FROM signals
                    WHERE status = 'active'
                    ORDER BY timestamp DESC
                    LIMIT 500
                """)
                rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    self._last_signal_id = max(r["id"] for r in rows)
                self.signals_loaded.emit(rows)
            conn.close()
        except Exception:
            pass

    def _check_new(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, symbol, signal_type, timestamp,
                           price, interval, vpms_score, vpms_mtf_score,
                           alpha, beta, zscore_ratio_percent, status, indicators
                    FROM signals
                    WHERE id > %s
                    ORDER BY id ASC
                """, (self._last_signal_id,))
                rows = cur.fetchall()
                for row in rows:
                    d = dict(row)
                    self._last_signal_id = max(self._last_signal_id, d["id"])
                    self.new_signal.emit(d)
            conn.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        self.wait()
