"""
SignalWorker — Veritabanından aktif sinyalleri ve yeni sinyal gelişlerini izler.
"""

import threading
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import redis
from PyQt6.QtCore import QThread, pyqtSignal

_SIGNAL_COLUMNS = """id, symbol, signal_type, opened_at,
       open_price, interval, vpms_score, mtf_score,
       alpha, beta, status, indicators,
       st_confirmed, sharpe_ratio, oi_data,
       stop_loss_price, take_profit_price,
       z_score_entry, is_confluence, trailing_stop_price,
       sortino_ratio, calmar_ratio,
       vpmv_pre_avg, vpmv_slope, vpmv_ratio,
       cvd_slope, vp_buy_avg, vp_sell_avg, vp_score,
       devisso_score, devisso_delta, devisso_ratio,
       pd_zone, market_structure, fvg_tfs, candle_pattern, atr"""


class SignalWorker(QThread):
    """
    Her `interval_ms` milisaniyede bir DB'yi sorgular.
    Yeni sinyal ve durum değişikliklerini Qt sinyalleriyle bildirir.
    Bağlantı kalıcıdır; koptuğunda bir sonraki turda yeniden kurulur ve
    durum değişimi connection_changed ile panele bildirilir.
    """

    signals_loaded = pyqtSignal(list)           # aktif sinyal listesi
    new_signal = pyqtSignal(dict)               # tek yeni sinyal
    signals_closed = pyqtSignal(list)           # kapanan sinyal ID listesi
    connection_changed = pyqtSignal(bool, str)  # connected, message

    def __init__(
        self,
        db_config: dict[str, Any],
        interval_ms: int = 5000,
        redis_url: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._db_config = db_config
        self._interval_ms = interval_ms
        self._redis_url = redis_url
        self._running = False
        self._last_signal_id: int = 0
        self._active_ids: set[int] = set()
        self._wake = threading.Event()
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._connected = False
        self._loaded = False

    def run(self) -> None:
        self._running = True
        if self._redis_url:
            threading.Thread(target=self._subscribe_loop, daemon=True).start()
        while self._running:
            if not self._loaded:
                self._load_all()
            else:
                self._check_new()
            self._wake.wait(timeout=self._interval_ms / 1000)
            self._wake.clear()

    def _subscribe_loop(self) -> None:
        """signal_opened kanalını dinler; mesaj gelince poll döngüsünü hemen uyandırır."""
        while self._running:
            sub = None
            pubsub = None
            try:
                sub = redis.Redis.from_url(self._redis_url, decode_responses=True,
                                           socket_connect_timeout=3)
                pubsub = sub.pubsub()
                pubsub.subscribe("signal_opened")
                for message in pubsub.listen():
                    if not self._running:
                        return
                    if message.get("type") == "message":
                        self._wake.set()
            except Exception:
                if self._running:
                    self._wake.wait(timeout=5)
            finally:
                try:
                    if pubsub:
                        pubsub.unsubscribe()
                    if sub:
                        sub.close()
                except Exception:
                    pass

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self._db_config, connect_timeout=5)
            self._conn.autocommit = True  # salt-okunur polling; idle-in-transaction bırakma
        return self._conn

    def _drop_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _set_connected(self, ok: bool, msg: str) -> None:
        if ok != self._connected:
            self._connected = ok
            self.connection_changed.emit(ok, msg)

    def _load_all(self) -> None:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT {_SIGNAL_COLUMNS}
                    FROM signals
                    WHERE status = 'active'
                    ORDER BY opened_at DESC
                    LIMIT 500
                """)
                rows = [dict(r) for r in cur.fetchall()]
                cur.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM signals")
                self._last_signal_id = cur.fetchone()["max_id"]
                self._active_ids = {r["id"] for r in rows}
                self.signals_loaded.emit(rows)
            self._loaded = True
            self._set_connected(True, "DB bağlantısı kuruldu")
        except Exception as exc:
            self._drop_conn()
            self._set_connected(False, f"DB bağlanamadı: {exc}")

    def _check_new(self) -> None:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Yeni sinyaller
                cur.execute(f"""
                    SELECT {_SIGNAL_COLUMNS}
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

            self._set_connected(True, "DB bağlantısı yeniden kuruldu")
        except Exception as exc:
            self._drop_conn()
            self._set_connected(False, f"DB bağlantısı koptu: {exc}")

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()
        self._drop_conn()
