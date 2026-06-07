"""
HealthWorker — Redis, DB ve WebSocket servis sağlığını izler.
"""

from typing import Any

import psycopg2
import redis
from PyQt6.QtCore import QThread, pyqtSignal


class ServiceStatus:
    OK = "ok"
    ERROR = "error"
    UNKNOWN = "unknown"


class HealthWorker(QThread):
    """Her 10 saniyede bir servis sağlığını kontrol eder."""

    health_updated = pyqtSignal(dict)  # {"redis": ok/error, "db": ok/error, "symbols": int}

    def __init__(
        self,
        redis_url: str,
        db_config: dict[str, Any],
        interval_ms: int = 10_000,
        parent=None,
    ):
        super().__init__(parent)
        self._redis_url = redis_url
        self._db_config = db_config
        self._interval_ms = interval_ms
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            self.health_updated.emit(self._check())
            self.msleep(self._interval_ms)

    def _check(self) -> dict:
        result = {
            "redis": ServiceStatus.UNKNOWN,
            "db": ServiceStatus.UNKNOWN,
            "symbol_count": 0,
            "active_signals": 0,
        }
        try:
            r = redis.Redis.from_url(self._redis_url, socket_connect_timeout=2)
            r.ping()
            result["redis"] = ServiceStatus.OK
            keys = r.keys("live_kline_data:*:1m")
            result["symbol_count"] = len(keys)
        except Exception:
            result["redis"] = ServiceStatus.ERROR

        try:
            conn = psycopg2.connect(**self._db_config, connect_timeout=2)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM signals WHERE status = 'active'")
                result["active_signals"] = cur.fetchone()[0]
            conn.close()
            result["db"] = ServiceStatus.OK
        except Exception:
            result["db"] = ServiceStatus.ERROR

        return result

    def stop(self) -> None:
        self._running = False
        self.wait()
