"""
TFAlignmentWorker — signals/tf_alignment_gate.py'nin (signal_service.py
sürecinde çalışır) `tf_alignment:candidates` Redis key'ine yayınladığı
bekleyen adayları okur. 18 Tem 2026.
"""

import json
import logging
import threading
import time
from typing import Optional

import redis
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_UPDATE_SEC = 10


class TFAlignmentWorker(QThread):
    candidates_updated = pyqtSignal(object)  # list[dict]
    status_updated = pyqtSignal(str)

    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._wake = threading.Event()

    def run(self) -> None:
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=True, socket_connect_timeout=3
            )
            self._redis.ping()
        except redis.RedisError as exc:
            self.status_updated.emit(f"Redis bağlanamadı: {exc}")
            return

        while self._running:
            try:
                result = self._read_snapshot()
                self.candidates_updated.emit(result)
                self.status_updated.emit(
                    f"{len(result)} aday  •  {time.strftime('%H:%M:%S')}"
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Aday okuma hatası: %s", exc, exc_info=True)

            self._wake.wait(timeout=_UPDATE_SEC)
            self._wake.clear()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()

    def refresh(self) -> None:
        self._wake.set()

    def _read_snapshot(self) -> list:
        raw = self._redis.get("tf_alignment:candidates")
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:  # pylint: disable=broad-exception-caught
            return []
