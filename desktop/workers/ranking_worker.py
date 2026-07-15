"""
RankingWorker — tüm takip edilen coinlerin VPMV sıralamasını gösterir.

14 Tem 2026: backend (live_data_manager.py::_ranking_publish_loop) artık her
~90 saniyede bir TÜM evreni (self.mtf_buffers'tan, Redis round-trip/Arrow
decode YOK) VPMV skoruna göre sıralayıp `ranking:snapshot` Redis key'ine
yazıyor — bu worker artık 500+ sembol × 4 TF için ham kline çekip RSI/ATR/
rolling-normalize'i SIFIRDAN hesaplamıyor (o eski yol, periyodik ~30GB'lık
bellek patlamalarının kaynağıydı — panel "takılı kalıyor" şikayetine yol
açan asıl neden), sadece backend'in yayınladığı hazır sıralamayı okuyor.
"""

import json
import logging
import threading
import time
from typing import Optional

import redis
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_UPDATE_SEC = 30  # artık ucuz bir Redis GET — backend'in yayın hızından bağımsız


class RankingWorker(QThread):
    ranking_updated = pyqtSignal(object)  # list[dict]
    status_updated = pyqtSignal(str)

    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._wake = threading.Event()

    # ------------------------------------------------------------------
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
                if result:
                    self.ranking_updated.emit(result)
                    self.status_updated.emit(
                        f"{len(result)} sembol  •  {time.strftime('%H:%M:%S')}"
                    )
                else:
                    self.status_updated.emit("Veri hesaplanıyor…")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Ranking okuma hatası: %s", exc, exc_info=True)

            self._wake.wait(timeout=_UPDATE_SEC)
            self._wake.clear()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()

    def refresh(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------
    def _read_snapshot(self) -> list:
        raw = self._redis.get("ranking:snapshot")
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:  # pylint: disable=broad-exception-caught
            return []
