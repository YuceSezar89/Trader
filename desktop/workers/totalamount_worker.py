"""
TotalamountWorker — Devisso Döngüsü / Rank sekmesi (13 Ağu 2026).

live_data_manager.py::_totalamount_rank1_loop'un yayınladığı aday sıralamasını
(`totalamount_rank1:snapshot` — {symbol, value, rank}) okur, backend bunu
90sn'de bir (_TOTALAMOUNT_RANK1_INTERVAL) yeniliyor.

13 Ağu 2026, ÜÇÜNCÜ düzeltme — grafik (ve onun ihtiyacı olan
`totalamount_live:{symbol}:5m` seri verisi + MGET okuması) tamamen kaldırıldı,
bkz. desktop/panels/rank_panel.py modül docstring'i (kanıtlanmış tekrar eden
PC kilitlenmesi, kök neden Qt/pyqtgraph render pipeline'ında, tam izole
edilemedi — kullanıcı kararıyla grafik kaldırıldı). Tablo için snapshot'taki
value/rank zaten yeterli, ayrı bir seri okumaya gerek yok — worker artık
sadece TEK bir Redis GET yapıyor (300 sembol için MGET bile gereksizleşti).
threading.Event ile aynı anda stop() çağrılırsa döngü beklemeden çıkar
(divergence_worker.py / live_metrics_worker.py ile aynı desen).
"""

import json
import logging
import threading
import time
from typing import Optional

import redis
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_POLL_SEC = 90  # live_data_manager.py::_TOTALAMOUNT_RANK1_INTERVAL ile aynı kademe


class TotalamountWorker(QThread):
    """totalamount_rank1 aday havuzunun sıralama snapshot'ını okuyan worker."""

    totalamount_updated = pyqtSignal(object)  # dict
    status_updated = pyqtSignal(str)

    def __init__(self, redis_url: str, interval_sec: int = _POLL_SEC, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._interval_sec = interval_sec
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._wake = threading.Event()

    def run(self) -> None:
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=False, socket_connect_timeout=3
            )
            self._redis.ping()
        except redis.RedisError as exc:
            self.status_updated.emit(f"Redis bağlanamadı: {exc}")
            return

        while self._running:
            try:
                result = self._compute()
                if result:
                    self.totalamount_updated.emit(result)
                    self.status_updated.emit(
                        f"{len(result['current'])} aday  •  {time.strftime('%H:%M:%S')}"
                    )
                else:
                    self.status_updated.emit("Aday bekleniyor…")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Totalamount hesaplama hatası: %s", exc, exc_info=True)
            self._wake.wait(timeout=self._interval_sec)
            self._wake.clear()

    def _compute(self) -> Optional[dict]:
        raw_snap = self._redis.get("totalamount_rank1:snapshot")
        if not raw_snap:
            return None
        try:
            snapshot = json.loads(raw_snap)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if not snapshot:
            return None

        current = {row["symbol"]: row["value"] for row in snapshot}
        ranking = {row["symbol"]: row["rank"] for row in snapshot}
        return {"current": current, "ranking": ranking}

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()
