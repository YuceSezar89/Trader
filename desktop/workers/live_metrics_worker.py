"""
LiveMetricsWorker — Aktif Sinyaller tablosundaki canlı metrikleri (Güç
Sıralaması, VPMV Ayrışması, Ayrışma Z-score, TF Hizalanma, Verimlilik) tek
worker'da toplayıp okur (27 Tem 2026, kullanıcı isteği).

Kaynaklar (hepsi backend'in zaten doldurduğu, "backend hesaplar, UI okur"
mimarisindeki Redis key'leri — bkz. live_data_manager.py):
  - ranking:snapshot                     tüm evren, ~90s     {symbol: rank_score}
  - ha_alignment:snapshot                tüm evren, ~90-120s {symbol: {...}}
  - vpmv_live:{signal_id}                sinyalin kendi TF'i, bar-kapanışı
  - devisso_live:{signal_id}             sinyalin kendi TF'i, bar-kapanışı
  - divergence_live:{symbol}:{interval}  sembol, herhangi bir TF bar-kapanışı

Aktif sinyal sayısı ~2000'e kadar çıkabildiği için sinyal-bazlı okumalar
MGET ile toplu yapılır (VpmvWorker'daki sıralı-GET deseni küçük, seçili bir
alt küme için tasarlanmıştı — bu worker TÜM aktif sinyaller için çalıştığından
o desen burada ölçek sorunlarına yol açar).
"""

import json
import logging
import threading
from typing import Optional

import redis
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_UPDATE_SEC = 5


class LiveMetricsWorker(QThread):
    metrics_updated = pyqtSignal(dict)

    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._signals: dict[int, dict] = {}  # signal_id → {"symbol":.., "interval":..}

    @pyqtSlot(list)
    def set_active_signals(self, signals: list) -> None:
        """signals: [{"id":.., "symbol":.., "interval":..}, ...] — VpmvWorker
        ile aynı desen (set_symbols), ilk yüklemede/tam yenilemede kullanılır."""
        with self._lock:
            self._signals = {
                int(s["id"]): s for s in signals if s.get("id") is not None
            }
        self._wake.set()

    @pyqtSlot(dict)
    def add_signal(self, signal: dict) -> None:
        if signal.get("id") is None:
            return
        with self._lock:
            self._signals[int(signal["id"])] = signal
        self._wake.set()

    @pyqtSlot(list)
    def remove_signals(self, signal_ids: list) -> None:
        with self._lock:
            for sid in signal_ids:
                self._signals.pop(int(sid), None)
        self._wake.set()

    def run(self) -> None:
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=True, socket_connect_timeout=3
            )
            self._redis.ping()
        except redis.RedisError as exc:
            logger.error("[LiveMetrics] Redis bağlanamadı: %s", exc)
            return

        while self._running:
            try:
                result = self._read_all()
                self.metrics_updated.emit(result)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("[LiveMetrics] okuma hatası: %s", exc, exc_info=True)
            self._wake.wait(timeout=_UPDATE_SEC)
            self._wake.clear()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()

    # ------------------------------------------------------------------
    def _read_all(self) -> dict:
        ranking = self._read_snapshot_field("ranking:snapshot", "rank_score")
        ha = self._read_snapshot_full("ha_alignment:snapshot")

        with self._lock:
            signals = dict(self._signals)
        signal_ids = list(signals.keys())
        sym_intervals = list(
            {
                (s["symbol"], s["interval"])
                for s in signals.values()
                if s.get("symbol") and s.get("interval")
            }
        )

        devisso = self._mget_field(
            [f"devisso_live:{sid}" for sid in signal_ids], signal_ids, "devisso"
        )
        vpmv = self._mget_field([f"vpmv_live:{sid}" for sid in signal_ids], signal_ids, "vpmv")
        divergence = self._mget_field(
            [f"divergence_live:{sym}:{iv}" for sym, iv in sym_intervals],
            sym_intervals,
            "z_now",
        )

        return {
            "ranking": ranking,
            "ha": ha,
            "devisso": devisso,
            "vpmv": vpmv,
            "divergence": divergence,
        }

    def _read_snapshot_field(self, key: str, field: str) -> dict:
        raw = self._redis.get(key)
        if not raw:
            return {}
        try:
            items = json.loads(raw)
            return {item["symbol"]: item.get(field) for item in items if "symbol" in item}
        except Exception:  # pylint: disable=broad-exception-caught
            return {}

    def _read_snapshot_full(self, key: str) -> dict:
        raw = self._redis.get(key)
        if not raw:
            return {}
        try:
            items = json.loads(raw)
            return {item["symbol"]: item for item in items if "symbol" in item}
        except Exception:  # pylint: disable=broad-exception-caught
            return {}

    def _mget_field(self, keys: list, ids: list, field: str) -> dict:
        if not keys:
            return {}
        try:
            values = self._redis.mget(keys)
        except redis.RedisError as exc:
            logger.debug("[LiveMetrics] MGET hatası: %s", exc)
            return {}
        result = {}
        for key_id, raw in zip(ids, values):
            if not raw:
                continue
            try:
                data = json.loads(raw)
                result[key_id] = data.get(field)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return result
