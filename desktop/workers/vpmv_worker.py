"""
VpmvWorker — aktif sinyallerin kendi TF'lerinde VPMV serisini hesaplar.

Her sembol için sinyal barından itibaren VPMV (0-100) serisi üretir.
Delta = vpmv_now - vpmv_signal → pozitif: momentum devam ediyor, negatif: söndü.
"""

import logging
import threading
import time
from datetime import datetime
from io import StringIO
from typing import Optional

import pandas as pd
import redis
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot  # pylint: disable=no-name-in-module

from utils.vpmv import compute_series

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_MIN_BARS    = 20


def _to_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _find_signal_bar(df: pd.DataFrame, opened_at: datetime) -> Optional[int]:
    """Sinyal zamanına en yakın bar indeksini döner."""
    col = next((c for c in ("open_time", "timestamp") if c in df.columns), None)
    if col is None:
        return None
    times = df[col]
    try:
        if pd.api.types.is_integer_dtype(times):
            target = opened_at.timestamp() * 1000
        else:
            target = pd.Timestamp(opened_at)
        pos = int((times - target).abs().idxmin())
        return df.index.get_loc(pos)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


class VpmvWorker(QThread):
    """Aktif sinyaller için post-sinyal VPMV serisini hesaplayan worker."""

    vpmv_updated   = pyqtSignal(object)  # dict
    status_updated = pyqtSignal(str)

    def __init__(
        self,
        redis_url: str,
        interval_ms: int = 30_000,
        parent=None,
    ):
        super().__init__(parent)
        self._redis_url   = redis_url
        self._interval_ms = interval_ms
        self._running     = False
        self._redis: Optional[redis.Redis] = None
        self._lock        = threading.Lock()
        self._wake        = threading.Event()
        self._signals: dict[int, dict] = {}   # signal_id → signal dict

    @pyqtSlot(list)
    def set_symbols(self, signals: list) -> None:
        with self._lock:
            self._signals = {}
            for s in signals:
                if isinstance(s, dict) and "id" in s:
                    self._signals[int(s["id"])] = s
        self._wake.set()

    @pyqtSlot(dict)
    def add_symbol(self, signal: dict) -> None:
        if not isinstance(signal, dict) or "id" not in signal:
            return
        with self._lock:
            self._signals[int(signal["id"])] = signal
        self._wake.set()

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

        _last_compute = 0.0
        _MIN_INTERVAL = 10.0

        while self._running:
            self._wake.wait(timeout=self._interval_ms / 1000)
            self._wake.clear()
            if not self._running:
                break

            now = time.monotonic()
            if now - _last_compute < _MIN_INTERVAL:
                continue

            with self._lock:
                signals = dict(self._signals)

            if not signals:
                self.status_updated.emit("Sinyal bekleniyor…")
                continue

            try:
                result = self._compute(signals)
                _last_compute = time.monotonic()
                if result:
                    self.vpmv_updated.emit(result)
                    self.status_updated.emit(
                        f"{len(result['current'])} sembol  •  VPMV"
                    )
                else:
                    self.status_updated.emit("Veri yok")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("VPMV hesaplama hatası: %s", exc, exc_info=True)

    def _compute(self, signals: dict) -> Optional[dict]:  # pylint: disable=too-many-locals
        series:         dict = {}
        current_vpmv:   dict = {}
        signal_vpmv:    dict = {}
        pre_vpmv:       dict = {}
        indicators_map: dict = {}
        tf_map:         dict = {}
        time_map:       dict = {}

        for sig in signals.values():
            symbol     = sig.get("symbol") or ""
            sig_type   = sig.get("signal_type") or "Long"
            opened_at  = _to_dt(sig.get("opened_at"))
            vpms_score = sig.get("vpms_score") or 0.0
            vpmv_pre   = sig.get("vpmv_pre_avg") or 0.0
            interval   = sig.get("interval") or "1h"
            display    = f"{symbol} ({interval})"

            redis_key = f"live_kline_data:{symbol}:{interval}".encode()
            df = self._fetch_klines(redis_key)
            if df is None or len(df) < _MIN_BARS:
                continue

            try:
                vpmv_ser = compute_series(df, sig_type)
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            sig_bar_idx = None
            if opened_at:
                sig_bar_idx = _find_signal_bar(df, opened_at)
            if sig_bar_idx is None:
                sig_bar_idx = max(0, len(vpmv_ser) - 50)

            post = vpmv_ser.iloc[sig_bar_idx:].values
            if len(post) == 0:
                continue

            v_now    = float(post[-1])
            v_signal = float(vpms_score) if vpms_score else float(vpmv_ser.iloc[sig_bar_idx])

            series[display]         = post
            current_vpmv[display]   = v_now
            signal_vpmv[display]    = v_signal
            pre_vpmv[display]       = vpmv_pre
            indicators_map[display] = sig.get("indicators") or ""
            tf_map[display]         = interval
            time_map[display]       = opened_at

        if not series:
            return None

        delta = {sym: val - signal_vpmv.get(sym, 0.0) for sym, val in current_vpmv.items()}

        return {
            "series":     series,
            "current":    current_vpmv,
            "signal":     signal_vpmv,
            "pre":        pre_vpmv,
            "delta":      delta,
            "indicators": indicators_map,
            "tf":         tf_map,
            "time":       time_map,
        }

    def _fetch_klines(self, key: bytes) -> Optional[pd.DataFrame]:
        raw = self._redis.get(key)
        if not raw:
            return None
        try:
            if raw[:4] == _ARROW_MAGIC:
                import pyarrow as pa  # pylint: disable=import-outside-toplevel
                reader = pa.ipc.open_stream(raw[4:])
                return reader.read_pandas()
            return pd.read_json(StringIO(raw.decode()), orient="split")
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()
