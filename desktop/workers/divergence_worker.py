"""
DivergenceWorker — sinyal gelen coinlerde fiyat Z-score zaman serisi hesaplar.
Her coinin kendi EMA/StdDev'ine göre: z = (close - EMA) / StdDev
Ters sinyal geldiğinde o anki z-score offset olarak alınır → çizgi 0'a döner.
"""

import logging
import threading
import time
from io import StringIO
from typing import Optional

import pandas as pd
import redis
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_EMA_PERIOD = 200
_MIN_BARS = 30
_SERIES_LEN = 100


def _direction(signal_type: str) -> str:
    return "short" if "short" in signal_type.lower() else "long"


def _extract_timestamps(df: pd.DataFrame, length: int):
    """open_time veya timestamp kolonundan Unix saniye dizisi döner."""
    col = next((c for c in ("open_time", "timestamp") if c in df.columns), None)
    if col is None:
        return None
    raw = df[col].values[-length:]
    if pd.api.types.is_integer_dtype(df[col]):
        return raw / 1000.0
    return pd.to_datetime(raw).astype("int64").values / 1e9


def _diverge_start(z_slice, ts) -> Optional[float]:
    """Son ayrışmanın başladığı timestamp'ı döner (|z| < 0.5 son geçişten sonraki an)."""
    for i in range(len(z_slice) - 2, -1, -1):
        if abs(z_slice[i]) < 0.5:
            return float(ts[min(i + 1, len(ts) - 1)])
    return float(ts[0]) if len(ts) > 0 else None


class DivergenceWorker(QThread):  # pylint: disable=too-many-instance-attributes
    """Sinyal coinlerinin Z-score zaman serilerini Redis'ten hesaplayan worker."""

    divergence_updated = pyqtSignal(object)  # dict
    status_updated = pyqtSignal(str)

    def __init__(
        self,
        redis_url: str,
        timeframe: str = "1h",
        interval_ms: int = 30_000,
        parent=None,
    ):
        super().__init__(parent)
        self._redis_url = redis_url
        self._timeframe = timeframe
        self._interval_ms = interval_ms
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()

        self._symbols: set[str] = set()
        self._directions: dict[str, str] = {}   # symbol → "long" | "short"
        self._offsets: dict[str, float] = {}    # symbol → z-score at reset
        self._pending_resets: set[str] = set()  # sıradaki hesaplamada offset alınacak

    def set_timeframe(self, tf: str) -> None:
        """Aktif zaman dilimini günceller ve hemen yeniden hesaplar."""
        self._timeframe = tf
        self._wake.set()

    @pyqtSlot(list)
    def set_symbols(self, signals: list) -> None:
        """Aktif sinyallerin tamamını yükler; ilk başlangıç veya yenileme için."""
        with self._lock:
            self._symbols = set()
            for s in signals:
                if not isinstance(s, dict):
                    continue
                sym = s["symbol"]
                self._symbols.add(sym)
                # İlk yükleme: yön kaydedilir, reset yapılmaz
                if sym not in self._directions:
                    self._directions[sym] = _direction(s.get("signal_type", ""))
        self._wake.set()

    @pyqtSlot(dict)
    def add_symbol(self, signal: dict) -> None:
        """Yeni sinyal ekler; ters yön geldiyse o sembol için reset planlar."""
        if not isinstance(signal, dict) or "symbol" not in signal:
            return
        sym = signal["symbol"]
        new_dir = _direction(signal.get("signal_type", ""))
        with self._lock:
            self._symbols.add(sym)
            current_dir = self._directions.get(sym)
            if current_dir and current_dir != new_dir:
                # Ters sinyal → bir sonraki hesaplamada offset sıfırlanacak
                self._pending_resets.add(sym)
                logger.info(
                    "Divergence reset: %s  %s → %s", sym, current_dir, new_dir
                )
            self._directions[sym] = new_dir
        self._wake.set()

    def run(self) -> None:
        """Worker döngüsü: Redis'e bağlanır, uyandığında Z-score hesaplar."""
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=False, socket_connect_timeout=3
            )
            self._redis.ping()
        except redis.RedisError as exc:
            self.status_updated.emit(f"Redis bağlanamadı: {exc}")
            return

        sub_thread = threading.Thread(target=self._subscribe_loop, daemon=True)
        sub_thread.start()

        while self._running:
            self._wake.wait(timeout=self._interval_ms / 1000)
            self._wake.clear()

            if not self._running:
                break

            with self._lock:
                symbols = set(self._symbols)
                pending = set(self._pending_resets)
                self._pending_resets.clear()

            if not symbols:
                self.status_updated.emit("Sinyal bekleniyor…")
                continue

            try:
                result = self._compute(symbols, pending)
                if result:
                    self.divergence_updated.emit(result)
                    n = len(result["current"])
                    self.status_updated.emit(
                        f"{n} sembol  •  TF: {self._timeframe}"
                    )
                else:
                    self.status_updated.emit(f"Veri yok ({self._timeframe})")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Divergence hesaplama hatası: %s", exc, exc_info=True)

    def _compute(self, symbols: set, pending_resets: set) -> Optional[dict]:
        series: dict = {}
        current: dict = {}
        timestamps: dict = {}
        diverge_since: dict = {}

        for symbol in symbols:
            key = f"live_kline_data:{symbol}:{self._timeframe}".encode()
            df = self._fetch_klines(key)
            if df is None or len(df) < _MIN_BARS:
                continue
            z = self._zscore_series(df)
            if z is None:
                continue

            z_now = float(z.iloc[-1])

            # Ters sinyal geldiyse: o anki z-score'u yeni offset olarak ayarla
            if symbol in pending_resets:
                with self._lock:
                    self._offsets[symbol] = z_now

            offset = self._offsets.get(symbol, 0.0)
            z_adj = z - offset

            z_slice = z_adj.values[-_SERIES_LEN:]
            series[symbol] = z_slice
            current[symbol] = float(z_adj.iloc[-1])

            ts = _extract_timestamps(df, len(z_slice))
            if ts is not None:
                timestamps[symbol] = ts
                diverge_since[symbol] = _diverge_start(z_slice, ts)

        if not series:
            return None

        return {
            "series": series,
            "current": current,
            "tf": self._timeframe,
            "timestamps": timestamps,
            "diverge_since": diverge_since,
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

    def _zscore_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        try:
            close = df["close"].astype(float).reset_index(drop=True)
            n = min(_EMA_PERIOD, len(close))
            ema = close.ewm(span=n, adjust=False).mean()
            std = close.rolling(n, min_periods=5).std().bfill()
            return (close - ema) / (std + 1e-8)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _subscribe_loop(self) -> None:
        """Redis pub/sub kanalını dinler; ilgili kline güncellemelerinde _wake'i tetikler."""
        while self._running:
            sub_redis = None
            pubsub = None
            try:
                sub_redis = redis.Redis.from_url(
                    self._redis_url, decode_responses=True, socket_connect_timeout=3
                )
                pubsub = sub_redis.pubsub()
                pubsub.subscribe("kline_updated")
                for message in pubsub.listen():
                    if not self._running:
                        return
                    if message["type"] != "message":
                        continue
                    data = message.get("data", "")
                    if ":" not in data:
                        continue
                    symbol, tf = data.rsplit(":", 1)
                    with self._lock:
                        relevant = symbol in self._symbols and tf == self._timeframe
                    if relevant:
                        self._wake.set()
            except redis.RedisError as exc:
                logger.warning("Pub/sub kesildi: %s — 5s sonra yeniden bağlanılıyor", exc)
                if self._running:
                    time.sleep(5)
            finally:
                try:
                    if pubsub:
                        pubsub.unsubscribe()
                    if sub_redis:
                        sub_redis.close()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    def stop(self) -> None:
        """Worker thread'i durdurur."""
        self._running = False
        self._wake.set()
        self.wait()
