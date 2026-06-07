"""
MarketWorker — Redis'ten canlı fiyat ve kline verisi çeker.
QThread içinde çalışır, Qt sinyalleriyle ana pencereye veri gönderir.
"""

import json
from io import StringIO
from typing import Optional

import pandas as pd
import redis
from PyQt6.QtCore import QThread, pyqtSignal

_ARROW_MAGIC = b"ARDF"


class MarketWorker(QThread):
    """
    Her `interval_ms` milisaniyede bir Redis'i okur.
    Fiyat ve kline güncellemelerini Qt sinyalleriyle yayınlar.
    """

    price_updated = pyqtSignal(str, float, float)   # symbol, price, change_pct
    klines_updated = pyqtSignal(str, str, object)   # symbol, timeframe, DataFrame
    connection_changed = pyqtSignal(bool, str)       # connected, message

    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def __init__(self, redis_url: str, interval_ms: int = 1000, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._interval_ms = interval_ms
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._watched_symbols: list[str] = []

    def set_symbols(self, symbols: list[str]) -> None:
        self._watched_symbols = symbols

    def run(self) -> None:
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=3,
            )
            self._redis.ping()
            self.connection_changed.emit(True, "Redis bağlantısı kuruldu")
        except Exception as exc:
            self.connection_changed.emit(False, f"Redis bağlanamadı: {exc}")
            return

        while self._running:
            self._poll()
            self.msleep(self._interval_ms)

    def _poll(self) -> None:
        if not self._watched_symbols:
            return
        for symbol in self._watched_symbols:
            df_1m = self._fetch_klines(symbol, "1m")
            if df_1m is not None and not df_1m.empty:
                price = float(df_1m["close"].iloc[-1])
                open_price = float(df_1m["open"].iloc[0])
                change_pct = (price - open_price) / open_price * 100 if open_price else 0.0
                self.price_updated.emit(symbol, price, change_pct)
                self.klines_updated.emit(symbol, "1m", df_1m)

    def _fetch_klines(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            key = f"live_kline_data:{symbol}:{timeframe}"
            raw = self._redis.get(key)
            if not raw:
                return None
            if raw[:4] == _ARROW_MAGIC:
                import pyarrow as pa
                reader = pa.ipc.open_stream(raw[4:])
                df = reader.read_pandas()
            else:
                df = pd.read_json(StringIO(raw.decode("utf-8")), orient="split")
            if "open_time" in df.columns and pd.api.types.is_integer_dtype(df["open_time"]):
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            return df
        except Exception:
            return None

    def stop(self) -> None:
        self._running = False
        self.wait()
