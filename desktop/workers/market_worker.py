"""
MarketWorker — Redis pub/sub ile anlık fiyat ve kline güncellemelerini yayınlar.
QThread içinde çalışır, Qt sinyalleriyle ana pencereye veri gönderir.
"""

import json
import logging
import threading
import time
from io import StringIO
from typing import Optional

import pandas as pd
import redis
import websocket  # websocket-client
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_FALLBACK_MS = 30_000
_MINI_TICKER_URL = "wss://fstream.binance.com/ws/!miniTicker@arr"


class MarketWorker(QThread):  # pylint: disable=too-many-instance-attributes
    """Redis pub/sub ile anlık fiyat ve kline güncellemelerini yayınlar."""

    price_updated = pyqtSignal(str, float, float, float)   # symbol, price, change_pct, volume
    klines_updated = pyqtSignal(str, str, object)           # symbol, timeframe, DataFrame
    connection_changed = pyqtSignal(bool, str)              # connected, message

    def __init__(self, redis_url: str, interval_ms: int = _FALLBACK_MS, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._interval_ms = interval_ms
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._lock = threading.Lock()
        self._watched_symbols: list[str] = []
        self._chart_symbol: str = ""
        self._chart_tf: str = ""
        self._change_cache: dict[str, float] = {}   # symbol → 24h priceChangePercent

    def set_symbols(self, symbols: list[str]) -> None:
        with self._lock:
            self._watched_symbols = list(symbols)

    def set_chart_watch(self, symbol: str, tf: str) -> None:
        with self._lock:
            self._chart_symbol = symbol
            self._chart_tf = tf

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
        except redis.RedisError as exc:
            self.connection_changed.emit(False, f"Redis bağlanamadı: {exc}")
            return

        threading.Thread(target=self._subscribe_loop, daemon=True).start()
        threading.Thread(target=self._mini_ticker_loop, daemon=True).start()

        while self._running:
            self._poll()
            self.msleep(self._interval_ms)

    # ── Binance miniTicker WebSocket ──────────────────────────────────────

    def _mini_ticker_loop(self) -> None:
        """!miniTicker@arr stream — her saniye tüm sembollerin 24h değişimi."""
        while self._running:
            try:
                ws = websocket.create_connection(_MINI_TICKER_URL, timeout=10)
                while self._running:
                    raw = ws.recv()
                    if not raw:
                        continue
                    items = json.loads(raw)
                    if not isinstance(items, list):
                        continue
                    with self._lock:
                        for item in items:
                            sym = item.get("s", "")
                            pct = item.get("P")
                            if sym and pct is not None:
                                self._change_cache[sym] = float(pct)
                ws.close()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("miniTicker WS kesildi: %s — 5s sonra yeniden bağlanılıyor", exc)
                if self._running:
                    time.sleep(5)

    # ── Redis kline pub/sub ───────────────────────────────────────────────

    def _subscribe_loop(self) -> None:
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
                    self._handle_update(symbol, tf)
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

    def _handle_update(self, symbol: str, tf: str) -> None:
        with self._lock:
            watched = list(self._watched_symbols)
            chart_sym = self._chart_symbol
            chart_tf = self._chart_tf
            change_pct = self._change_cache.get(symbol)

        if tf == "1m" and symbol in watched:
            df = self._fetch_klines(symbol, "1m")
            if df is not None and not df.empty:
                price = float(df["close"].iloc[-1])
                volume = float(df["volume"].sum()) if "volume" in df.columns else 0.0
                if change_pct is None:
                    open_price = float(df["open"].iloc[0])
                    change_pct = (price - open_price) / open_price * 100 if open_price else 0.0
                self.price_updated.emit(symbol, price, change_pct, volume)
                self.klines_updated.emit(symbol, "1m", df)

        if symbol == chart_sym and tf == chart_tf:
            df = self._fetch_klines(symbol, tf)
            if df is not None and not df.empty:
                self.klines_updated.emit(symbol, tf, df)

    def _poll(self) -> None:
        with self._lock:
            symbols = list(self._watched_symbols)
            chart_sym = self._chart_symbol
            chart_tf = self._chart_tf

        for symbol in symbols:
            df_1m = self._fetch_klines(symbol, "1m")
            if df_1m is not None and not df_1m.empty:
                price = float(df_1m["close"].iloc[-1])
                volume = float(df_1m["volume"].sum()) if "volume" in df_1m.columns else 0.0
                with self._lock:
                    change_pct = self._change_cache.get(symbol)
                if change_pct is None:
                    open_price = float(df_1m["open"].iloc[0])
                    change_pct = (price - open_price) / open_price * 100 if open_price else 0.0
                self.price_updated.emit(symbol, price, change_pct, volume)

        if chart_sym:
            df = self._fetch_klines(chart_sym, chart_tf)
            if df is not None and not df.empty:
                self.klines_updated.emit(chart_sym, chart_tf, df)

    def _fetch_klines(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            key = f"live_kline_data:{symbol}:{timeframe}".encode()
            raw = self._redis.get(key)
            if not raw:
                return None
            if raw[:4] == _ARROW_MAGIC:
                import pyarrow as pa  # pylint: disable=import-outside-toplevel
                reader = pa.ipc.open_stream(raw[4:])
                df = reader.read_pandas()
            else:
                df = pd.read_json(StringIO(raw.decode("utf-8")), orient="split")
            if "open_time" in df.columns and pd.api.types.is_integer_dtype(df["open_time"]):
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            return df
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def stop(self) -> None:
        self._running = False
        self.wait()
