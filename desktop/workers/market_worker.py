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
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_FALLBACK_MS = 3_000
_SYMBOL_DISCOVERY_INTERVAL = 60  # saniye — sembol evreni (yeni listing/delisting) bu kadar seyrek değişir
_FALLBACK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


class MarketWorker(QThread):  # pylint: disable=too-many-instance-attributes
    """Redis pub/sub ile anlık fiyat ve kline güncellemelerini yayınlar."""

    price_updated = pyqtSignal(str, float, float, float, float)  # symbol, price, change_pct, volume, funding_rate
    prices_updated = pyqtSignal(dict)                       # {symbol: canlı fiyat} — tek toplu güncelleme
    klines_updated = pyqtSignal(str, str, object)           # symbol, timeframe, DataFrame
    connection_changed = pyqtSignal(bool, str)              # connected, message
    symbols_discovered = pyqtSignal(list)                   # sembol evreni (yeniden) keşfedildiğinde

    def __init__(self, redis_url: str, interval_ms: int = _FALLBACK_MS, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._interval_ms = interval_ms
        self._running = False
        self._redis: Optional[redis.Redis] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._watched_symbols: list[str] = []
        self._chart_symbol: str = ""
        self._chart_tf: str = ""
        self._last_symbol_discovery: float = 0.0

    def request_symbol_refresh(self) -> None:
        """Manuel yenile — sıradaki poll döngüsünde sembol evrenini yeniden keşfeder."""
        self._last_symbol_discovery = 0.0
        self._wake.set()

    def set_chart_watch(self, symbol: str, tf: str) -> None:
        """Grafik için takip edilecek sembol ve zaman dilimini ayarlar."""
        with self._lock:
            self._chart_symbol = symbol
            self._chart_tf = tf

    def run(self) -> None:
        """Thread başlatır: Redis bağlantısı kurar, pub/sub ve poll döngülerini çalıştırır."""
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=3,
                socket_timeout=5,
            )
            self._redis.ping()
            self.connection_changed.emit(True, "Redis bağlantısı kuruldu")
        except redis.RedisError as exc:
            self.connection_changed.emit(False, f"Redis bağlanamadı: {exc}")
            return

        threading.Thread(target=self._subscribe_loop, daemon=True).start()

        while self._running:
            self._poll()
            self._wake.wait(timeout=self._interval_ms / 1000)
            self._wake.clear()

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
                    self._wake.wait(timeout=5)
            finally:
                try:
                    if pubsub:
                        pubsub.unsubscribe()
                    if sub_redis:
                        sub_redis.close()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    # ── Ticker okuma ──────────────────────────────────────────────────────

    def _fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Redis'ten ticker:{symbol} JSON verisini okur (backend ticker stream'inden)."""
        try:
            raw = self._redis.get(f"ticker:{symbol}")
            if raw:
                return json.loads(raw)
            return None
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    # ── Sembol keşfi ──────────────────────────────────────────────────────

    def _discover_symbols(self) -> None:
        """Sembol evrenini periyodik olarak keşfeder (KEYS değil SCAN — tek seferde
        tüm keyspace'i taramaz, Redis'i bloklamaz). Önceden watchlist_panel.py bunu
        GUI thread'de, timeout'suz bir KEYS çağrısıyla yapıyordu — Redis ara sıra
        yavaşladığında bu, süresiz olarak tüm tıklamaları dondurabiliyordu. Artık
        bu thread'de, zaten var olan bağlantı üzerinden çalışıyor."""
        now = time.time()
        if now - self._last_symbol_discovery < _SYMBOL_DISCOVERY_INTERVAL:
            return
        self._last_symbol_discovery = now

        symbols: list[str] = []
        try:
            keys = [k.decode() for k in self._redis.scan_iter(match="ticker:*", count=500)]
            symbols = sorted(k.split(":", 1)[1] for k in keys)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        if not symbols:
            try:
                keys = [k.decode() for k in self._redis.scan_iter(match="live_kline_data:*:1m", count=500)]
                symbols = sorted(k.split(":")[1] for k in keys if k.count(":") == 2)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        if not symbols:
            symbols = _FALLBACK_SYMBOLS

        with self._lock:
            self._watched_symbols = symbols
        self.symbols_discovered.emit(symbols)

    # ── Update / poll ─────────────────────────────────────────────────────

    def _handle_update(self, symbol: str, tf: str) -> None:
        with self._lock:
            chart_sym = self._chart_symbol
            chart_tf = self._chart_tf

        if symbol == chart_sym and tf == chart_tf:
            df = self._fetch_klines(symbol, tf)
            if df is not None and not df.empty:
                self.klines_updated.emit(symbol, tf, df)

    def _poll(self) -> None:
        self._discover_symbols()

        with self._lock:
            symbols = list(self._watched_symbols)
            chart_sym = self._chart_symbol
            chart_tf = self._chart_tf

        # Canlı tick fiyatları (backend 2s'de bir yazar) — tek GET, tek toplu emit
        try:
            raw = self._redis.get("prices:live")
            if raw:
                self.prices_updated.emit(json.loads(raw))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Ticker verisi (24h change/volume/funding) — tek pipeline'da oku
        if symbols:
            try:
                pipe = self._redis.pipeline(transaction=False)
                for symbol in symbols:
                    pipe.get(f"ticker:{symbol}")
                results = pipe.execute()
            except Exception:  # pylint: disable=broad-exception-caught
                results = [None] * len(symbols)
            for symbol, raw in zip(symbols, results):
                if not raw:
                    continue
                try:
                    ticker = json.loads(raw)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                self.price_updated.emit(
                    symbol,
                    float(ticker.get("price", 0.0)),
                    float(ticker.get("change_pct", 0.0)),
                    float(ticker.get("volume", 0.0)),
                    float(ticker.get("funding_rate", 0.0)),
                )

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
        """Thread'i durdurur ve bitmesini bekler."""
        self._running = False
        self._wake.set()
        self.wait()
