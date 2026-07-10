"""
Asyncio-native Binance WebSocket bağlantı yöneticisi (websockets kütüphanesi).

binance-connector-python'ın thread-tabanlı UMFuturesWebsocketClient'ının yerini
alan alternatif taşıma katmanı — her bağlantı kendi OS thread'i (BinanceSocketManager
(threading.Thread), senkron websocket-client kütüphanesi) yerine, AYNI event loop
içinde bir asyncio task'tır. N bağlantı = N task (GIL çekişmesi yaratmaz), N thread
değil.

Gölge testlerle doğrulandı (scripts/ws_shadow_test.py, 10 Tem 2026):
150 sembol × 10 TF = 1500 stream, 8 bağlantı, 40 dakika, 789K mesaj, 0 hata,
0 reconnect, thread sayısı test boyunca SABİT (production'daki gibi bağlantı
başına artmadı), production'la %99.8+ doğruluk (kalan ~%0.2 fark production'ın
kendi geçici yazma gecikmesinden — ayrı, bilinen bir konu, bkz. memory
project_pc_power_loss_10temmuz.md / project_data_layer_debt.md).

Kullanım: LiveDataManager, Config.WS_BACKEND=="asyncio" olduğunda bu modülü
kullanır (bkz. live_data_manager.py::start_streams). Varsayılan "thread" —
mevcut davranış AYNEN korunur, flag açıkça değiştirilmeden hiçbir şey değişmez.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)

DEFAULT_PING_INTERVAL = 20.0
# 10s→30s (10 Tem 2026, saat sınırı vakası): tüm WS bağlantıları AYNI event loop'ta
# olduğu için, burst anında (1m+5m+15m+30m+1h hepsi aynı anda kapanınca) event loop
# birkaç saniye meşgul kalabiliyor — kütüphanenin kendi ping'ine pong zamanında
# işlenemeyip YANLIŞ POZİTİF "bağlantı koptu" sanılıyordu (17:00'da 18+ bağlantı
# aynı 4 saniyede koptu). 30s, gerçek bir ağ kopmasını hâlâ makul sürede yakalar
# ama event loop'un kısa süreli meşguliyetine tahammül eder.
DEFAULT_PING_TIMEOUT = 30.0


class AsyncioBinanceStream:
    """Tek bir Binance combined-stream WebSocket bağlantısı — asyncio task
    olarak çalışır. subscribe, reconnect+backoff, health tracking içerir."""

    def __init__(
        self,
        connection_id: int,
        streams: List[str],
        base_url: str,
        on_message: Callable[["AsyncioBinanceStream", str], None],
        ping_interval: float = DEFAULT_PING_INTERVAL,
        ping_timeout: float = DEFAULT_PING_TIMEOUT,
    ):
        self.connection_id = connection_id
        self.streams = streams
        self.url = f"{base_url}/stream"
        self.on_message = on_message
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self.last_message_time: float = 0.0
        self.is_connected = False
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run(), name=f"ws_asyncio_conn_{self.connection_id}")
        return self._task

    async def _run(self) -> None:
        reconnect_delay = 1.0
        while not self._stop:
            try:
                async with websockets.connect(
                    self.url, ping_interval=self.ping_interval, ping_timeout=self.ping_timeout
                ) as ws:
                    await ws.send(json.dumps({
                        "method": "SUBSCRIBE",
                        "params": self.streams,
                        "id": self.connection_id,
                    }))
                    self.is_connected = True
                    self.last_message_time = time.time()
                    logger.info(
                        "[AsyncioWS #%d] Bağlandı, %d stream'e subscribe olundu.",
                        self.connection_id, len(self.streams),
                    )
                    reconnect_delay = 1.0

                    async for raw_msg in ws:
                        self.last_message_time = time.time()
                        try:
                            self.on_message(self, raw_msg)
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            logger.error(
                                "[AsyncioWS #%d] Mesaj işleme hatası: %s",
                                self.connection_id, exc, exc_info=True,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.is_connected = False
                logger.warning(
                    "[AsyncioWS #%d] Bağlantı koptu (%s), %.0fs sonra tekrar denenecek",
                    self.connection_id, exc, reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    async def stop(self) -> None:
        self._stop = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-exception-caught
                pass


class AsyncioBinanceStreamManager:
    """Stream listesini (symbol@kline_tf) max_streams_per_connection'a göre
    connection'lara böler, her biri ayrı AsyncioBinanceStream (asyncio task)
    olarak başlatılır — production'ın thread-per-connection chunking mantığıyla
    birebir aynı, sadece taşıma katmanı thread yerine task."""

    def __init__(
        self,
        base_url: str,
        on_message: Callable[[AsyncioBinanceStream, str], None],
        max_streams_per_connection: int = 200,
    ):
        self.base_url = base_url
        self.on_message = on_message
        self.max_streams_per_connection = max_streams_per_connection
        self.connections: Dict[int, AsyncioBinanceStream] = {}

    async def start(self, all_streams: List[str]) -> Dict[int, AsyncioBinanceStream]:
        chunks = [
            all_streams[i:i + self.max_streams_per_connection]
            for i in range(0, len(all_streams), self.max_streams_per_connection)
        ]
        for i, chunk in enumerate(chunks):
            conn_id = i + 1
            conn = AsyncioBinanceStream(
                connection_id=conn_id,
                streams=chunk,
                base_url=self.base_url,
                on_message=self.on_message,
            )
            logger.info("[AsyncioWS] Connection #%d: %d stream başlatılıyor...", conn_id, len(chunk))
            conn.start()
            self.connections[conn_id] = conn
            await asyncio.sleep(0.25)  # production'daki gibi bağlantıları stagger et
        return self.connections

    async def stop(self) -> None:
        for conn in list(self.connections.values()):
            await conn.stop()
        self.connections.clear()
