"""
Asyncio-native (websockets kütüphanesi) WS gölge testi — TAM ÖLÇEK (10 TF) testi.

Production'a DOKUNMAZ: ayrı, bağımsız Binance WS bağlantıları açar (production'ın
gerçek sembol listesinden, production'daki TÜM 10 TF için — 1m/5m/15m/30m/1h/4h/
6h/8h/12h/1d), kapanan barları toplar ve production'ın Redis'e yazdığı
live_kline_data ile karşılaştırır.

Amaç: gerçek migrasyonun yapacağı şeyi (çoklu TF, tek connection'da interleaved
mesajlar) küçük bir zaman penceresinde (30-60dk) doğrulamak. Uzun TF'ler (1h+)
bu pencerede muhtemelen hiç kapanmayacak — onlar için sadece mesaj/parse
sağlığı (hata yok) kontrol edilir, kapanış karşılaştırması sadece kapananlar
için yapılır.

Kullanım: .venv/bin/python -m scripts.ws_shadow_test [süre_saniye] [sembol_sayısı]
Örnek:    .venv/bin/python -m scripts.ws_shadow_test 2400 150
"""

import asyncio
import json
import logging
import resource
import sys
import threading
import time

import websockets

from config import Config
from utils.redis_client import RedisClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ws_shadow_test")

DEFAULT_DURATION_SEC = 2400  # 40 dakika — 1m/5m/15m/30m/1h'de en az bir kapanış yakalar
DEFAULT_NUM_SYMBOLS = 150
MAX_STREAMS_PER_CONNECTION = 200  # production'la aynı Binance limiti
LOG_SAMPLE_SIZE = 5  # her kapanan barı loglamak yerine ilk N sembolü örnekle
TIMEFRAMES = list(getattr(Config, "MTF_TIMEFRAMES", ["1m", "5m", "15m"]))


class ShadowStats:
    def __init__(self):
        self.msg_count = 0
        self.errors = 0
        self.reconnects = 0
        self.parse_errors_by_tf: dict[str, int] = {}
        self.closed_bars: dict[tuple[str, str], list[dict]] = {}  # (symbol, tf) -> bars
        self.compare_ok = 0
        self.compare_mismatch = 0
        self.compare_missing = 0


async def fetch_production_symbols(limit: int) -> list[str]:
    """Production'ın şu an gerçekten takip ettiği sembol listesini Redis'ten çeker
    (live_kline_data:{symbol}:1m key'leri) — uydurma bir liste değil, gerçek evren."""
    client = await RedisClient.get_client()
    keys = await client.keys("live_kline_data:*:1m")
    symbols = sorted({
        (k.decode() if isinstance(k, bytes) else k).split(":")[1]
        for k in keys
    })
    logger.info("Production'da toplam %d sembol bulundu, %d tanesi test edilecek", len(symbols), min(limit, len(symbols)))
    return symbols[:limit]


async def run_shadow_connection(streams: list[str], conn_id: int, stats: ShadowStats, log_sample: set[str]):
    """Bağımsız WS bağlantısı: production'ın kullandığı base URL'e, aynı
    SUBSCRIBE mesaj protokolüyle bağlanır (apples-to-apples karşılaştırma).
    streams: 'symbol@kline_tf' formatında, birden çok TF karışık olabilir
    (production'ın combined stream'i de böyle interleaved geliyor)."""
    base = Config.BINANCE_WS_BASE  # örn. wss://fstream.binance.com/market
    url = f"{base}/stream"
    reconnect_delay = 1

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": conn_id,
                }))
                logger.info("[conn#%d] Bağlandı, %d stream'e subscribe olundu.", conn_id, len(streams))
                reconnect_delay = 1

                async for raw_msg in ws:
                    stats.msg_count += 1
                    try:
                        data = json.loads(raw_msg)
                        payload = data.get("data", data)
                        k = payload.get("k")
                        if not k:
                            continue  # subscribe onay mesajı vb.

                        tf = k["i"]
                        if not k["x"]:
                            continue  # sadece kapanan barları izliyoruz

                        symbol = k["s"]
                        stats.closed_bars.setdefault((symbol, tf), []).append({
                            "open_time": k["t"],
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "received_at": time.time(),
                        })
                        if symbol in log_sample:
                            logger.info("[SHADOW] %s %s kapandı: %s", symbol, tf, k["c"])
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        stats.errors += 1
                        logger.error("[conn#%d] Mesaj parse hatası: %s", conn_id, exc)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            stats.reconnects += 1
            logger.warning("[conn#%d] Bağlantı koptu (%s), %ds sonra tekrar denenecek", conn_id, exc, reconnect_delay)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)


async def compare_with_production(symbols: list[str], stats: ShadowStats, log_sample: set[str]):
    """Her dakika, shadow'da kapanan (symbol, tf) barlarını production'ın
    Redis'e yazdığı live_kline_data ile karşılaştırır. Sadece bu pencerede
    GERÇEKTEN kapanmış barlar karşılaştırılabilir (1h+ TF'ler muhtemelen hiç
    kapanmaz — bu normal, 'eksik' değil, sadece veri yok demektir)."""
    await asyncio.sleep(75)
    while True:
        await asyncio.sleep(60)
        round_ok = round_mismatch = round_missing = 0

        for (symbol, tf), bars in list(stats.closed_bars.items()):
            if len(bars) < 2:
                continue
            # DİKKAT: en son kapanan barı DEĞİL, bir önceki turu karşılaştır —
            # production'ın kendi write pipeline'ının nihai değeri Redis'e
            # yazmasına yeterli süre tanı (bkz. önceki tur bulgusu).
            shadow_last = bars[-2]

            prod_df = await RedisClient.get_mtf_klines(symbol, tf)
            if prod_df is None or prod_df.empty:
                round_missing += 1
                stats.compare_missing += 1
                continue

            match_rows = prod_df[prod_df["open_time"] == shadow_last["open_time"]]
            if match_rows.empty:
                round_missing += 1
                stats.compare_missing += 1
                continue
            prod_last = match_rows.iloc[-1]

            match_close = abs(float(prod_last["close"]) - shadow_last["close"]) < 1e-9
            match_volume = abs(float(prod_last["volume"]) - shadow_last["volume"]) < 1e-6

            if match_close and match_volume:
                round_ok += 1
                stats.compare_ok += 1
                if symbol in log_sample:
                    logger.info("[KARŞILAŞTIRMA] %-10s %-4s OK (close=%s vol=%.4f)",
                                symbol, tf, shadow_last["close"], shadow_last["volume"])
            else:
                round_mismatch += 1
                stats.compare_mismatch += 1
                logger.warning(
                    "[KARŞILAŞTIRMA] %-10s %-4s FARK VAR ⚠️ shadow(close=%s vol=%.4f) prod(close=%s vol=%.4f)",
                    symbol, tf, shadow_last["close"], shadow_last["volume"],
                    prod_last["close"], prod_last["volume"],
                )

        logger.info(
            "[KARŞILAŞTIRMA-ÖZET] bu turda: OK=%d, FARK=%d, eksik=%d",
            round_ok, round_mismatch, round_missing,
        )


async def report_resource_usage(num_symbols: int, num_connections: int):
    """Thread sayısı + RSS bellek — TF sayısı/stream sayısı arttıkça bunların
    NASIL değiştiğini (yoksa sabit mi kaldığını) kanıtlamak için periyodik log."""
    while True:
        await asyncio.sleep(30)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        logger.info(
            "[KAYNAK] sembol=%d bağlantı(task)=%d thread=%d RSS_peak=%.1fMB",
            num_symbols, num_connections, threading.active_count(), rss_mb,
        )


async def main(duration_sec: int, num_symbols: int):
    stats = ShadowStats()
    symbols = await fetch_production_symbols(num_symbols)
    if not symbols:
        logger.error("Production'da hiç sembol bulunamadı, çıkılıyor.")
        return

    all_streams = [f"{s.lower()}@kline_{tf}" for s in symbols for tf in TIMEFRAMES]
    chunks = [all_streams[i:i + MAX_STREAMS_PER_CONNECTION] for i in range(0, len(all_streams), MAX_STREAMS_PER_CONNECTION)]
    log_sample = set(symbols[:LOG_SAMPLE_SIZE])

    logger.info("=" * 70)
    logger.info("TAM ÖLÇEK TESTİ: %d sembol × %d TF = %d stream, %d bağlantı(task), %ds sürecek",
                len(symbols), len(TIMEFRAMES), len(all_streams), len(chunks), duration_sec)
    logger.info("TF'ler: %s", TIMEFRAMES)
    logger.info("Örnek loglanacak semboller: %s", sorted(log_sample))
    logger.info("Başlangıç thread sayısı: %d", threading.active_count())

    tasks = [asyncio.create_task(report_resource_usage(len(symbols), len(chunks)))]
    for i, chunk in enumerate(chunks, start=1):
        tasks.append(asyncio.create_task(run_shadow_connection(chunk, i, stats, log_sample)))
        await asyncio.sleep(0.25)  # production'daki gibi bağlantıları stagger et
    tasks.append(asyncio.create_task(compare_with_production(symbols, stats, log_sample)))

    await asyncio.sleep(duration_sec)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("=" * 70)
    logger.info(
        "TEST BİTTİ — sembol=%d stream=%d bağlantı=%d | mesaj=%d hata=%d reconnect=%d",
        len(symbols), len(all_streams), len(chunks), stats.msg_count, stats.errors, stats.reconnects,
    )
    logger.info("TF başına kapanan bar sayısı (toplam, tüm semboller):")
    per_tf_counts: dict[str, int] = {}
    for (_symbol, tf), bars in stats.closed_bars.items():
        per_tf_counts[tf] = per_tf_counts.get(tf, 0) + len(bars)
    for tf in TIMEFRAMES:
        logger.info("  %-4s %d bar kapandı (kapanmadıysa bu pencerede hiç kapanmamış demektir, hata değil)",
                    tf, per_tf_counts.get(tf, 0))
    logger.info(
        "KARŞILAŞTIRMA TOPLAM — OK=%d FARK=%d eksik=%d",
        stats.compare_ok, stats.compare_mismatch, stats.compare_missing,
    )
    logger.info("Bitiş thread sayısı: %d", threading.active_count())


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DURATION_SEC
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_NUM_SYMBOLS
    asyncio.run(main(dur, n))
