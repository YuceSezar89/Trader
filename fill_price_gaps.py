#!/usr/bin/env python3
"""
price_data tablosundaki iç gap'leri tespit eder ve Binance'ten doldurur.

Kullanım:
    python fill_price_gaps.py                      # son 30 gün, tüm semboller, 1m
    python fill_price_gaps.py --days 7             # son 7 gün
    python fill_price_gaps.py --symbols BTCUSDT    # tek sembol
    python fill_price_gaps.py --interval 4h        # farklı interval
"""

import argparse
import asyncio
import logging
import os
from typing import Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Interval → milisaniye
_INTERVAL_MS: dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


def _db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "trader"),
        user=os.getenv("DB_USER", "trader"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def find_gaps(symbol: str, interval: str, days: int) -> list[tuple[int, int]]:
    """(gap_start_ms, gap_end_ms) listesi döner — her biri doldurulacak bir boşluk."""
    interval_ms = _INTERVAL_MS.get(interval, 60_000)
    threshold_ms = interval_ms * 2  # 2 bar'dan büyük fark = gap

    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            EXTRACT(EPOCH FROM prev_ts AT TIME ZONE 'Europe/Istanbul')::bigint * 1000 AS gap_start_ms,
            EXTRACT(EPOCH FROM curr_ts AT TIME ZONE 'Europe/Istanbul')::bigint * 1000 AS gap_end_ms
        FROM (
            SELECT
                timestamp                                           AS curr_ts,
                LAG(timestamp) OVER (ORDER BY timestamp)           AS prev_ts
            FROM price_data
            WHERE symbol   = %s
              AND interval  = %s
              AND timestamp >= NOW() - (%s * INTERVAL '1 day')
        ) t
        WHERE prev_ts IS NOT NULL
          AND EXTRACT(EPOCH FROM (curr_ts - prev_ts)) * 1000 > %s
        ORDER BY prev_ts
        """,
        (symbol, interval, days, threshold_ms),
    )
    gaps = [(int(r[0]), int(r[1])) for r in cur.fetchall()]
    conn.close()
    return gaps


async def _fetch_with_ban_retry(symbol: str, interval: str, start_ms: int) -> Optional[pd.DataFrame]:
    """Kline verisi çeker — ban (-1003) gelirse ban kalkana kadar bekler."""
    import time, re
    import aiohttp

    url = (
        f"https://fapi.binance.com/fapi/v1/klines"
        f"?symbol={symbol}&interval={interval}&limit=1500&startTime={start_ms}"
    )

    for attempt in range(10):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    data = await resp.json()

            if isinstance(data, dict):
                code = data.get("code", 0)
                msg  = data.get("msg", "")
                if code == -1003:  # IP ban
                    match = re.search(r"banned until (\d+)", msg)
                    if match:
                        ban_until_ms = int(match.group(1))
                        wait_s = max(10, (ban_until_ms - int(time.time() * 1000)) / 1000 + 15)
                    else:
                        wait_s = 60
                    logger.warning("IP ban — %.0f saniye bekleniyor...", wait_s)
                    await asyncio.sleep(wait_s)
                    continue
                logger.warning("[%s] API hatası: %s", symbol, data)
                return None

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ])
            df = df.astype({"open_time": int, "open": float, "high": float,
                            "low": float, "close": float, "volume": float})
            return df

        except Exception as exc:
            logger.warning("[%s] Bağlantı hatası: %s — 10s bekleniyor", symbol, exc)
            await asyncio.sleep(10)

    return None


async def fill_gap(
    symbol: str,
    interval: str,
    gap_start_ms: int,
    gap_end_ms: int,
) -> int:
    """Tek gap'i Binance'ten çekip DB'ye yazar. Eklenen bar sayısını döner."""
    from database.crud import bulk_insert_price_data

    interval_ms = _INTERVAL_MS[interval]
    fetch_start = gap_start_ms + interval_ms  # gap'in hemen sonraki bar'ı
    total = 0

    while fetch_start < gap_end_ms:
        df = await _fetch_with_ban_retry(symbol, interval, fetch_start)
        if df is None or df.empty:
            break

        # gap_end'den sonrasını kes
        df = df[df["open_time"] < gap_end_ms]
        if df.empty:
            break

        await bulk_insert_price_data(symbol, df, interval=interval)
        total += len(df)

        last_ts = int(df["open_time"].iloc[-1])
        if last_ts <= fetch_start or len(df) < 1500:
            break
        fetch_start = last_ts + interval_ms
        await asyncio.sleep(0.5)  # live system ile paylaşılan rate limit

    return total


async def process_symbol(symbol: str, interval: str, days: int) -> int:
    gaps = find_gaps(symbol, interval, days)
    if not gaps:
        return 0

    total = 0
    for gap_start, gap_end in gaps:
        from datetime import datetime
        start_str = datetime.fromtimestamp(gap_start / 1000).strftime("%m/%d %H:%M")
        end_str   = datetime.fromtimestamp(gap_end   / 1000).strftime("%m/%d %H:%M")
        gap_min   = (gap_end - gap_start) / 60_000
        logger.info("[%s] Gap: %s → %s (%.0f dk)", symbol, start_str, end_str, gap_min)

        added = await fill_gap(symbol, interval, gap_start, gap_end)
        logger.info("[%s] + %d bar eklendi", symbol, added)
        total += added

    return total


async def main(
    symbols: Optional[list[str]],
    interval: str,
    days: int,
    concurrency: int,
) -> None:
    if not symbols:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT symbol FROM price_data
            WHERE interval = %s
              AND timestamp >= NOW() - (%s * INTERVAL '1 day')
            ORDER BY symbol
            """,
            (interval, days),
        )
        symbols = [r[0] for r in cur.fetchall()]
        conn.close()
        logger.info("%d sembol bulundu", len(symbols))

    # Gap tespiti
    logger.info("Gap analizi yapılıyor...")
    gap_counts = {s: find_gaps(s, interval, days) for s in symbols}
    need_fill  = {s: g for s, g in gap_counts.items() if g}
    logger.info(
        "%d / %d sembolde gap var (toplam %d gap)",
        len(need_fill), len(symbols),
        sum(len(g) for g in need_fill.values()),
    )

    if not need_fill:
        logger.info("Doldurulacak gap yok, çıkılıyor.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def run(sym):
        async with sem:
            return await process_symbol(sym, interval, days)

    results = await asyncio.gather(*[run(s) for s in need_fill], return_exceptions=True)

    total_added = 0
    for sym, res in zip(need_fill, results):
        if isinstance(res, Exception):
            logger.error("[%s] Hata: %s", sym, res)
        else:
            total_added += res

    logger.info("Tamamlandı — toplam %d bar eklendi", total_added)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="price_data iç gap doldurma")
    parser.add_argument("--symbols", nargs="+", help="Semboller (varsayılan: tümü)")
    parser.add_argument("--interval", default="1m", choices=list(_INTERVAL_MS))
    parser.add_argument("--days", type=int, default=30, help="Kaç günlük geçmişe bak")
    parser.add_argument("--concurrency", type=int, default=3, help="Eş zamanlı sembol sayısı")
    args = parser.parse_args()

    asyncio.run(main(args.symbols, args.interval, args.days, args.concurrency))
