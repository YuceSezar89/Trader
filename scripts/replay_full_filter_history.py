"""
SignalFilter tam geçmiş replay — sistem ilk kayıttan bugüne kadar hiç kesintiye
uğramadan çalışsaydı referans noktaları (last_short_high/last_long_low) ne
olurdu, onu cagg_5m/cagg_15m'deki TAM geçmiş veriyi kullanarak yeniden inşa edip
signal_filter_events tablosuna (bkz. migration 016, signals/signal_filter.py)
yazar.

Neden gerekli: SignalFilter referansları sadece canlı iken check() çağrıldığında
güncelleniyor. Tablo baştan boşsa (yeni deploy) veya sistem restart olduysa,
downtime süresince gerçekleşen crossover'lar hiç check edilmemiş olur — ilk
gerçek sinyal her (symbol, interval, indicator, yön) kombinasyonunda referans
yokluğundan reddedilir, organik dolması haftalar sürebilir.

signals tablosundan backfill etmek YETERSİZ: o tablo sadece KABUL EDİLEN
sinyalleri görür, oysa PineScript'in referansı her DENEMEDE (kabul/red fark
etmeksizin) güncellenir. Bu script ham fiyat verisini indikatör hesaplama +
sinyal tespiti zincirinden geçirip her denemeyi gerçekten simüle eder.

signals tablosuna HİÇBİR ŞEY YAZMAZ, hiçbir DB kaydı açmaz/kapatmaz/değiştirmez
— tek etkisi signal_filter_events. Zaten dolu (symbol, interval) kombinasyonları
ATLANIR (idempotent — ikinci kez çalıştırmak canlı birikmiş veriyi duplike etmez,
sadece boş kalanları doldurur).

Kullanım:
    python scripts/replay_full_filter_history.py [--concurrency 20] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

import config  # noqa: F401  # .env yüklemesini (load_dotenv) tetikler — database.engine'den önce import edilmeli
from database.engine import get_session
from database.crud import get_cagg_klines
from indicators.core import add_all_indicators
from signals.signal_engine import SignalEngine

_TFS = ["5m", "15m"]
_MAX_BARS = 200_000  # tüm geçmişi kapsayacak kadar büyük (en eski sembol bile bu kadar bar biriktirmedi)


async def _get_all_symbols() -> list[str]:
    async with get_session() as session:
        result = await session.execute(
            text("SELECT DISTINCT symbol FROM cagg_5m UNION SELECT DISTINCT symbol FROM cagg_15m")
        )
        return sorted(r[0] for r in result.fetchall())


async def _get_populated_pairs() -> set[tuple[str, str]]:
    """Zaten en az bir olayı olan (symbol, interval) kombinasyonları — bunlar
    canlı sistem tarafından doldurulmuş (veya önceki bir replay'den kalma)
    sayılır, tekrar replay edilmez."""
    async with get_session() as session:
        result = await session.execute(
            text("SELECT DISTINCT symbol, interval FROM signal_filter_events WHERE interval = ANY(:tfs)"),
            {"tfs": _TFS},
        )
        return {(r[0], r[1]) for r in result.fetchall()}


async def _replay_one(
    engine: SignalEngine, symbol: str, tf: str, semaphore: asyncio.Semaphore, loop: asyncio.AbstractEventLoop
) -> int:
    async with semaphore:
        try:
            df = await get_cagg_klines(symbol, tf, _MAX_BARS)
            if df.empty:
                return 0
            df_ind = await loop.run_in_executor(None, add_all_indicators, df)
            return await engine.replay_filter_state(df_ind, symbol, tf, replay_from_ms=0)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  [HATA] {symbol} {tf}: {e}")
            return 0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = await _get_all_symbols()
    populated = await _get_populated_pairs()
    todo = [(sym, tf) for sym in symbols for tf in _TFS if (sym, tf) not in populated]

    print(
        f"{len(symbols)} sembol x {len(_TFS)} TF = {len(symbols) * len(_TFS)} kombinasyon. "
        f"{len(populated)} zaten dolu (atlanacak), {len(todo)} replay edilecek."
    )

    if args.dry_run:
        print("--dry-run: hiçbir yazma yapılmadı.")
        return

    if not todo:
        print("Yapılacak bir şey yok, tüm kombinasyonlar zaten dolu.")
        return

    engine = SignalEngine()
    semaphore = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_event_loop()
    tasks = [_replay_one(engine, sym, tf, semaphore, loop) for sym, tf in todo]
    results = await asyncio.gather(*tasks)
    total_bars = sum(results)

    async with get_session() as session:
        result = await session.execute(
            text("SELECT count(*) FROM signal_filter_events WHERE interval = ANY(:tfs)"),
            {"tfs": _TFS},
        )
        total_events = result.scalar()

    print(f"Replay tamamlandı: {total_bars} bar işlendi, signal_filter_events'te toplam {total_events} olay.")


if __name__ == "__main__":
    asyncio.run(main())
