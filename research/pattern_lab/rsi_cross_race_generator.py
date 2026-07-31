"""
rsi_cross_race_generator.py — do_open_streak_race_generator.py'nin RSI_Cross
sürümü. do_open_streak'ten FARKLI: RSI_Cross zaten `signals` tablosuna canlı
signal_processor.py tarafından yazılıyor — ayrı bir detector-doğrulama
gerekmiyor, DB kaydı doğrudan "gerçek" kabul edilir (bkz. `signals.opened_at`
== `price_data.timestamp` konvansiyonu, ikisi de naive-yerel, 31 Tem 2026'da
doğrulandı).

Belirli bir gün + yön (Long/Short) için o günün RSI_Cross sinyallerinden
RASTGELE N tanesini seçip aynı gerçek-saat yarış formatıyla (fiyat + TF
hizalanma + VPMV + EVOL/ERSI) görselleştirir. Zaman serisi üretimi ve HTML
render'ı do_open_streak_race_generator.py ile ORTAK (aynı fonksiyonlar
import edilip kullanılıyor) — kod tekrarı yok.

Kullanım:
    python -m research.pattern_lab.rsi_cross_race_generator 2026-07-24 Long
    python -m research.pattern_lab.rsi_cross_race_generator 2026-07-24 Short --n 15 --seed 7 --out /tmp/x.html
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # noqa: E402  pylint: disable=wrong-import-position

from research.pattern_lab.do_open_streak_race_generator import (  # noqa: E402  pylint: disable=wrong-import-position
    _one_symbol_series,
    render_html,
)

DEFAULT_N = 12


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def find_day_signals(
    date_str: str, direction: str, n: int, seed: "int | None"
) -> list[tuple[str, str, pd.Timestamp]]:
    """O gün (yerel takvim günü), verilen yönde tetiklenen TÜM RSI_Cross
    sinyallerini DB'den çeker, gerekirse n'e rastgele indirger.

    Döner: [(etiket, gerçek_sembol, giriş_zamanı_yerel), ...] zaman sırasıyla.
    """
    day_start = pd.Timestamp(date_str)
    day_end = day_start + pd.Timedelta(days=1)

    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, opened_at
        FROM signals
        WHERE indicators LIKE 'RSI_Cross%%'
          AND signal_type = %s
          AND opened_at >= %s AND opened_at < %s
        ORDER BY opened_at
        """,
        (direction, day_start.to_pydatetime(), day_end.to_pydatetime()),
    )
    rows = cur.fetchall()
    conn.close()

    print(f"[{date_str}] {direction} yönünde toplam {len(rows)} RSI_Cross sinyali bulundu")

    if len(rows) > n:
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
        rows.sort(key=lambda r: r[1])
        print(f"  -> rastgele {n} tanesi seçildi (seed={seed})")

    dup_count: dict[str, int] = {}
    labeled: list[tuple[str, str, pd.Timestamp]] = []
    for sym, opened_at in rows:
        dup_count[sym] = dup_count.get(sym, 0) + 1
        label = sym if dup_count[sym] == 1 else f"{sym}{dup_count[sym]}"
        labeled.append((label, sym, pd.Timestamp(opened_at)))
    return labeled


async def build_dataset(
    date_str: str, direction: str, n: int, seed: "int | None"
) -> tuple[list[dict], list[tuple[str, pd.Timestamp]]]:
    signals = find_day_signals(date_str, direction, n, seed)
    if not signals:
        return [], []

    all_rows: list[dict] = []
    meta: list[tuple[str, pd.Timestamp]] = []
    day_start = min(t for _, _, t in signals)

    for label, real_sym, entry_local in signals:
        rows = await _one_symbol_series(label, real_sym, entry_local)
        if not rows:
            print(f"  [uyarı] {label}: veri bulunamadı, atlanıyor")
            continue
        offset_hours = (entry_local - day_start).total_seconds() / 3600.0
        for r in rows:
            r["day_hours"] = round(r["hours"] + offset_hours, 2)
            r["entry_clock"] = entry_local.strftime("%H:%M")
        all_rows.extend(rows)
        meta.append((label, entry_local))
        print(f"  {label:14} {entry_local.strftime('%H:%M')}  ({len(rows)} nokta)")

    return all_rows, meta


async def main_async(date_str: str, direction: str, n: int, seed: "int | None", out_path: str) -> None:
    print(f"[{date_str}] RSI_Cross {direction} sinyalleri aranıyor...")
    rows, meta = await build_dataset(date_str, direction, n, seed)
    if not meta:
        print("Bu gün/yönde hiç RSI_Cross sinyali bulunamadı.")
        return

    print(f"\nToplam {len(meta)} sinyal, {len(rows)} veri noktası. HTML üretiliyor...")
    html = render_html(date_str, rows, meta)
    # başlığı RSI_Cross'a göre güncelle (do_open_streak template'i genel tutuldu,
    # render_html placeholder'ları ZATEN doldurduğu için burada LİTERAL/render
    # edilmiş metinler üzerinde değişiklik yapılıyor, placeholder değil)
    html = html.replace("<title>do_open_streak —", f"<title>RSI_Cross {direction} —")
    html = html.replace(
        "tetiklenen {} sinyal — girişten".format(len(meta)),
        f"RSI_Cross {direction} yönünde rastgele seçilen {len(meta)} sinyal — girişten",
    )
    html = html.replace(
        "Kaynak: <code>signals/do_open_streak.py</code> (canlı kod ile doğrulanmış giriş anları) +",
        f"Kaynak: <code>signals</code> tablosu (RSI_Cross, {direction}, {date_str} gününden rastgele n={n}, seed={seed}) +",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Yazıldı: {out_path} ({len(html):,} byte)")


def main() -> None:
    parser = argparse.ArgumentParser(description="RSI_Cross günlük rastgele yarış HTML'i üretir")
    parser.add_argument("date", help="YYYY-MM-DD (yerel takvim günü)")
    parser.add_argument("direction", choices=["Long", "Short"], help="Sinyal yönü")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Kaç sinyal (varsayılan {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=None, help="Rastgele seçim için sabit seed (tekrarlanabilirlik)")
    parser.add_argument("--out", default=None, help="Çıktı HTML yolu")
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.getcwd(), f"rsi_cross_race_{args.direction.lower()}_{args.date}.html"
    )
    asyncio.run(main_async(args.date, args.direction, args.n, args.seed, out_path))


if __name__ == "__main__":
    main()
