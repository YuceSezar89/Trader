"""
do_open_streak_race_generator.py — belirli bir gün için do_open_streak
sinyallerinin gerçek-saat "yarış" görselleştirmesini (fiyat + TF hizalanma +
VPMV bileşenleri + EVOL/ERSI, hepsi senkronize) üretir.

_race_template.html (bu dizinde) veri ile doldurulup tek bir bağımsız HTML
dosyası yazılır — Artifact tool'una doğrudan verilebilir.

Yalnızca signals/do_open_streak.py::DoOpenStreakDetector.check() (canlı kod,
ayrı bir backtest kopyası DEĞİL) ile doğrulanan sinyaller dahil edilir — bkz.
29 Tem 2026 backtest/canlı sapma dersi (do_open_streak_tf_alignment_bt.py).

Kullanım:
    python -m research.pattern_lab.do_open_streak_race_generator 2026-07-21
    python -m research.pattern_lab.do_open_streak_race_generator 2026-07-21 --out /tmp/race.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.crud import get_cagg_klines  # noqa: E402  pylint: disable=wrong-import-position
from indicators.core import calculate_evol  # noqa: E402  pylint: disable=wrong-import-position
from signals.do_open_streak import DoOpenStreakDetector  # noqa: E402  pylint: disable=wrong-import-position
from signals.market_context import (  # noqa: E402  pylint: disable=wrong-import-position
    compute_cvd_slope,
    compute_vp_score,
)
from signals.signal_processor import _compute_devisso_score  # noqa: E402  pylint: disable=wrong-import-position
from signals.vpm_calculator import VPMCalculator  # noqa: E402  pylint: disable=wrong-import-position
from utils.vpmv import compute_components  # noqa: E402  pylint: disable=wrong-import-position

import research.pattern_lab.do_open_streak_tf_alignment_bt as tfa  # noqa: E402  pylint: disable=wrong-import-position

_TFS = ["5m", "15m", "1h", "4h", "6h", "8h", "12h"]
_FORWARD_STEPS = 96  # 24h @ 15m
_STEP_EVERY = 2  # 30 dakikada bir örnekle
_KLINE_LOOKBACK = 3000

_TR_MONTHS_SHORT = {
    1: "Oca", 2: "Şub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz",
    7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki", 11: "Kas", 12: "Ara",
}
_TR_MONTHS_LONG = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}

_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_race_template.html")


def _ts_to_local(ts_ms: int) -> pd.Timestamp:
    """UTC-eşdeğeri (naive, ms) -> yerel (+3) naive Timestamp. Proje genelinde
    kullanılan tek dönüşüm: ham ms değeri hiçbir sistem-TZ yorumlamasından
    geçirilmeden +3 saat eklenir (bkz. get_cagg_klines'ın open_time'ı)."""
    return pd.to_datetime(int(ts_ms), unit="ms") + pd.Timedelta(hours=3)


async def find_day_signals(date_str: str) -> list[tuple[str, str, pd.Timestamp]]:
    """O gün (yerel takvim günü) canlı kodun (DoOpenStreakDetector.check())
    gerçekten onayladığı TÜM do_open_streak sinyallerini zaman sırasıyla döner.

    Döner: [(etiket, gerçek_sembol, giriş_zamanı_yerel), ...] — aynı sembol aynı
    gün birden fazla tetiklenirse etiket "SEMBOL", "SEMBOL2", "SEMBOL3"... olur.
    """
    target_date = pd.Timestamp(date_str).date()

    conn = tfa._conn()  # pylint: disable=protected-access
    bad = tfa._bad_symbols(conn.cursor())  # pylint: disable=protected-access
    conn.close()

    days_back = max((pd.Timestamp.now().normalize() - pd.Timestamp(date_str)).days + 3, 10)
    df15 = tfa._fetch("cagg_15m", days_back, bad)  # pylint: disable=protected-access

    det = DoOpenStreakDetector()
    found: list[tuple[str, str, pd.Timestamp, int]] = []  # + trigger_idx (debug)

    for sym, g in df15.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < tfa.MIN_BARS:  # pylint: disable=protected-access
            continue
        ts = g["ts"]
        ts_ms = ts.astype("int64").to_numpy() // 1_000_000
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = tfa._daily_open(  # pylint: disable=protected-access
            pd.to_datetime(ts) + pd.Timedelta(hours=3), o
        )
        gate = tfa._do_break_gate(o, c, daily_open)  # pylint: disable=protected-access
        events = tfa._detect_events(o, c, gate)  # pylint: disable=protected-access

        for streak_start, trigger_idx in events:
            local_ts = _ts_to_local(int(ts_ms[trigger_idx]))
            if local_ts.date() != target_date:
                continue

            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 >= len(g):
                continue
            if not tfa._pullback_ok(h, l, bar1, bar2, bar3):  # pylint: disable=protected-access
                continue
            start_low = l[bar1]
            long_perc = (h[trigger_idx] - start_low) / start_low * 100.0
            if tfa._gauss_sum(round(long_perc, 2)) < tfa.GAUSS_THRESHOLD:  # pylint: disable=protected-access
                continue

            window = g.iloc[: trigger_idx + 1].rename(columns={"ts": "open_time"})
            window = window.assign(open_time=(window["open_time"].astype("int64") // 1_000_000))
            if det.check(sym, window) is None:
                continue  # canlı kod reddetti (ör. 200-bar gate penceresi dışı)

            found.append((sym, sym, local_ts, trigger_idx))

    found.sort(key=lambda r: r[2])

    dup_count: dict[str, int] = {}
    labeled: list[tuple[str, str, pd.Timestamp]] = []
    for sym, real_sym, local_ts, _ in found:
        dup_count[sym] = dup_count.get(sym, 0) + 1
        label = sym if dup_count[sym] == 1 else f"{sym}{dup_count[sym]}"
        labeled.append((label, real_sym, local_ts))
    return labeled


async def _one_symbol_series(label: str, real_sym: str, entry_local: pd.Timestamp) -> list[dict]:
    entry_utc_ms = int((entry_local - pd.Timedelta(hours=3)).timestamp() * 1000)
    df = await get_cagg_klines(real_sym, "15m", _KLINE_LOOKBACK)
    idx = df.index[df["open_time"] >= entry_utc_ms]
    if len(idx) == 0:
        return []
    entry_idx = idx[0]
    entry_price = float(df["close"].iloc[entry_idx])

    ha_series = {}
    for tf in _TFS:
        tdf = await get_cagg_klines(real_sym, tf, _KLINE_LOOKBACK)
        if tdf.empty:
            ha_series[tf] = None
            continue
        tdf_r = tdf.rename(columns={"open_time": "ts"})
        ha_series[tf] = tfa._ha_bull_series(  # pylint: disable=protected-access
            tdf_r.assign(ts=pd.to_datetime(tdf_r["ts"], unit="ms"))
        )

    rows: list[dict] = []
    for i in range(entry_idx, min(entry_idx + _FORWARD_STEPS + 1, len(df)), _STEP_EVERY):
        window = df.iloc[: i + 1]
        if len(window) < 60:
            continue
        price = float(window["close"].iloc[-1])
        pnl = (price - entry_price) / entry_price * 100.0
        try:
            cvd = compute_cvd_slope(window)
            vp_buy, vp_sell = compute_vp_score(window)
            vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long")
            vpmv = VPMCalculator.calculate(
                vol_score=vol_s, momentum_score=mom_s, vlt_score=vlt_s, price_score=prc_s
            )
            evol = calculate_evol(window)
            ersi = _compute_devisso_score(window)
        except Exception:  # pylint: disable=broad-exception-caught
            continue

        hours = (i - entry_idx) * 15 / 60.0
        at_ms = int(df["open_time"].iloc[i])
        n_aligned = 0
        ha_row = {}
        for tf in _TFS:
            b = None
            if ha_series.get(tf) is not None:
                close_arr, bull_arr = ha_series[tf]
                b = tfa._last_closed_bull(close_arr, bull_arr, at_ms)  # pylint: disable=protected-access
            ha_row[f"ha_{tf}"] = bool(b) if b is not None else False
            if b:
                n_aligned += 1

        row = {
            "symbol": label,
            "hours": round(hours, 2),
            "price_pct": round(pnl, 3),
            "cvd": _clean_float(cvd),
            "vp": _clean_float(vp_buy - vp_sell) if vp_buy is not None else None,
            "vol": _clean_float(vol_s),
            "mom": _clean_float(mom_s),
            "vlt": _clean_float(vlt_s),
            "prc": _clean_float(prc_s),
            "combined": _clean_float(vpmv),
            "evol": _clean_float(evol),
            "ersi": _clean_float(ersi),
        }
        row.update(ha_row)
        row["n_aligned"] = n_aligned
        rows.append(row)
    return rows


def _clean_float(v) -> float | None:
    """NaN/Inf -> None (JS'in JSON.parse'ı literal NaN'ı kabul etmiyor —
    29 Tem 2026 "sayfa boş" hatasının kök nedeni, bkz. commit d453e81 sonrası
    forensik)."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)


async def build_dataset(date_str: str) -> tuple[list[dict], list[tuple[str, pd.Timestamp]]]:
    signals = await find_day_signals(date_str)
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


def render_html(date_str: str, rows: list[dict], meta: list[tuple[str, pd.Timestamp]]) -> str:
    with open(_TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    day_start = min(t for _, t in meta)
    d = pd.Timestamp(date_str)

    title_short = f"{d.day} {_TR_MONTHS_SHORT[d.month]}"
    title_long = f"{d.day} {_TR_MONTHS_LONG[d.month]}"
    day_start_expr = f"{day_start.hour} + {day_start.minute} / 60"
    day_start_label = day_start.strftime("%H:%M")

    html = template
    html = html.replace("{TITLE_SHORT}", title_short)
    html = html.replace("{TITLE_LONG}", title_long)
    html = html.replace("{N_SIGNALS}", str(len(meta)))
    html = html.replace("{DAY_START_HOUR_EXPR}", day_start_expr)
    html = html.replace("{DAY_START_LABEL}", day_start_label)
    html = html.replace("{DATA_JSON}", json.dumps(rows, allow_nan=False))
    return html


async def main_async(date_str: str, out_path: str) -> None:
    print(f"[{date_str}] canlı kodla doğrulanan do_open_streak sinyalleri aranıyor...")
    rows, meta = await build_dataset(date_str)
    if not meta:
        print("Bu günde hiç canlı-doğrulanmış sinyal bulunamadı.")
        return

    print(f"\nToplam {len(meta)} sinyal, {len(rows)} veri noktası. HTML üretiliyor...")
    html = render_html(date_str, rows, meta)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Yazıldı: {out_path} ({len(html):,} byte)")


def main() -> None:
    parser = argparse.ArgumentParser(description="do_open_streak günlük yarış HTML'i üretir")
    parser.add_argument("date", help="YYYY-MM-DD (yerel takvim günü)")
    parser.add_argument(
        "--out", default=None, help="Çıktı HTML yolu (varsayılan: ./do_open_streak_race_<tarih>.html)"
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.getcwd(), f"do_open_streak_race_{args.date}.html")
    asyncio.run(main_async(args.date, out_path))


if __name__ == "__main__":
    main()
