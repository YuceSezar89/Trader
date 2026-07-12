"""
rsi_cross_sl_sweep_bt.py'nin devamı — mum-şekli+VPMV birleşik filtresini
(v2-9, `rsi_cross_combined_economic_bt.py`, $243/ay — o test `signals`
tablosundaki gerçek realized_pnl'i, yani reversal/timeout kapanışını
kullanıyordu) SL taramasıyla BİRLEŞTİRİR: filtre geçen olaylarda SL eklemek
ekonomiyi daha da iyileştiriyor mu?

Kural (v2-9 ile birebir): kategori != "üst-fitil-baskın" VE VPMV sıçraması
(post[+1]-pre[-1]) üst tercilde. Eşik SADECE ilk yarıdan (in-sample)
türetilip ikinci yarıya (out-of-sample) sabit uygulanıyor. RSI_Cross
event tanımı + SignalFilter replikasyonu + SL simülasyonu
rsi_cross_sl_sweep_bt.py'den doğrudan import edildi — kod tekrarı yok.
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD,
    ROUND_TRIP_FEE,
)
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
)
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_candle_shape_bt import (
    _classify,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.rsi_cross_sl_sweep_bt import (  # pylint: disable=wrong-import-position
    SL_MULTIPLES,
    _apply_signal_filter,
    _dollar_stats,
    _rsi_cross_events,
    _signed_ret,
    _simulate_sl_exit,
)
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position


def _fetch_with_volume() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _report(label: str, events: list, days_span: float) -> None:
    print(f"\n=== {label} (n={len(events)}) ===")
    if not events:
        return

    blind_pnls = []
    sl_pnls: dict[float, list] = {sl: [] for sl in SL_MULTIPLES}

    for direction, h, l, c, atr_val, i, entry_price in events:
        last_i = min(i + HORIZON_BARS, len(c) - 1)
        blind_pnls.append(
            _signed_ret(direction, c[last_i], entry_price) * POSITION_USD - ROUND_TRIP_FEE
        )

        for sl in SL_MULTIPLES:
            exit_price, _reason = _simulate_sl_exit(
                direction, h, l, c, i, entry_price, atr_val, sl, HORIZON_BARS
            )
            sl_pnls[sl].append(
                _signed_ret(direction, exit_price, entry_price) * POSITION_USD - ROUND_TRIP_FEE
            )

    print(f"{'yöntem':24} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    s = _dollar_stats(np.array(blind_pnls), days_span)
    print(
        f"{'kör 24h bekleme':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
        f"{s['total_usd']:>10} {s['usd_per_month']:>10}"
    )
    for sl in SL_MULTIPLES:
        s = _dollar_stats(np.array(sl_pnls[sl]), days_span)
        print(
            f"{'SL='+str(sl)+'×ATR (TP yok)':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
            f"{s['total_usd']:>10} {s['usd_per_month']:>10}"
        )


def run():
    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    clean_events = []  # (direction, h, l, c, atr_val, i, entry_price, ts, kategori, jump)

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        fast = calculate_rsi(g, period=Config.RSI_FAST_WINDOW).to_numpy()
        slow = calculate_rsi(g, period=Config.RSI_SLOW_WINDOW).to_numpy()
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        raw_events = _rsi_cross_events(fast, slow)
        filtered_events = _apply_signal_filter(raw_events, h, l)
        if not filtered_events:
            continue

        series_long = compute_series(g, "Long")
        series_short = compute_series(g, "Short")

        for i, direction in filtered_events:
            if (
                i + HORIZON_BARS >= len(c)
                or i - 1 < 0
                or i + 1 >= len(g)
                or not (np.isfinite(atr[i]) and atr[i] > 0)
            ):
                continue
            series = series_long if direction == "Long" else series_short
            pre_v, post_v = series.iloc[i - 1], series.iloc[i + 1]
            if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                continue
            kategori = _classify(g.iloc[i])
            clean_events.append(
                (
                    direction,
                    h,
                    l,
                    c,
                    atr[i],
                    i,
                    c[i],
                    ts.iloc[i],
                    kategori,
                    post_v - pre_v,
                )
            )

    t_min = min(e[7] for e in clean_events)
    t_max = max(e[7] for e in clean_events)
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max}")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    is_events = [e for e in clean_events if e[7] < mid]
    oos_events = [e for e in clean_events if e[7] >= mid]
    print(f"in-sample olay: {len(is_events)} | out-of-sample olay: {len(oos_events)}\n")

    jump_threshold = float(np.percentile([e[9] for e in is_events], 66.7))
    print(f"SABİT (OOS'a uygulanan) VPMV sıçrama eşiği: {jump_threshold:.2f}\n")

    oos_baseline = [e[:7] for e in oos_events]
    oos_shape_only = [e[:7] for e in oos_events if e[8] != "üst-fitil-baskın"]
    oos_vpmv_only = [e[:7] for e in oos_events if e[9] >= jump_threshold]
    oos_combined = [
        e[:7] for e in oos_events if e[8] != "üst-fitil-baskın" and e[9] >= jump_threshold
    ]

    _report("OOS baseline (temiz, filtresiz)", oos_baseline, oos_days)
    _report("OOS sadece mum-şekli filtreli", oos_shape_only, oos_days)
    _report("OOS sadece VPMV sıçraması filtreli", oos_vpmv_only, oos_days)
    _report("OOS BİRLEŞİK (mum-şekli + VPMV)", oos_combined, oos_days)

    print(
        "\n\n──────── OOS PENCERESİNİN KENDİ İÇİNDE SPLIT-PERIOD "
        "(VPMV-filtreli, canlıya alma öncesi son kontrol) ────────"
    )
    oos_mid = mid + (t_max - mid) / 2
    half_days = oos_days / 2
    oos_first = [e[:7] for e in oos_events if e[9] >= jump_threshold and e[7] < oos_mid]
    oos_second = [e[:7] for e in oos_events if e[9] >= jump_threshold and e[7] >= oos_mid]
    print(f"OOS orta nokta: {oos_mid}")
    _report("OOS — ilk yarı (VPMV-filtreli)", oos_first, half_days)
    _report("OOS — ikinci yarı (VPMV-filtreli)", oos_second, half_days)


if __name__ == "__main__":
    run()
