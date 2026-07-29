"""
Saatlik-Open kırılım senaryosu (do_open_streak_hourly.pine) + son-muma-ağırlıklı
momentum testi bir arada (22 Tem 2026, kullanıcı isteği).

do_open_streak_recency_weight_bt.py ile AYNI weighted_score/back_loaded_ratio
metrikleri, ama tetikleyici olay DAILY Open kırılımı değil, HER SAATTE BİR
yenilenen Hourly Open kırılımı (do_open_streak_hourly.pine ile birebir mantık).

Kullanım: python -m research.pattern_lab.hourly_open_streak_recency_weight_bt
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from indicators.core import calculate_atr
from research.pattern_lab.do_open_streak_full_clean_bt import (
    DAYS,
    FEE_RATE,
    GAUSS_THRESHOLD,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from research.pattern_lab.do_open_streak_hourly_clean_bt import _do_break_gate, _detect_events
from research.pattern_lab.do_open_streak_recency_weight_bt import _analyze_metric

_PLACEBO_ITER = 300


def _hourly_open(ts: pd.Series, o: np.ndarray) -> np.ndarray:
    """Her tam saat başında (dk=00) yenilenen açılış — do_open_streak_hourly.pine
    ile birebir (whole-hour sınırı zaman dilimi kaymasından etkilenmez)."""
    is_new_hour = (ts.dt.minute == 0).to_numpy()
    hourly_open = np.where(is_new_hour, o, np.nan)
    return pd.Series(hourly_open).ffill().to_numpy()


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    records = []
    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        usd_vol = vol * c

        hourly_open = _hourly_open(ts, o)
        gate = _do_break_gate(o, c, hourly_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()

        for streak_start, trig in events:
            if trig + 1 >= len(c):
                continue
            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 != trig:
                continue
            start_low = l[bar1]
            long_perc = (h[trig] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val < GAUSS_THRESHOLD:
                continue
            if not _pullback_ok(h, l, bar1, bar2, bar3):
                continue
            liq_start = max(0, trig - LIQ_WINDOW_BARS)
            avg_liq = float(usd_vol[liq_start:trig].mean()) if trig > liq_start else 0.0
            if avg_liq < MIN_LIQUIDITY_USD:
                continue
            atr_val = atr[trig]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue

            c1 = (c[bar1] - o[bar1]) / o[bar1] * 100.0
            c2 = (c[bar2] - o[bar2]) / o[bar2] * 100.0
            c3 = (c[bar3] - o[bar3]) / o[bar3] * 100.0
            total = c1 + c2 + c3
            if total <= 0:
                continue
            weighted_score = 1 * c1 + 2 * c2 + 3 * c3
            back_loaded_ratio = c3 / total

            entry_price = c[trig]
            sl_price = entry_price - 3.0 * atr_val
            sl_dist = entry_price - sl_price
            if sl_dist <= 0:
                continue
            pos = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)
            pnl_pct, reason, hold_bars = _simulate_exit(c, l, trig, entry_price, sl_price)
            fee = pos * FEE_RATE * 2
            pnl_usd = pnl_pct / 100 * pos - fee
            records.append({
                "symbol": sym, "ts": ts.iloc[trig],
                "c1": c1, "c2": c2, "c3": c3,
                "weighted_score": weighted_score, "back_loaded_ratio": back_loaded_ratio,
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, toplam olay (saatlik-open kırılımı): {len(records)}\n")
    return pd.DataFrame(records)


def main() -> None:
    df = _collect()
    if df.empty:
        print("Olay yok.")
        return
    _analyze_metric(df, "weighted_score", "[Saatlik-Open] Son muma ağırlıklı toplam")
    _analyze_metric(df, "back_loaded_ratio", "[Saatlik-Open] Son mumun payı")


if __name__ == "__main__":
    main()
