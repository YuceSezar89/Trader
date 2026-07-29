"""
do_open_streak_all_up_bt.py ile AYNI all_up testi, ama Daily Open yerine
Hourly Open kırılım senaryosunda (22 Tem 2026, kullanıcı isteği) — hourly
senaryoda olay sayısı ~3 kat fazla olduğu için "bir önceki aynı sembol
olayına göre delta" karşılaştırması çok daha sık/yakın zamanlı, all_up
mantığının anlamlı olma ihtimali daha yüksek.

Kullanım: python -m research.pattern_lab.hourly_open_streak_all_up_bt
"""

import numpy as np
import pandas as pd

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
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from research.pattern_lab.do_open_streak_all_up_bt import _add_all_up, _deep_validate, _fetch_with_buy_volume
from research.pattern_lab.do_open_streak_hourly_clean_bt import _do_break_gate, _detect_events
from research.pattern_lab.hourly_open_streak_recency_weight_bt import _hourly_open
from utils.vpmv import compute_components

_COMPONENT_WINDOW = 60


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch_with_buy_volume(bad)
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
        buy_vol = g["buy_volume"].fillna(0).to_numpy(float)
        sell_vol = vol - buy_vol
        usd_vol = vol * c

        hourly_open = _hourly_open(ts, o)
        gate = _do_break_gate(o, c, hourly_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()

        for streak_start, trig in events:
            if trig + 1 >= len(c) or trig < _COMPONENT_WINDOW:
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

            window = g.iloc[trig - _COMPONENT_WINDOW + 1: trig + 1].copy()
            window["buy_volume"] = buy_vol[trig - _COMPONENT_WINDOW + 1: trig + 1]
            window["sell_volume"] = sell_vol[trig - _COMPONENT_WINDOW + 1: trig + 1]
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long", volume_mode="real")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if any(pd.isna(v) for v in (vol_s, mom_s, vlt_s, prc_s)):
                continue

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
                "vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s,
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, toplam olay (saatlik-open kırılımı): {len(records)}\n")
    return pd.DataFrame(records)


def main() -> None:
    raw = _collect()
    if raw.empty:
        print("Olay yok.")
        return
    df = _add_all_up(raw)
    print(f"[delta hesaplandı] {len(df)} sinyal (bir önceki aynı sembol olayı olanlar)\n")

    _deep_validate("all_up=True", df[df["all_up"]], df[~df["all_up"]])


if __name__ == "__main__":
    main()
