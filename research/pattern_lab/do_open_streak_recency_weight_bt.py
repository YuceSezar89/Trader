"""
Streak'in SON mumunda mı yoksa İLK mumunda mı ağırlıklı hareket ettiği testi
(22 Tem 2026, kullanıcı fikri) — TEMİZ yöntem (gün sınırı düzeltmesi dahil,
bkz. do_open_streak_hourly_clean_bt.py'deki UTC 00:00 düzeltmesi).

İki aday metrik, her ikisi de sadece 3 mumun kendi open->close yüzde
değişimlerinden (c1, c2, c3) hesaplanıyor:
  weighted_score = 1*c1 + 2*c2 + 3*c3        (son muma orantılı ağırlık)
  back_loaded_ratio = c3 / (c1+c2+c3)        (toplam hareketin son mumdaki payı)

Hipotez: hareket SON mumda yoğunlaşmışsa (ivmeleniyor) mu, yoksa İLK mumda
yoğunlaşmışsa (tükeniyor) mu daha güvenilir?

Kullanım: python -m research.pattern_lab.do_open_streak_recency_weight_bt
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
    STREAK_TH,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from research.pattern_lab.do_open_streak_hourly_clean_bt import (
    _day_and_hour_marks,
    _do_break_gate,
    _detect_events,
)

_PLACEBO_ITER = 300


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

        is_new_day, _, _ = _day_and_hour_marks(ts)
        daily_open = np.where(is_new_day, o, np.nan)
        daily_open = pd.Series(daily_open).ffill().to_numpy()

        gate = _do_break_gate(o, c, daily_open)
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

    print(f"analize giren sembol: {n_syms}, toplam olay: {len(records)}\n")
    return pd.DataFrame(records)


def _analyze_metric(df: pd.DataFrame, col: str, label: str) -> None:
    print(f"\n{'='*70}\n[{label}] metrik: {col}\n{'='*70}")
    rho, p = spearmanr(df[col], df["pnl_usd"])
    print(f"  Korelasyon: rho={rho:+.4f} (p={p:.4f})")

    try:
        q = pd.qcut(df[col], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
        g = df.groupby(q, observed=True)["pnl_usd"].agg(ort="mean", n="count", toplam="sum")
        print(g.to_string())
    except ValueError:
        print("  çeyrek hesaplanamadı (tekrarlı değer)")

    top_q = df[df[col] >= df[col].quantile(0.75)]
    bot_q = df[df[col] <= df[col].quantile(0.25)]
    print(f"\n  en yüksek %25: n={len(top_q)} toplam=${top_q['pnl_usd'].sum():.2f} "
          f"WR%={_stats(top_q['pnl_pct'].to_numpy()).get('wr','-')}")
    print(f"  en düşük  %25: n={len(bot_q)} toplam=${bot_q['pnl_usd'].sum():.2f} "
          f"WR%={_stats(bot_q['pnl_pct'].to_numpy()).get('wr','-')}")

    real_gap = top_q["pnl_usd"].mean() - bot_q["pnl_usd"].mean() if len(top_q) and len(bot_q) else float("nan")
    rng = np.random.default_rng(42)
    vals = df[col].to_numpy()
    target = df["pnl_usd"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
    )
    print(f"  placebo (korelasyon karıştırma): %{count_ge/_PLACEBO_ITER*100:.1f}")

    if len(top_q) >= 30:
        t_sorted = top_q.sort_values("ts")
        mid = t_sorted["ts"].iloc[len(t_sorted)//2]
        fh = _stats(t_sorted[t_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(t_sorted[t_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"  en yüksek %25 içinde split-period: ilk yarı {fh} | ikinci yarı {sh}")


def main() -> None:
    df = _collect()
    if df.empty:
        print("Olay yok.")
        return
    _analyze_metric(df, "weighted_score", "Son muma ağırlıklı toplam (1*c1+2*c2+3*c3)")
    _analyze_metric(df, "back_loaded_ratio", "Son mumun payı (c3 / toplam)")


if __name__ == "__main__":
    main()
