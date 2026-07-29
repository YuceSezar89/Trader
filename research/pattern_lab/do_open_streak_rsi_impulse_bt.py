"""
"Devis'So Trend %Rsi Change" Pine göstergesinin RSI-itki mantığı do_open_streak'e
filtre olarak test ediliyor (22 Tem 2026, kullanıcının paylaştığı Pine kodu).

Pine formülü BİREBİR:
  rsiChange = RSI(14)'ün bar-bar değişimi
  longPower  = rsiChange > 2.0  ? rsiChange  : 0   (ani YUKARI itki gücü)
  shortPower = rsiChange < -2.0 ? -rsiChange : 0   (ani AŞAĞI itki gücü)
  mavi  = EMA(longPower, 5)
  pembe = EMA(shortPower, 5)
  trendLine = mavi - pembe   (net itki gücü)

do_open_streak'in her tetiklendiği barda (3. mum) bu trendLine değeri
hesaplanıp, GERÇEKÇİ bar-bar SL/timeout P&L'iyle (do_open_streak_full_
clean_bt.py'nin YENİ mantığı — pullback+likidite+tavan dahil) ilişkisi
test ediliyor: net RSI-itki gücü YÜKSEKSE (yukarı itkiler baskınsa) sinyal
daha mı güvenilir?

Kullanım: python -m research.pattern_lab.do_open_streak_rsi_impulse_bt
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from indicators.core import calculate_atr, calculate_rsi
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
    _detect_events,
    _do_break_gate,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from signals.do_kirilimi import _daily_open

RSI_LEN = 14
IMPULSE_LIMIT = 2.0
SMOOTH_LEN = 5
_PLACEBO_ITER = 300


def _rsi_impulse_trendline(close_series: pd.Series) -> np.ndarray:
    rsi = calculate_rsi(pd.DataFrame({"close": close_series}), period=RSI_LEN, price_col="close")
    rsi_change = rsi.diff()
    long_power = rsi_change.where(rsi_change > IMPULSE_LIMIT, 0.0)
    short_power = (-rsi_change).where(rsi_change < -IMPULSE_LIMIT, 0.0)
    mavi = long_power.ewm(span=SMOOTH_LEN, adjust=False).mean()
    pembe = short_power.ewm(span=SMOOTH_LEN, adjust=False).mean()
    return (mavi - pembe).to_numpy()


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

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()
        trend_line = _rsi_impulse_trendline(g["close"])

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
            tl = trend_line[trig]
            if not np.isfinite(tl):
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
                "symbol": sym, "ts": ts.iloc[trig], "trend_line": tl,
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, toplam olay (YENİ mantık: Gauss+pullback+likidite): {len(records)}\n")
    return pd.DataFrame(records)


def main() -> None:
    df = _collect()
    if df.empty:
        print("Olay yok.")
        return

    print("=== [1] Korelasyon (RSI-itki net gücü trendLine, ham) ===")
    rho, p = spearmanr(df["trend_line"], df["pnl_usd"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    print("\n=== [2] Çeyrek kırılımı ===")
    try:
        q = pd.qcut(df["trend_line"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
        g = df.groupby(q, observed=True)["pnl_usd"].agg(ort="mean", n="count", toplam="sum")
        print(g.to_string())
    except ValueError:
        print("  çeyrek hesaplanamadı (tekrarlı değer)")

    print("\n=== [3] trendLine > 0 (net itki YUKARI baskın) vs <= 0 ===")
    pos_grp = df[df["trend_line"] > 0]
    neg_grp = df[df["trend_line"] <= 0]
    print(f"  trendLine>0: n={len(pos_grp)}  toplam=${pos_grp['pnl_usd'].sum():.2f}  ort=${pos_grp['pnl_usd'].mean():.2f}  "
          f"WR%={_stats(pos_grp['pnl_pct'].to_numpy()).get('wr','-')}")
    print(f"  trendLine<=0: n={len(neg_grp)}  toplam=${neg_grp['pnl_usd'].sum():.2f}  ort=${neg_grp['pnl_usd'].mean():.2f}  "
          f"WR%={_stats(neg_grp['pnl_pct'].to_numpy()).get('wr','-')}")

    real_gap = pos_grp["pnl_usd"].mean() - neg_grp["pnl_usd"].mean() if len(pos_grp) and len(neg_grp) else float("nan")
    rng = np.random.default_rng(42)
    labels = (df["trend_line"] > 0).to_numpy()
    target = df["pnl_usd"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"  placebo: gerçek ort-$ farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(pos_grp) >= 30:
        p_sorted = pos_grp.sort_values("ts")
        mid = p_sorted["ts"].iloc[len(p_sorted)//2]
        fh = _stats(p_sorted[p_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(p_sorted[p_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"  split-period (trendLine>0 içi): ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print("  split-period için örneklem yetersiz")


if __name__ == "__main__":
    main()
