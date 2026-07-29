"""
do_open_streak (YENİ mantık — pullback+likidite+tavan, canlı `signals/
do_open_streak.py` ile birebir) — giriş anındaki Shannon Entropy'nin
(pencere=50, kutu=5 — entropy_clean_forward_return_bt.py'de en iyi çıkan
kombinasyon) trade sonucunu öngörüp öngörmediği. (24 Tem 2026, kullanıcı
isteği — entropy bulgusunun ham indikatörlerin ötesinde, TAMAMEN FARKLI
bir sinyal mekanizmasında (DO kırılımı + streak, 15m) da geçerli olup
olmadığı.)

do_open_streak_full_clean_bt.py'nin event-detection + SL/timeout replay
motoruyla BİREBİR aynı — sadece her trigger bar'da entropy ekleniyor
(15m close log-return'leri, trigger'dan önceki 50 bar).

Kullanım: python -m research.pattern_lab.do_open_streak_entropy_bt
"""

import numpy as np
import pandas as pd

from indicators.core import calculate_atr
from research.pattern_lab.do_open_streak_full_clean_bt import (
    FEE_RATE,
    GAUSS_THRESHOLD,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    SL_ATR_MULT,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _daily_open,
    _detect_events,
    _do_break_gate,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from research.pattern_lab.entropy_clean_forward_return_bt import _entropy

_WINDOW = 50
_BINS = 5


def run_with_entropy() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar\n")

    records = []
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        usd_vol = vol * c
        log_ret = np.diff(np.log(c))

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)

        atr_series = calculate_atr(g[["high", "low", "close"]], period=14)
        atr = atr_series.to_numpy()

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

            atr_val = atr[trig]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue

            if not _pullback_ok(h, l, bar1, bar2, bar3):
                continue
            liq_start = max(0, trig - LIQ_WINDOW_BARS)
            avg_liq = float(usd_vol[liq_start:trig].mean()) if trig > liq_start else 0.0
            if avg_liq < MIN_LIQUIDITY_USD:
                continue

            if trig < _WINDOW:
                continue
            ent = _entropy(log_ret[trig - _WINDOW : trig], _BINS)
            if ent is None:
                continue

            entry_price = c[trig]
            sl_price = entry_price - SL_ATR_MULT * atr_val
            pnl_pct, reason, hold_bars = _simulate_exit(c, l, trig, entry_price, sl_price)

            sl_dist = entry_price - sl_price
            pos = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD) if sl_dist > 0 else 0
            fee = pos * FEE_RATE * 2
            pnl_usd = pnl_pct / 100 * pos - fee
            records.append(
                {
                    "symbol": sym, "ts": ts.iloc[trig], "pnl_usd": pnl_usd,
                    "reason": reason, "entropy": ent,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    print("=" * 78)
    print("do_open_streak (YENİ mantık) — giriş anı Shannon Entropy ile trade sonucu")
    print(f"(pencere={_WINDOW}, kutu={_BINS})")
    print("=" * 78)

    df = run_with_entropy()
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"entropy dağılımı: min={df['entropy'].min():.3f} medyan={df['entropy'].median():.3f} "
          f"max={df['entropy'].max():.3f}")

    print("\n-- Kartil bucket'ları (entropy) --")
    df["q"] = pd.qcut(df["entropy"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"], duplicates="drop")
    for q in df["q"].cat.categories:
        sub = df[df["q"] == q]
        rng = f"{sub['entropy'].min():.3f}-{sub['entropy'].max():.3f}"
        print(f"  {str(q):14s} E={rng:16s} {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: entropy <= X olan popülasyon --")
    for th in [0.6, 0.7, 0.8, 0.9]:
        sub = df[df["entropy"] <= th]
        print(f"  <= {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["entropy"], df["pnl_usd"])[0, 1]
    print(f"\nentropy vs pnl_usd korelasyonu: {corr:.3f}")

    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["entropy"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"placebo (karıştırmada eşit/büyük |rho| sıklığı): %{pct_ge:.1f}")

    if len(df) >= 30:
        d_sorted = df.sort_values("ts")
        mid = d_sorted["ts"].iloc[len(d_sorted) // 2]
        fh = _stats(d_sorted[d_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(d_sorted[d_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"\nsplit-period (tüm popülasyon): ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
