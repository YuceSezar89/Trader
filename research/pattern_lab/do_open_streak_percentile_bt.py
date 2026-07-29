"""
do_open_streak — tf_alignment_live'daki "akranlarına göre percentile" mantığının
Gauss/long_perc üzerinde denenmesi (22 Tem 2026, kullanıcı isteği).

tf_alignment_live: bir aday, O AN bekleyen AYNI YÖNDEKİ diğer adaylara göre
erken_pct'i %80+ percentile'daysa açılır (sabit eşik değil, akranlarına göre
göreceli seçicilik).

Burada aynı fikir do_open_streak'in Gauss büyüklüğüne (gauss_val/long_perc)
uygulanıyor: her olay, tetiklendiği andan geriye 24 saatlik pencerede
tetiklenmiş TÜM diğer do_open_streak olaylarına göre gauss_val percentile'ı
hesaplanır — sabit GAUSS_THRESHOLD=4.5 yerine (ya da onun ÜSTÜNE), "o gün
diğer adaylara göre en güçlüsü müydü" sorusu soruluyor.

Temel popülasyon: do_open_streak_full_clean_bt.py'nin GÜNCEL (pullback+tavan+
likidite dahil) mantığıyla tespit edilen olaylar — gerçekçi bar-bar SL/timeout
P&L'i zaten hesaplanmış durumda, üzerine sadece percentile katmanı ekleniyor.

Kullanım: python -m research.pattern_lab.do_open_streak_percentile_bt
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.pattern_lab.do_open_streak_full_clean_bt import (
    DAYS,
    FEE_RATE,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    STREAK_TH,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _detect_events,
    _do_break_gate,
    _econ,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from indicators.core import calculate_atr
from signals.do_kirilimi import _daily_open

_PEER_WINDOW_HOURS = 24
_PLACEBO_ITER = 300
_PERCENTILE_CUTOFF = 0.80


def _collect_events() -> pd.DataFrame:
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
            if gauss_val < 1.0:  # sabit eşik YOK burada, sadece anlamsız-sıfır ayıklanıyor
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
                "symbol": sym, "ts": ts.iloc[trig], "gauss_val": gauss_val,
                "long_perc": long_perc, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, GAUSS_THRESHOLD'suz (sadece pullback+likidite) toplam olay: {len(records)}\n")
    return pd.DataFrame(records)


def _add_peer_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Her olay için, geriye dönük 24 saatlik pencerede tetiklenmiş TÜM diğer
    olaylara göre gauss_val percentile'ı (0-1). Kendisi dahil edilir (tf_alignment
    ile aynı: 'şu an izlenen herkes' arasında kendi sırası)."""
    df = df.sort_values("ts").reset_index(drop=True)
    ts = df["ts"].to_numpy()
    gauss = df["gauss_val"].to_numpy()
    n = len(df)
    percentiles = np.zeros(n)
    window = pd.Timedelta(hours=_PEER_WINDOW_HOURS)
    lo = 0
    for i in range(n):
        while ts[i] - pd.Timestamp(ts[lo]) > window:
            lo += 1
        peers = gauss[lo:i + 1]
        percentiles[i] = (peers <= gauss[i]).mean()
    df["peer_percentile"] = percentiles
    return df


def main() -> None:
    df = _collect_events()
    if df.empty:
        print("Olay yok.")
        return
    df = _add_peer_percentile(df)

    days_span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400

    print("=== [1] Sabit eşik (GAUSS_THRESHOLD>=4.5, mevcut sistem) referans ===")
    fixed = df[df["gauss_val"] >= 4.5]
    e = _econ(fixed, days_span)
    print(f"  n={e['n']}  WR%={e['wr']}  toplam=${e['toplam_usd']}  işlem/ay={e['islem_ay']}  $/ay={e['usd_ay']}")

    print(f"\n=== [2] Akran-percentile (24s pencerede >=%{_PERCENTILE_CUTOFF*100:.0f}) ===")
    top = df[df["peer_percentile"] >= _PERCENTILE_CUTOFF]
    rest = df[df["peer_percentile"] < _PERCENTILE_CUTOFF]
    e_top = _econ(top, days_span)
    e_rest = _econ(rest, days_span)
    print(f"  top   : n={e_top['n']}  WR%={e_top['wr']}  toplam=${e_top['toplam_usd']}  işlem/ay={e_top['islem_ay']}  $/ay={e_top['usd_ay']}")
    print(f"  geri kalan: n={e_rest['n']}  WR%={e_rest['wr']}  toplam=${e_rest['toplam_usd']}  işlem/ay={e_rest['islem_ay']}  $/ay={e_rest['usd_ay']}")

    rho, p = spearmanr(df["peer_percentile"], df["pnl_usd"])
    print(f"\n  korelasyon (peer_percentile, ham): rho={rho:+.4f} (p={p:.4f})")

    real_gap = top["pnl_usd"].mean() - rest["pnl_usd"].mean() if len(top) and len(rest) else float("nan")
    rng = np.random.default_rng(42)
    labels = (df["peer_percentile"] >= _PERCENTILE_CUTOFF).to_numpy()
    target = df["pnl_usd"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"  placebo: gerçek ort-$ farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(top) >= 30:
        t_sorted = top.sort_values("ts")
        mid = t_sorted["ts"].iloc[len(t_sorted)//2]
        fh = _stats(t_sorted[t_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(t_sorted[t_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"  split-period (top grubu içi): ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print("  split-period için örneklem yetersiz")

    print("\n=== [3] Çeyrek kırılımı (peer_percentile) ===")
    try:
        q = pd.qcut(df["peer_percentile"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
        g = df.groupby(q, observed=True)["pnl_usd"].agg(ort="mean", n="count", toplam="sum")
        print(g.to_string())
    except ValueError:
        print("  çeyrek hesaplanamadı (tekrarlı değer)")


if __name__ == "__main__":
    main()
