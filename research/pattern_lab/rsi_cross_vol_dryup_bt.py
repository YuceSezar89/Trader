"""
"Hacim kuruması" (düşük vol_score) — TEMİZ yöntemle, RSI_Cross(9,24) genelinde
test (21 Tem 2026, kullanıcı isteği).

all_up'ın canlı-formül veri toplama altyapısını (rsi_cross_allup_candleshape_
clean_bt.py — utils/vpmv.py::compute_components, yönlü buy/sell hacmi) yeniden
kullanır — ama DELTA (bir önceki sinyale göre artış) yerine, `vol` skorunun
HAM/anlık seviyesine (0-100) bakar: DÜŞÜK vol_score = hacim kurumuş = daha
sağlıklı devam mı, YÜKSEK vol_score = kalabalık/gürültülü = tükenmeye yakın mı?

Bu, do_open_streak'teki consecutiveUp bulgusunun (düşük hacim birikimi iyi)
daha genel bir popülasyonda (RSI_Cross, tüm semboller/yönler) sınanmasıdır.

Kullanım: python -m research.pattern_lab.rsi_cross_vol_dryup_bt
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import (
    DIRECTIONS,
    _bad_symbols,
    _collect,
    _conn,
)

_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _analyze(df: pd.DataFrame, direction: str) -> None:
    print(f"\n{'='*70}\nRSI_Cross(9,24) — {direction} (vol_score, n={len(df):,})\n{'='*70}")

    rho, p = spearmanr(df["vol"], df["fwd_ret"])
    print(f"[1] Korelasyon (vol_score, ham seviye): rho={rho:+.4f} (p={p:.4f})")

    try:
        tercile = pd.qcut(df["vol"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
    except ValueError:
        try:
            tercile = pd.qcut(df["vol"], 2, labels=["1.düşük", "2.yüksek"], duplicates="drop")
        except ValueError:
            tercile = None
    if tercile is not None:
        g = df.groupby(tercile, observed=True)["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
        print(g.to_string())
    else:
        print("  (çeyrek/tercil hesaplanamadı — vol_score bu grupta çok fazla tekrarlı değer taşıyor)")

    low_q = df[df["vol"] <= df["vol"].quantile(0.25)]
    high_q = df[df["vol"] >= df["vol"].quantile(0.75)]
    print(f"\n  en düşük %25 (hacim kurumuş): {_stats(low_q['fwd_ret'].to_numpy())}")
    print(f"  en yüksek %25 (kalabalık): {_stats(high_q['fwd_ret'].to_numpy())}")

    rng = np.random.default_rng(42)
    vals = df["vol"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
    )
    print(f"\n  placebo (korelasyon karıştırma): %{count_ge/_PLACEBO_ITER*100:.1f}")

    if len(low_q) >= 40:
        lq_sorted = low_q.sort_values("opened_at")
        mid = lq_sorted["opened_at"].iloc[len(lq_sorted)//2]
        fh = _stats(lq_sorted[lq_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(lq_sorted[lq_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"  en düşük %25 içinde split-period: ilk yarı {fh} | ikinci yarı {sh}")

    crash = low_q[low_q["opened_at"] < _REGIME_SPLIT]
    recovery = low_q[low_q["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 50:
            print(f"  Split — {label} (n={len(sub)}): {_stats(sub['fwd_ret'].to_numpy())}")
        else:
            print(f"  Split — {label}: yetersiz örnek (n={len(sub)})")


_MIN_TF_N = 150


def main() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")
    for direction in DIRECTIONS:
        print(f"\n[{direction}] sinyaller toplanıyor...")
        raw = _collect(conn, "RSI_Cross(9,24)", direction, bad)
        if raw.empty:
            print(f"{direction}: veri yok.")
            continue

        print(f"\n{'#'*70}\n# {direction} — TÜM TF'LER HAVUZLANMIŞ (referans, TF ayrımı YOK)\n{'#'*70}")
        _analyze(raw, direction)

        print(f"\n{'#'*70}\n# {direction} — TF BAZLI KIRILIM\n{'#'*70}")
        for interval in sorted(raw["interval"].unique()):
            sub = raw[raw["interval"] == interval]
            if len(sub) < _MIN_TF_N:
                print(f"\n[{interval}] yetersiz örnek (n={len(sub)}), atlanıyor")
                continue
            _analyze(sub, f"{direction} / {interval}")
    conn.close()


if __name__ == "__main__":
    main()
