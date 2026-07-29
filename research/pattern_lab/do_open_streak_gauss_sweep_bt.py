"""
GAUSS_THRESHOLD tarama — bugünkü temiz altyapıyla (pullback+likidite+tavan
dahil, gerçekçi bar-bar SL/timeout replay), 22 Tem 2026.

12 Tem'deki ders unutulmadı: "en iyi eşiği bul" tarzı optimizasyon overfitting
riski taşıyor (o zaman IS'te monotonik artan PF, OOS'ta çöküp placebo %39
vermişti). Bu yüzden burada SIKI IS/OOS ayrımı var: eşik SADECE IS (ilk
yarı) verisine bakılarak seçiliyor, sonra o eşik OOS'ta (ikinci yarı,
seçim sırasında hiç görülmemiş veri) BAĞIMSIZ olarak doğrulanıyor.

Kullanım: python -m research.pattern_lab.do_open_streak_gauss_sweep_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_percentile_bt import _collect_events
from research.pattern_lab.do_open_streak_full_clean_bt import GAUSS_THRESHOLD as CURRENT_THRESHOLD

_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
_PLACEBO_ITER = 300
_MIN_N = 30


def _econ_row(sub: pd.DataFrame, days: float) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "wr": None, "toplam": None, "usd_ay": None}
    wr = round(float((sub["pnl_usd"] > 0).mean() * 100), 1)
    total = float(sub["pnl_usd"].sum())
    per_month = n / days * 30 if days > 0 else 0
    return {
        "n": n, "wr": wr, "toplam": round(total, 2),
        "usd_ay": round(per_month * (total / n), 2),
    }


def main() -> None:
    df = _collect_events()
    if df.empty:
        print("Olay yok.")
        return
    df = df.sort_values("ts").reset_index(drop=True)

    mid = df["ts"].iloc[len(df) // 2]
    is_df = df[df["ts"] < mid]
    oos_df = df[df["ts"] >= mid]
    is_days = (is_df["ts"].max() - is_df["ts"].min()).total_seconds() / 86400
    oos_days = (oos_df["ts"].max() - oos_df["ts"].min()).total_seconds() / 86400
    print(f"\nIS (kalibrasyon): {is_df['ts'].min()} .. {is_df['ts'].max()} ({is_days:.1f} gün, n={len(is_df)})")
    print(f"OOS (doğrulama) : {oos_df['ts'].min()} .. {oos_df['ts'].max()} ({oos_days:.1f} gün, n={len(oos_df)})\n")

    print(f"{'esik':>6} | {'IS n':>6} {'IS WR%':>7} {'IS $/ay':>10} | {'OOS n':>6} {'OOS WR%':>7} {'OOS $/ay':>10}")
    print("-" * 70)
    best_threshold, best_is_usd = None, -1e18
    rows = []
    for th in _THRESHOLDS:
        is_sub = is_df[is_df["gauss_val"] >= th]
        oos_sub = oos_df[oos_df["gauss_val"] >= th]
        is_e = _econ_row(is_sub, is_days)
        oos_e = _econ_row(oos_sub, oos_days)
        rows.append((th, is_e, oos_e))
        marker = " <- mevcut (4.5)" if abs(th - CURRENT_THRESHOLD) < 1e-9 else ""
        print(f"{th:>6} | {is_e['n']:>6} {is_e['wr'] if is_e['wr'] is not None else '-':>7} "
              f"{is_e['usd_ay'] if is_e['usd_ay'] is not None else '-':>10} | "
              f"{oos_e['n']:>6} {oos_e['wr'] if oos_e['wr'] is not None else '-':>7} "
              f"{oos_e['usd_ay'] if oos_e['usd_ay'] is not None else '-':>10}{marker}")
        if is_e["n"] >= _MIN_N and is_e["usd_ay"] is not None and is_e["usd_ay"] > best_is_usd:
            best_is_usd = is_e["usd_ay"]
            best_threshold = th

    print(f"\n=== IS'te en iyi görünen eşik: {best_threshold} (${best_is_usd}/ay, SADECE IS verisiyle seçildi) ===")
    oos_sub = oos_df[oos_df["gauss_val"] >= best_threshold]
    oos_e = _econ_row(oos_sub, oos_days)
    print(f"  Bu eşiğin OOS (görülmemiş veri) performansı: n={oos_e['n']} WR%={oos_e['wr']} "
          f"toplam=${oos_e['toplam']} $/ay={oos_e['usd_ay']}")

    if oos_e["n"] >= _MIN_N:
        real_mean = oos_sub["pnl_usd"].mean()
        pool = oos_df["pnl_usd"].to_numpy()
        rng = np.random.default_rng(42)
        n_sel = len(oos_sub)
        count_ge = 0
        for _ in range(_PLACEBO_ITER):
            sample = rng.choice(pool, size=n_sel, replace=False)
            if sample.mean() >= real_mean:
                count_ge += 1
        print(f"  placebo (OOS içinde rastgele aynı-boy altküme bunu eşitler mi): %{count_ge/_PLACEBO_ITER*100:.1f}")
    else:
        print("  OOS örneklemi placebo için yetersiz")

    print(f"\n=== Karşılaştırma: mevcut sabit eşik (4.5) tüm veri ===")
    full_at_current = df[df["gauss_val"] >= CURRENT_THRESHOLD]
    all_days = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400
    e = _econ_row(full_at_current, all_days)
    print(f"  n={e['n']} WR%={e['wr']} toplam=${e['toplam']} $/ay={e['usd_ay']}")


if __name__ == "__main__":
    main()
