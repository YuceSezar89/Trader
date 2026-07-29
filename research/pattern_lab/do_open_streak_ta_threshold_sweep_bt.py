"""
do_open_streak + TA-kovalama percentile eşiği — IS/OOS disiplinli sweep
(24 Tem 2026, kullanıcı isteği: "percentil varyasyonlarını deneyelim").
Önceki hızlı taramada (do_open_streak_ta_combo_bt.py) eşik 85'te tepe
yapmıştı ama bu TÜM veriyle seçilmişti — 12 Tem'in overfitting dersine göre
(sabit eşik TÜM veride optimize edilirse şişer) burada eşik SADECE IS (ilk
yarı) verisiyle seçilip OOS'ta (ikinci yarı) bağımsız doğrulanıyor.

Popülasyon: pullback+likidite (gauss eşiği YOK — önceki taramada bu daha
büyük örneklem + daha yüksek $/ay veriyordu).

Kullanım: python -m research.pattern_lab.do_open_streak_ta_threshold_sweep_bt
(önce do_open_streak_ta_combo_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _econ, _stats
from research.pattern_lab.do_open_streak_ta_combo_bt import _CACHE_PATH

_THRESHOLDS = [55, 60, 65, 70, 75, 80, 85, 90, 95]
_MIN_N = 20
_PLACEBO_ITER = 300


def _kovalama_mask(df: pd.DataFrame, th: float) -> pd.Series:
    return ((df["pct_1h"] >= th) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= th) & (df["slope_4h"] > 0))


def main() -> None:
    df = pd.read_parquet(_CACHE_PATH)
    df = df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"[TA hesaplanabilir popülasyon, pullback+likidite] n={len(df)}\n")

    mid = df["ts"].iloc[len(df) // 2]
    is_df = df[df["ts"] < mid]
    oos_df = df[df["ts"] >= mid]
    is_days = (is_df["ts"].max() - is_df["ts"].min()).total_seconds() / 86400
    oos_days = (oos_df["ts"].max() - oos_df["ts"].min()).total_seconds() / 86400
    print(f"IS: {is_df['ts'].min()} .. {is_df['ts'].max()} ({is_days:.1f} gün, n={len(is_df)})")
    print(f"OOS: {oos_df['ts'].min()} .. {oos_df['ts'].max()} ({oos_days:.1f} gün, n={len(oos_df)})\n")

    print("=" * 90)
    print("EŞİK TARAMASI (SADECE IS verisiyle)")
    print("=" * 90)
    print(f"{'esik':>6} | {'IS n':>6} {'IS WR%':>7} {'IS $/ay':>10} | {'OOS n':>6} {'OOS WR%':>7} {'OOS $/ay':>10}")
    print("-" * 90)
    best_th, best_is_usd = None, -1e18
    for th in _THRESHOLDS:
        is_sub = is_df[_kovalama_mask(is_df, th)]
        oos_sub = oos_df[_kovalama_mask(oos_df, th)]
        is_e = _econ(is_sub, is_days)
        oos_e = _econ(oos_sub, oos_days)
        print(f"{th:>6} | {is_e['n']:>6} {is_e['wr'] if is_e.get('wr') is not None else '-':>7} "
              f"{is_e.get('usd_ay','-'):>10} | "
              f"{oos_e['n']:>6} {oos_e['wr'] if oos_e.get('wr') is not None else '-':>7} "
              f"{oos_e.get('usd_ay','-'):>10}")
        if is_e["n"] >= _MIN_N and is_e.get("usd_ay") is not None and is_e["usd_ay"] > best_is_usd:
            best_is_usd = is_e["usd_ay"]
            best_th = th

    print(f"\n=== IS'te en iyi eşik: {best_th} (${best_is_usd}/ay, SADECE IS verisiyle seçildi) ===")
    oos_best = oos_df[_kovalama_mask(oos_df, best_th)]
    oos_e = _econ(oos_best, oos_days)
    print(f"  Bu eşiğin OOS (görülmemiş veri) performansı: n={oos_e['n']} WR%={oos_e.get('wr')} "
          f"toplam=${oos_e.get('toplam_usd')} $/ay={oos_e.get('usd_ay')}")

    if oos_e["n"] >= _MIN_N:
        real_mean = oos_best["pnl_usd"].mean()
        pool = oos_df["pnl_usd"].to_numpy()
        rng = np.random.default_rng(42)
        n_sel = len(oos_best)
        count_ge = 0
        for _ in range(_PLACEBO_ITER):
            sample = rng.choice(pool, size=n_sel, replace=False)
            if sample.mean() >= real_mean:
                count_ge += 1
        print(f"  placebo (OOS içinde rastgele aynı-boy altküme bunu eşitler mi): %{count_ge/_PLACEBO_ITER*100:.1f}")

    print(f"\n=== Seçilen eşiğin ({best_th}) TÜM veri üzerinde tam doğrulaması ===")
    full_mask = _kovalama_mask(df, best_th)
    group, rest = df[full_mask], df[~full_mask]
    e_full = _econ(group, (df["ts"].max() - df["ts"].min()).total_seconds() / 86400)
    print(f"  grup: n={e_full['n']} WR%={e_full.get('wr')} toplam=${e_full.get('toplam_usd')} $/ay={e_full.get('usd_ay')}")
    print(f"  grup istatistik: {_stats(group['pnl_usd'].to_numpy())}")
    print(f"  geri kalan istatistik: {_stats(rest['pnl_usd'].to_numpy())}")

    if len(group) >= 30:
        g_sorted = group.sort_values("ts")
        gmid = g_sorted["ts"].iloc[len(g_sorted)//2]
        fh = _stats(g_sorted[g_sorted["ts"] < gmid]["pnl_usd"].to_numpy())
        sh = _stats(g_sorted[g_sorted["ts"] >= gmid]["pnl_usd"].to_numpy())
        print(f"  split-period: ilk yarı {fh} | ikinci yarı {sh}")

    days_span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400
    summarize(f"do_open_streak+TA eşik={best_th} (tüm veri)", group["pnl_usd"].to_numpy(), group["ts"], days_span)


if __name__ == "__main__":
    main()
