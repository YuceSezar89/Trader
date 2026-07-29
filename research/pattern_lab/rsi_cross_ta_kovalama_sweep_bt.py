"""
Madde 5: kovalama tanımındaki percentile eşiğinin (şu an sabit 80, HTML'in
varsayılanından alındı) IS/OOS disiplinli taraması — gerçekten en iyi seçim
mi? (24 Tem 2026, kullanıcı isteği)

kovalama(eşik) = (pct_1h>=eşik & slope_1h>0) OR (pct_4h>=eşik & slope_4h>0)

Disiplin: eşik SADECE IS (ilk yarı) verisiyle seçilir, OOS'ta (ikinci yarı,
görülmemiş) bağımsız doğrulanır — 12 Tem/22 Tem gauss-sweep ile aynı yöntem.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_kovalama_sweep_bt
(önce rsi_cross_ta_percentile_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_percentile_bt import _CACHE_PATH, _decompose

_THRESHOLDS = [60, 65, 70, 75, 80, 85, 90, 95]
_MIN_N = 30
_CURRENT_TH = 80


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "ort_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _kovalama_mask(df: pd.DataFrame, th: float) -> pd.Series:
    return ((df["pct_1h"] >= th) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= th) & (df["slope_4h"] > 0))


def main() -> None:
    df = pd.read_parquet(_CACHE_PATH)
    df = _add_all_up(df)
    df = df[df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    print(f"[all_up=True popülasyonu] n={len(df)}\n")

    df = df.sort_values("opened_at").reset_index(drop=True)
    mid = df["opened_at"].iloc[len(df) // 2]
    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]
    print(f"IS: n={len(is_df)}  OOS: n={len(oos_df)}\n")

    print("=" * 78)
    print("EŞİK TARAMASI (SADECE IS verisiyle)")
    print("=" * 78)
    print(f"{'esik':>6} | {'IS n':>6} {'IS WR%':>7} {'IS PF':>8} | {'OOS n':>6} {'OOS WR%':>7} {'OOS PF':>8}")
    print("-" * 70)
    best_th, best_is_pf = None, -1.0
    for th in _THRESHOLDS:
        is_sub = is_df[_kovalama_mask(is_df, th)]
        oos_sub = oos_df[_kovalama_mask(oos_df, th)]
        is_s = _stats(is_sub["fwd_ret"].to_numpy())
        oos_s = _stats(oos_sub["fwd_ret"].to_numpy())
        marker = " <- mevcut (80)" if th == _CURRENT_TH else ""
        print(f"{th:>6} | {is_s['n']:>6} {is_s['wr'] if is_s['wr'] is not None else '-':>7} "
              f"{is_s['pf'] if is_s['pf'] is not None else '-':>8} | "
              f"{oos_s['n']:>6} {oos_s['wr'] if oos_s['wr'] is not None else '-':>7} "
              f"{oos_s['pf'] if oos_s['pf'] is not None else '-':>8}{marker}")
        if is_s["n"] >= _MIN_N and is_s["pf"] is not None and isinstance(is_s["pf"], float) and is_s["pf"] > best_is_pf:
            best_is_pf = is_s["pf"]
            best_th = th

    print(f"\n=== IS'te en iyi eşik: {best_th} (PF={best_is_pf}, SADECE IS verisiyle seçildi) ===")
    oos_best = oos_df[_kovalama_mask(oos_df, best_th)]
    print(f"  Bu eşiğin OOS (görülmemiş veri) performansı: {_stats(oos_best['fwd_ret'].to_numpy())}")

    print(f"\n=== Karşılaştırma: mevcut sabit eşik (80) tüm veri ===")
    current_group = df[_kovalama_mask(df, _CURRENT_TH)]
    print(f"  n={len(current_group)}  {_stats(current_group['fwd_ret'].to_numpy())}")

    print(f"\n=== Seçilen eşiğin ({best_th}) tüm veri üzerinde tam derin doğrulaması ===")
    best_mask = _kovalama_mask(df, best_th)
    best_group, rest = df[best_mask], df[~best_mask]
    df["_g"] = best_mask
    _deep_validate(f"kovalama eşik={best_th}", best_group, rest, df)

    print(f"\n  PF şüphesi (medyan): {_decompose(best_group['fwd_ret'].to_numpy())}")
    days_span = (best_group["opened_at"].max() - best_group["opened_at"].min()).total_seconds() / 86400
    summarize(f"kovalama eşik={best_th} — fwd_ret% serisi", best_group["fwd_ret"].to_numpy(), best_group["opened_at"], days_span)


if __name__ == "__main__":
    main()
