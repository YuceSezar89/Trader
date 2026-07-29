"""
Sinyal-kalabalığı (concurrent_count, 4sa pencere) için basit eşik — IS/OOS
disiplinli sweep (24 Tem 2026, kullanıcı isteği: "önce basit eşiği test
edelim sonra ML'e geçelim"). [[project_combo_clustering_feature_24tem]]'de
korelasyon/çeyreklik bulundu, burada gerçek bir filtre eşiği aranıyor:
"son 4 saatte eşyönlü sinyal sayısı <= eşik ise işlemi al, değilse atla."

Kullanım: python -m research.pattern_lab.combo_clustering_threshold_sweep_bt
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize

_FILES = {
    ("RSI_Cross", "Long"): "_cache_replay_long.parquet",
    ("RSI_Cross", "Short"): "_cache_replay_short.parquet",
    ("HA_Cross", "Long"): "_cache_replay_ha_long.parquet",
    ("HA_Cross", "Short"): "_cache_replay_ha_short.parquet",
}
_WINDOW_HOURS = 4
_THRESHOLDS = [3, 5, 7, 10, 13, 17, 20, 25, 999]  # 999 = filtresiz (üst sınır yok)
_MIN_N = 20


def _load() -> pd.DataFrame:
    frames = []
    for (indicator, direction), fname in _FILES.items():
        path = os.path.join(os.path.dirname(__file__), fname)
        df = pd.read_parquet(path)
        df["indicator"] = indicator
        df["direction"] = direction
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    return df.sort_values("opened_at").reset_index(drop=True)


def _concurrent_count(times: np.ndarray, window_hours: float) -> np.ndarray:
    window_ns = window_hours * 3600 * 1e9
    counts = np.zeros(len(times), dtype=int)
    left = 0
    for i in range(len(times)):
        while times[i] - times[left] > window_ns:
            left += 1
        counts[i] = i - left
    return counts


def _stats(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return {"n": 0, "wr": None, "toplam_$": None, "pf": None}
    g, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "n": len(pnl), "wr": round(float((pnl > 0).mean() * 100), 1),
        "toplam_$": round(float(pnl.sum()), 2),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _run_for_direction(df: pd.DataFrame, direction: str) -> None:
    sub = df[df["direction"] == direction].sort_values("opened_at").reset_index(drop=True)
    times = sub["opened_at"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    sub["conc"] = _concurrent_count(times, _WINDOW_HOURS)

    mid = sub["opened_at"].iloc[len(sub) // 2]
    is_df = sub[sub["opened_at"] < mid]
    oos_df = sub[sub["opened_at"] >= mid]

    print("=" * 78)
    print(f"{direction} — kalabalık eşiği taraması (SADECE IS verisiyle, n={len(sub)}, IS={len(is_df)} OOS={len(oos_df)})")
    print("=" * 78)
    print(f"{'esik':>6} | {'IS n':>6} {'IS WR%':>7} {'IS $/işl':>9} | {'OOS n':>6} {'OOS WR%':>7} {'OOS $/işl':>9}")
    print("-" * 70)
    best_th, best_is_avg = None, -1e18
    for th in _THRESHOLDS:
        is_sub = is_df[is_df["conc"] <= th]
        oos_sub = oos_df[oos_df["conc"] <= th]
        is_avg = is_sub["pnl_usd"].mean() if len(is_sub) else float("nan")
        oos_avg = oos_sub["pnl_usd"].mean() if len(oos_sub) else float("nan")
        is_wr = round((is_sub["pnl_usd"] > 0).mean() * 100, 1) if len(is_sub) else None
        oos_wr = round((oos_sub["pnl_usd"] > 0).mean() * 100, 1) if len(oos_sub) else None
        print(f"{th:>6} | {len(is_sub):>6} {is_wr if is_wr is not None else '-':>7} {is_avg:>9.3f} | "
              f"{len(oos_sub):>6} {oos_wr if oos_wr is not None else '-':>7} {oos_avg:>9.3f}")
        if len(is_sub) >= _MIN_N and is_avg > best_is_avg:
            best_is_avg = is_avg
            best_th = th

    print(f"\n=== IS'te en iyi eşik: {best_th} (ort ${best_is_avg:.3f}/işlem, SADECE IS verisiyle seçildi) ===")
    oos_best = oos_df[oos_df["conc"] <= best_th]
    print(f"  OOS performansı: {_stats(oos_best['pnl_usd'].to_numpy())}")

    print(f"\n=== Tam veri: filtresiz vs eşik={best_th} ===")
    print(f"  filtresiz     : {_stats(sub['pnl_usd'].to_numpy())}")
    filtered = sub[sub["conc"] <= best_th]
    print(f"  eşik<={best_th}: {_stats(filtered['pnl_usd'].to_numpy())}")

    if len(filtered) >= _MIN_N:
        days_span = (filtered["opened_at"].max() - filtered["opened_at"].min()).total_seconds() / 86400
        summarize(f"{direction} kalabalık eşik<={best_th}", filtered["pnl_usd"].to_numpy(), filtered["opened_at"], days_span)
    print()


def main() -> None:
    df = _load()
    _run_for_direction(df, "Short")
    _run_for_direction(df, "Long")


if __name__ == "__main__":
    main()
