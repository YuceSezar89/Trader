"""
Trajectory Lab — close_pos × VPMV İKİ YÖNLÜ etkileşim testi (bkz.
CONTEXT_LAB_STATUS.md, 8 Ağustos). Amaç "ikisi de güçlü mü" değil:

  1. VPMV mevcutken close_pos HANGİ VPMV seviyesinde ek bilgi taşıyor?
     (VPMV tertillerinde: ΔAUC = AUC(evol+vpmv+cvd_level+close_pos)
      - AUC(evol+vpmv+cvd_level))
  2. close_pos mevcutken VPMV HANGİ close_pos seviyesinde ek bilgi taşıyor?
     (close_pos tertillerinde: ΔAUC = AUC(evol+vpmv+cvd_level+close_pos)
      - AUC(evol+cvd_level+close_pos))

Tertiller örneklem-bazlı (qcut, dengeli) — strateji eşiği İCAT EDİLMEDİ,
sadece analiz için nötr bölümleme. Her hücrede n + ΔAUC + bootstrap %95 CI
zorunlu. Mümkünse 3 ailede AYRI da raporlanır (pooled ile birlikte).

Veri: stage2_anatomy.py::build_matrix() — yeni DB sorgusu YOK.

Kullanım:
    python -m research.trajectory_lab.interaction_vpmv_closepos
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from research.trajectory_lab.feature_acceptance import (
    N_BOOTSTRAP,
    RNG_SEED,
    bootstrap_delta_ci,
    fit_eval_lgbm,
)
from research.trajectory_lab.stage2_anatomy import BASELINE_PEAKS, build_matrix

MIN_CELL_N = 150


def _classify(delta: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "GÜÇLÜ koşullu katkı (CI tamamen pozitif)"
    if hi < 0:
        return "olası TERS ilişki (CI tamamen negatif) — 'kötü feature' denemiyoruz, sadece işaretliyoruz"
    if abs(delta) < 0.003:
        return "katkı yok / redundant olabilir (~0, CI sıfırı içeriyor)"
    if delta > 0:
        return "belirsiz (pozitif yönlü ama CI sıfırı içeriyor)"
    return "belirsiz (negatif yönlü ama CI sıfırı içeriyor)"


def _eval_cell(df: pd.DataFrame, baseline_cols: list, candidate_col: str, rng: np.random.Generator) -> dict | None:
    if len(df) < MIN_CELL_N or df["y"].nunique() < 2:
        return None
    df = df.sort_values("t0").reset_index(drop=True)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]
    if train["y"].nunique() < 2 or test["y"].nunique() < 2:
        return None
    full_cols = baseline_cols + [candidate_col]
    auc_base = fit_eval_lgbm(train, test, baseline_cols)[0]["auc"]
    auc_full = fit_eval_lgbm(train, test, full_cols)[0]["auc"]
    delta = auc_full - auc_base
    lo, hi = bootstrap_delta_ci(train, test, baseline_cols, candidate_col, N_BOOTSTRAP, rng)
    return {"n": len(df), "auc_base": auc_base, "auc_full": auc_full, "delta": delta, "lo": lo, "hi": hi}


def _print_direction(
    label: str, full: pd.DataFrame, bucket_col: str, baseline_cols: list, candidate_col: str, rng: np.random.Generator
) -> None:
    print(f"\n--- {label} ---")
    print(f"    baseline={'+'.join(baseline_cols)}  candidate={candidate_col}  bucket={bucket_col}")

    for scope_name, scope_df in [("POOLED (3 aile)", full)] + [
        (source, full[full["source"] == source]) for source in sorted(full["source"].unique())
    ]:
        print(f"\n  [{scope_name}]")
        for tertile in ("LOW", "MID", "HIGH"):
            sub = scope_df[scope_df[f"{bucket_col}_tertile"] == tertile]
            res = _eval_cell(sub, baseline_cols, candidate_col, rng)
            if res is None:
                print(f"    {tertile:5} n={len(sub):6} — yetersiz örneklem (min {MIN_CELL_N}), atlandı")
                continue
            verdict = _classify(res["delta"], res["lo"], res["hi"])
            print(
                f"    {tertile:5} n={res['n']:6} AUC {res['auc_base']:.4f}->{res['auc_full']:.4f} "
                f"ΔAUC={res['delta']:+.4f} CI=[{res['lo']:+.4f},{res['hi']:+.4f}]  [{verdict}]"
            )


def run() -> None:
    rng = np.random.default_rng(RNG_SEED)
    full = build_matrix()
    print(f"Toplam sinyal: {len(full)}, Kaynak dağılımı: {full['source'].value_counts().to_dict()}")

    full["vpmv_tertile"] = pd.qcut(full["vpmv"], 3, labels=["LOW", "MID", "HIGH"])
    full["close_pos_tertile"] = pd.qcut(full["close_pos"], 3, labels=["LOW", "MID", "HIGH"])
    print(f"VPMV tertile sınırları: {full.groupby('vpmv_tertile', observed=True)['vpmv'].agg(['min', 'max']).to_dict()}")
    print(
        f"close_pos tertile sınırları: "
        f"{full.groupby('close_pos_tertile', observed=True)['close_pos'].agg(['min', 'max']).to_dict()}"
    )

    baseline_cols = list(BASELINE_PEAKS.keys())  # evol, vpmv, cvd_level

    _print_direction(
        "YÖN 1 — VPMV mevcutken close_pos'un VPMV seviyesine göre ek katkısı",
        full, "vpmv", baseline_cols, "close_pos", rng,
    )

    baseline_minus_vpmv_plus_closepos = [c for c in baseline_cols if c != "vpmv"] + ["close_pos"]
    _print_direction(
        "YÖN 2 — close_pos mevcutken VPMV'nin close_pos seviyesine göre ek katkısı",
        full, "close_pos", baseline_minus_vpmv_plus_closepos, "vpmv", rng,
    )

    print("\n--- KARŞILAŞTIRMA: önceki POOLED (bucket'sız) Stage 2 sonucu ---")
    print("  close_pos (tüm veri): ΔAUC=+0.0079, CI=[+0.0058,+0.0103] (bkz. CONTEXT_LAB_STATUS.md)")


if __name__ == "__main__":
    run()
