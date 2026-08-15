"""
Trajectory Lab — close_pos × VPMV etkileşim ŞEKLİNİN zaman içinde
İSTİKRARLI olup olmadığını test eder (bkz. CONTEXT_LAB_STATUS.md, 8
Ağustos). Önceki testlerde "feature'ın etkisi zaman içinde stabil mi"
sorusuna bakmıştık (body_size_pct/close_pos) — burada soru farklı:
"feature'ların birbirleriyle olan İLİŞKİSİ (etkileşim ŞEKLİ) zaman
içinde stabil mi?"

Expanding-window (8 dilim, stage2_pctrank/anatomy'nin --stability'siyle
AYNI iskelet). Her dilimde:
  - VPMV ve close_pos tertilleri O DİLİMİN TEST verisinden TAZE
    hesaplanır (qcut, nötr — bucket sınırı İCAT/optimize EDİLMEDİ,
    gelecek dilime sızıntı yok).
  - YÖN 1 (VPMV→close_pos katkısı, beklenen: TERS-U) ve YÖN 2
    (close_pos→VPMV katkısı, beklenen: MONOTON ARTAN) her dilimde
    yeniden ölçülür.
  - Model dilim+yön başına BİR KEZ eğitilir (expanding train), bucket
    alt-kümelerinde SADECE predict_proba + roc_auc_score (yeniden eğitim
    YOK) — performans optimizasyonu, istatistiksel anlam DEĞİŞMEDİ
    (LGBM fixed seed ile aynı train'de deterministik, tekrar fit
    gereksizdi).
  - Bootstrap CI: test-bucket alt-kümesi yeniden örneklenir, ÖNCEDEN
    hesaplanmış olasılıklar üzerinden AUC yeniden hesaplanır (model
    tekrar eğitilmiyor).
  - Pooled + 3 aile ayrı ayrı.

Amaç: bucket sınırı/threshold/trade kuralı üretmek DEĞİL — sadece
etkileşim ŞEKLİNİN (ters-U / monoton artan) zaman içinde tekrarlanıp
tekrarlanmadığını görmek.

Kullanım:
    python -m research.trajectory_lab.interaction_stability
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.trajectory_lab.feature_acceptance import N_BOOTSTRAP, RNG_SEED, fit_eval_lgbm, walk_forward_folds
from research.trajectory_lab.stage2_anatomy import BASELINE_PEAKS, build_matrix

N_FOLDS = 8
MIN_CELL_N = 100


def _classify(delta: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "GÜÇLÜ+"
    if hi < 0:
        return "TERS-"
    if abs(delta) < 0.003:
        return "~0"
    return "belirsiz+" if delta > 0 else "belirsiz-"


def _fast_bootstrap_ci(y: np.ndarray, proba_base: np.ndarray, proba_full: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            deltas[i] = np.nan
            continue
        auc_base = roc_auc_score(yb, proba_base[idx])
        auc_full = roc_auc_score(yb, proba_full[idx])
        deltas[i] = auc_full - auc_base
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return np.nan, np.nan
    return tuple(np.percentile(deltas, [2.5, 97.5]))


def _eval_scope_bucket(
    scope_df: pd.DataFrame, bucket_col: str, tertile: str,
    baseline_cols: list, candidate_col: str,
    model_base, model_full, rng: np.random.Generator,
) -> dict | None:
    sub = scope_df[scope_df[f"{bucket_col}_tertile"] == tertile]
    if len(sub) < MIN_CELL_N or sub["y"].nunique() < 2:
        return None
    y = sub["y"].to_numpy()
    proba_base = model_base.predict_proba(sub[baseline_cols])[:, 1]
    proba_full = model_full.predict_proba(sub[baseline_cols + [candidate_col]])[:, 1]
    delta = roc_auc_score(y, proba_full) - roc_auc_score(y, proba_base)
    lo, hi = _fast_bootstrap_ci(y, proba_base, proba_full, N_BOOTSTRAP, rng)
    return {"n": len(sub), "delta": delta, "lo": lo, "hi": hi, "verdict": _classify(delta, lo, hi)}


def _fmt_cell(res: dict | None) -> str:
    if res is None:
        return "  (yetersiz)  "
    return f"{res['delta']:+.4f}[{res['verdict']:9}]n={res['n']}"


def run() -> None:
    rng = np.random.default_rng(RNG_SEED)
    full = build_matrix()
    baseline_cols = list(BASELINE_PEAKS.keys())  # evol, vpmv, cvd_level
    baseline_minus_vpmv_plus_cp = [c for c in baseline_cols if c != "vpmv"] + ["close_pos"]

    print(f"Toplam sinyal: {len(full)}, Kaynak dağılımı: {full['source'].value_counts().to_dict()}")
    fold_idx = walk_forward_folds(full, N_FOLDS)

    scopes_order = ["POOLED"] + sorted(full["source"].unique())
    y1_shapes = {s: [] for s in scopes_order}  # YÖN1: her dilim için (LOW,MID,HIGH) delta
    y2_shapes = {s: [] for s in scopes_order}

    for i in range(1, N_FOLDS):
        train_idx = np.concatenate(fold_idx[:i])
        test_idx = fold_idx[i]
        train, test = full.iloc[train_idx].copy(), full.iloc[test_idx].copy()

        # tertiller BU DİLİMİN test verisinden taze hesaplanır (nötr, sızıntı yok)
        test["vpmv_tertile"] = pd.qcut(test["vpmv"], 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")
        test["close_pos_tertile"] = pd.qcut(test["close_pos"], 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")

        print(
            f"\n===== Dilim {i} [{test['t0'].min()} -> {test['t0'].max()}] "
            f"train n={len(train)} test n={len(test)} ====="
        )

        # YÖN 1: baseline=evol+vpmv+cvd_level, candidate=close_pos, bucket=vpmv
        _, model_base_1, _ = fit_eval_lgbm(train, test, baseline_cols)
        _, model_full_1, _ = fit_eval_lgbm(train, test, baseline_cols + ["close_pos"])

        # YÖN 2: baseline=evol+cvd_level+close_pos, candidate=vpmv, bucket=close_pos
        _, model_base_2, _ = fit_eval_lgbm(train, test, baseline_minus_vpmv_plus_cp)
        _, model_full_2, _ = fit_eval_lgbm(train, test, baseline_minus_vpmv_plus_cp + ["vpmv"])

        for scope in scopes_order:
            scope_df = test if scope == "POOLED" else test[test["source"] == scope]

            row1 = []
            for tertile in ("LOW", "MID", "HIGH"):
                res = _eval_scope_bucket(scope_df, "vpmv", tertile, baseline_cols, "close_pos", model_base_1, model_full_1, rng)
                row1.append(res)
            y1_shapes[scope].append(row1)

            row2 = []
            for tertile in ("LOW", "MID", "HIGH"):
                res = _eval_scope_bucket(scope_df, "close_pos", tertile, baseline_minus_vpmv_plus_cp, "vpmv", model_base_2, model_full_2, rng)
                row2.append(res)
            y2_shapes[scope].append(row2)

            print(f"  [{scope}]")
            print(f"    YÖN1 (VPMV→close_pos, beklenen TERS-U): "
                  f"LOW={_fmt_cell(row1[0])}  MID={_fmt_cell(row1[1])}  HIGH={_fmt_cell(row1[2])}")
            print(f"    YÖN2 (close_pos→VPMV, beklenen MONOTON): "
                  f"LOW={_fmt_cell(row2[0])}  MID={_fmt_cell(row2[1])}  HIGH={_fmt_cell(row2[2])}")

    print("\n\n===== ŞEKİL İSTİKRARI ÖZETİ (bucket/threshold üretilmedi — sadece şekil tekrar ediyor mu) =====")
    for scope in scopes_order:
        print(f"\n[{scope}]")
        rows1 = y1_shapes[scope]
        n_shape1 = 0
        n_valid1 = 0
        for row in rows1:
            if any(r is None for r in row):
                continue
            n_valid1 += 1
            lo, mid, hi = (r["delta"] for r in row)
            if mid > lo and mid > hi:
                n_shape1 += 1
        print(f"  YÖN1 TERS-U tekrarı: {n_shape1}/{n_valid1} dilim (geçerli veri olan dilimler arasında)")

        rows2 = y2_shapes[scope]
        n_shape2 = 0
        n_valid2 = 0
        for row in rows2:
            if any(r is None for r in row):
                continue
            n_valid2 += 1
            lo, mid, hi = (r["delta"] for r in row)
            if lo <= mid <= hi and hi > lo:
                n_shape2 += 1
        print(f"  YÖN2 MONOTON-ARTAN tekrarı: {n_shape2}/{n_valid2} dilim (geçerli veri olan dilimler arasında)")


if __name__ == "__main__":
    run()
