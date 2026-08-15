"""
Trajectory Lab — Sinyal Mumu Anatomisi adaylarının (close_pos, buy_pct)
Stage 2 (Kazananın Doğrulanması) bataryası. `stage2_pctrank.py` ile AYNI
standart (baseline=evol+vpmv+cvd_level, walk-forward, 1000-permütasyon
placebo, bootstrap CI, aile-bazlı doğrulama) — feature_acceptance.py'nin
fonksiyonlarını YENİDEN KULLANIR.

Sıralı iki aşama (kullanıcı talimatı, 7 Ağustos):
  1. close_pos TEK BAŞINA, HAM (percentile-rank/normalize İCAT EDİLMEDİ) —
     baseline üzerine incremental katkısı var mı?
  2. close_pos geçerse: buy_pct, (baseline+close_pos) üzerine EK katkı
     sağlıyor mu — yoksa close_pos'un taşıdığı bilgiyi mi tekrarlıyor?

Kullanım:
    python -m research.trajectory_lab.stage2_anatomy --stage 1
    python -m research.trajectory_lab.stage2_anatomy --stage 2
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.trajectory_lab.feature_acceptance import (
    N_BOOTSTRAP,
    N_PLACEBO_PERMUTATIONS,
    N_WALK_FORWARD_FOLDS,
    RNG_SEED,
    bootstrap_delta_ci,
    cohens_d,
    fit_eval_lgbm,
    incremental_placebo_test,
    walk_forward_folds,
)

CORPUS_FILES = {
    "HA_Cross_Long": "research/trajectory_corpus/HA_Cross_Long.parquet",
    "RSI_Cross_Long": "research/trajectory_corpus/RSI_Cross_9,24_Long.parquet",
    "Supertrend_Long": "research/trajectory_corpus/Supertrend_10,3.0_Long.parquet",
}
ANATOMY_FILES = {
    "HA_Cross_Long": "research/trajectory_corpus/anatomy_HA_Cross_Long.parquet",
    "RSI_Cross_Long": "research/trajectory_corpus/anatomy_RSI_Cross_9_24_Long.parquet",
    "Supertrend_Long": "research/trajectory_corpus/anatomy_Supertrend_10_3.0_Long.parquet",
}
BASELINE_PEAKS = {"evol": 4, "vpmv": 0, "cvd_level": 5}


def build_matrix() -> pd.DataFrame:
    rows = []
    for name, path in CORPUS_FILES.items():
        df = pd.read_parquet(path)
        df = df[df["outcome"].isin(["winner", "loser"])]
        meta = df[["signal_id", "t0", "outcome"]].drop_duplicates().set_index("signal_id")
        feat = pd.DataFrame(index=meta.index)
        for metric, t_off in BASELINE_PEAKS.items():
            sub = df[(df["metric"] == metric) & (df["t_offset"] == t_off)].set_index("signal_id")["value"]
            feat[metric] = sub
        feat["t0"] = meta["t0"]
        feat["outcome"] = meta["outcome"]
        feat["source"] = name

        anatomy = pd.read_parquet(ANATOMY_FILES[name]).set_index("signal_id")
        feat["close_pos"] = anatomy["close_pos"]
        feat["buy_pct"] = anatomy["buy_pct"]
        rows.append(feat)
    full = pd.concat(rows)
    full = full.dropna(subset=list(BASELINE_PEAKS.keys()) + ["close_pos", "buy_pct"])
    full["y"] = (full["outcome"] == "winner").astype(int)
    return full.sort_values("t0").reset_index(drop=True)


def _run_battery(full: pd.DataFrame, baseline_cols: list, candidate_col: str, label: str) -> None:
    full_cols = baseline_cols + [candidate_col]
    rng = np.random.default_rng(RNG_SEED)

    print(f"===== Stage 2 — {label} =====")
    print(f"Toplam sinyal (winner+loser, havuzlanmış): {len(full)}")
    print(f"Sınıf dağılımı: {full['y'].value_counts().to_dict()}")
    print(f"Kaynak dağılımı: {full['source'].value_counts().to_dict()}")
    print(f"Tarih aralığı: {full['t0'].min()} -> {full['t0'].max()}\n")

    print("--- COHEN'S D (tek başına, TÜM havuz) ---")
    for metric in full_cols:
        print(f"  {metric}: d = {cohens_d(full, metric):.4f}")
    print()

    print("--- MUTUAL INFORMATION (outcome ile) ---")
    mi = mutual_info_classif(full[full_cols].to_numpy(), full["y"].to_numpy(), random_state=RNG_SEED)
    for metric, val in zip(full_cols, mi):
        print(f"  {metric}: MI = {val:.4f}")
    print()

    print("--- KORELASYON MATRİSİ ---")
    print(full[full_cols + ["y"]].corr().round(3))
    print()

    print(f"--- REDUNDANCY: {candidate_col}, baseline residual'iyla korele mi? ---")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(full[baseline_cols])
    lr = LogisticRegression().fit(x_scaled, full["y"])
    residual = full["y"] - lr.predict_proba(x_scaled)[:, 1]
    partial_corr = np.corrcoef(residual, full[candidate_col])[0, 1]
    print(f"  corr({candidate_col}, residual[y ~ {'+'.join(baseline_cols)}]) = {partial_corr:.4f}\n")

    fold_idx = walk_forward_folds(full, N_WALK_FORWARD_FOLDS)
    stages = {"baseline": baseline_cols, f"+{candidate_col}": full_cols}

    print(f"--- WALK-FORWARD VALIDATION ({N_WALK_FORWARD_FOLDS} dilim) ---")
    for stage_name, cols in stages.items():
        fold_metrics = []
        for i in range(1, N_WALK_FORWARD_FOLDS):
            train_idx = np.concatenate(fold_idx[:i])
            test_idx = fold_idx[i]
            m, _, _ = fit_eval_lgbm(full.iloc[train_idx], full.iloc[test_idx], cols)
            fold_metrics.append(m)
        dfm = pd.DataFrame(fold_metrics)
        print(f"  {stage_name}: AUC={dfm['auc'].mean():.4f}±{dfm['auc'].std():.4f}, "
              f"F1={dfm['f1'].mean():.4f}, Brier={dfm['brier'].mean():.4f}")
    print()

    final_train = full.iloc[np.concatenate(fold_idx[:-1])]
    final_test = full.iloc[fold_idx[-1]]

    print(f"--- {N_PLACEBO_PERMUTATIONS}-PERMÜTASYON PLACEBO TESTİ ---")
    placebo = incremental_placebo_test(
        final_train, final_test, baseline_cols, candidate_col, N_PLACEBO_PERMUTATIONS, rng
    )
    print(f"  AUC(baseline)={placebo['base_auc']:.4f} -> AUC(+{candidate_col})={placebo['obs_auc']:.4f}, "
          f"ΔAUC={placebo['obs_delta']:+.4f}")
    print(f"  Ampirik p-değeri = {placebo['p_value']:.4f}\n")

    print("--- BOOTSTRAP %95 CI (ΔAUC) ---")
    lo, hi = bootstrap_delta_ci(final_train, final_test, baseline_cols, candidate_col, N_BOOTSTRAP, rng)
    print(f"  ΔAUC %95 CI = [{lo:+.4f}, {hi:+.4f}]\n")

    print("--- HER AİLEDE AYRI DOĞRULAMA ---")
    for source in full["source"].unique():
        sub = full[full["source"] == source].sort_values("t0").reset_index(drop=True)
        split = int(len(sub) * 0.7)
        train_s, test_s = sub.iloc[:split], sub.iloc[split:]
        auc_base = fit_eval_lgbm(train_s, test_s, baseline_cols)[0]["auc"]
        auc_full = fit_eval_lgbm(train_s, test_s, full_cols)[0]["auc"]
        print(f"  {source} (n={len(sub)}): AUC(baseline)={auc_base:.4f} -> "
              f"AUC(+{candidate_col})={auc_full:.4f}, ΔAUC={auc_full - auc_base:+.4f}")
    print()


def run_stability(n_folds: int = 8) -> None:
    """close_pos'un Stage 2 sonucunun (ΔAUC=+0.0079, bkz. CONTEXT_LAB_STATUS.md)
    zaman içinde İSTİKRARLI mı olduğunu kontrol eder — body_size_pct'te
    kullanılan AYNI yöntem (stage2_pctrank.py::run_stability): 8 dilim,
    expanding window, her dilim kendi bootstrap %95 CI'siyle. Kabul
    kriteri AYNI (yeni eşik İCAT EDİLMEDİ) — yön + CI. Ayrıca (kullanıcı
    talimatı, 7 Ağustos) her dilimde MÜMKÜNSE aile-bazlı kırılım: aynı
    fit edilmiş modeller test dilimindeki her ailenin kendi alt-kümesinde
    ayrıca değerlendirilir (yeniden eğitim YOK, sadece predict)."""
    from sklearn.metrics import roc_auc_score

    from research.trajectory_lab.feature_acceptance import fit_eval_lgbm

    baseline_cols = list(BASELINE_PEAKS.keys())
    candidate_col = "close_pos"
    full_cols = baseline_cols + [candidate_col]
    rng = np.random.default_rng(RNG_SEED)

    full = build_matrix()
    print(f"===== Stage 2 Stability — close_pos (ham), {n_folds} dilim =====")
    print(f"Toplam sinyal: {len(full)}, Tarih aralığı: {full['t0'].min()} -> {full['t0'].max()}\n")

    fold_idx = walk_forward_folds(full, n_folds)

    print("--- DİLİM BAZINDA ΔAUC (expanding window: train=[0..i-1], test=dilim i) ---")
    deltas = []
    for i in range(1, n_folds):
        train_idx = np.concatenate(fold_idx[:i])
        test_idx = fold_idx[i]
        train_df, test_df = full.iloc[train_idx], full.iloc[test_idx]

        m_base, model_base, _ = fit_eval_lgbm(train_df, test_df, baseline_cols)
        m_full, model_full, _ = fit_eval_lgbm(train_df, test_df, full_cols)
        auc_base, auc_full = m_base["auc"], m_full["auc"]
        delta = auc_full - auc_base
        lo, hi = bootstrap_delta_ci(train_df, test_df, baseline_cols, candidate_col, N_BOOTSTRAP, rng)
        deltas.append(delta)
        sig = "pozitif" if lo > 0 else ("negatif" if hi < 0 else "belirsiz(0 CI içinde)")
        print(
            f"  Dilim {i} [{test_df['t0'].min()} -> {test_df['t0'].max()}] n={len(test_df)} "
            f"(train n={len(train_df)}): AUC {auc_base:.4f}->{auc_full:.4f} ΔAUC={delta:+.4f} "
            f"CI=[{lo:+.4f},{hi:+.4f}] [{sig}]"
        )

        for source in sorted(test_df["source"].unique()):
            mask = (test_df["source"] == source).to_numpy()
            sub = test_df[mask]
            if sub["y"].nunique() < 2 or len(sub) < 50:
                print(f"      {source}: n={len(sub)} (yetersiz, atlandı)")
                continue
            proba_base = model_base.predict_proba(sub[baseline_cols])[:, 1]
            proba_full = model_full.predict_proba(sub[full_cols])[:, 1]
            auc_b = roc_auc_score(sub["y"], proba_base)
            auc_f = roc_auc_score(sub["y"], proba_full)
            print(f"      {source}: n={len(sub)} AUC {auc_b:.4f}->{auc_f:.4f} ΔAUC={auc_f - auc_b:+.4f}")

    deltas = np.array(deltas)
    n_pos = int((deltas > 0).sum())
    print(
        f"\n  Özet: {n_pos}/{len(deltas)} dilim pozitif, ortalama ΔAUC={deltas.mean():+.4f}, "
        f"std={deltas.std():.4f}, min={deltas.min():+.4f}, max={deltas.max():+.4f}\n"
    )


def run_stage1() -> None:
    full = build_matrix()
    baseline_cols = list(BASELINE_PEAKS.keys())
    _run_battery(full, baseline_cols, "close_pos", "AŞAMA 1 — close_pos (ham) vs baseline (evol+vpmv+cvd_level)")


def run_stage2() -> None:
    full = build_matrix()
    baseline_plus_close = list(BASELINE_PEAKS.keys()) + ["close_pos"]
    _run_battery(
        full, baseline_plus_close, "buy_pct",
        "AŞAMA 2 — buy_pct'in EK katkısı, (baseline+close_pos) üzerine",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--folds", type=int, default=8)
    args = parser.parse_args()
    if args.stability:
        run_stability(args.folds)
    elif args.stage == 1:
        run_stage1()
    elif args.stage == 2:
        run_stage2()
    else:
        parser.error("--stage 1/2 ya da --stability belirtilmeli")
