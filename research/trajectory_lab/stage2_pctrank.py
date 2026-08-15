"""
Trajectory Lab — "Sıra Dışılık" (sembol-içi percentile-rank) adaylarının
Stage 2 (Kazananın Doğrulanması) bataryası.

feature_acceptance.py'nin fonksiyonlarını (cohens_d, fit_eval_lgbm,
walk_forward_folds, incremental_placebo_test, bootstrap_delta_ci) YENİDEN
KULLANIR — sadece feature-matrix inşası farklı: baseline (evol/vpmv/
cvd_level) ham değerleriyle, aday ham metriğin (body_size/range_size/
volume_level/roc) (source, symbol) içindeki percentile-rank'iyle eklenir
(min MIN_SIGNALS_PER_SYMBOL kendi sinyali olan sembol+aile grupları).

Kullanım:
    python -m research.trajectory_lab.stage2_pctrank --candidate body_size
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
BASELINE_PEAKS = {"evol": 4, "vpmv": 0, "cvd_level": 5}
MIN_SIGNALS_PER_SYMBOL = 30


def build_matrix(
    corpus_files: dict, candidate_metrics: str | list[str], candidate_t_offset: int = 0
) -> pd.DataFrame:
    if isinstance(candidate_metrics, str):
        candidate_metrics = [candidate_metrics]
    raw_cols = [f"{cm}_raw" for cm in candidate_metrics]

    rows = []
    for name, path in corpus_files.items():
        df = pd.read_parquet(path)
        df = df[df["outcome"].isin(["winner", "loser"])]
        meta = df[["signal_id", "symbol", "t0", "outcome"]].drop_duplicates().set_index("signal_id")
        feat = pd.DataFrame(index=meta.index)
        for metric, t_off in BASELINE_PEAKS.items():
            sub = df[(df["metric"] == metric) & (df["t_offset"] == t_off)].set_index("signal_id")["value"]
            feat[metric] = sub
        for cm in candidate_metrics:
            cand = df[(df["metric"] == cm) & (df["t_offset"] == candidate_t_offset)].set_index("signal_id")["value"]
            feat[f"{cm}_raw"] = cand
        feat["symbol"] = meta["symbol"]
        feat["t0"] = meta["t0"]
        feat["outcome"] = meta["outcome"]
        feat["source"] = name
        rows.append(feat)
    full = pd.concat(rows)
    full = full.dropna(subset=list(BASELINE_PEAKS.keys()) + raw_cols)

    full["_key"] = list(zip(full["source"], full["symbol"]))
    sizes = full.groupby("_key").size()
    valid_keys = set(sizes[sizes >= MIN_SIGNALS_PER_SYMBOL].index)
    full = full[full["_key"].isin(valid_keys)].copy()
    for cm in candidate_metrics:
        full[f"{cm}_pct"] = full.groupby("_key")[f"{cm}_raw"].rank(pct=True) * 100.0
    full["y"] = (full["outcome"] == "winner").astype(int)
    return full.sort_values("t0").reset_index(drop=True)


def incremental_placebo_test_multi(
    train_df: pd.DataFrame, test_df: pd.DataFrame, base_cols: list, new_cols: list,
    n_perm: int, rng: np.random.Generator,
) -> dict:
    """incremental_placebo_test'in çok-kolonlu hali — new_cols BİRLİKTE (aynı
    permütasyon indeksiyle) karıştırılır, böylece aralarındaki korelasyon
    korunur ama ikisinin BİRLİKTE y ile ilişkisi kırılır (çift'in ortak
    artan katkısını test eder)."""
    base_auc = fit_eval_lgbm(train_df, test_df, base_cols)[0]["auc"]
    full_cols = base_cols + new_cols
    obs_auc = fit_eval_lgbm(train_df, test_df, full_cols)[0]["auc"]
    obs_delta = obs_auc - base_auc

    null_deltas = np.empty(n_perm)
    train_shuf = train_df.copy()
    for i in range(n_perm):
        perm_idx = rng.permutation(len(train_shuf))
        train_shuf[new_cols] = train_shuf[new_cols].to_numpy()[perm_idx]
        m_shuf = fit_eval_lgbm(train_shuf, test_df, full_cols)[0]
        null_deltas[i] = m_shuf["auc"] - base_auc
    p_value = (null_deltas >= obs_delta).mean()
    return {
        "base_auc": base_auc, "obs_auc": obs_auc, "obs_delta": obs_delta,
        "null_mean": null_deltas.mean(), "null_std": null_deltas.std(), "p_value": p_value,
    }


def bootstrap_delta_ci_multi(
    train_df: pd.DataFrame, test_df: pd.DataFrame, base_cols: list, new_cols: list,
    n_boot: int, rng: np.random.Generator,
) -> tuple:
    full_cols = base_cols + new_cols
    n_test = len(test_df)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        test_b = test_df.iloc[idx]
        if test_b["y"].nunique() < 2:
            deltas[i] = np.nan
            continue
        auc_base = fit_eval_lgbm(train_df, test_b, base_cols)[0]["auc"]
        auc_full = fit_eval_lgbm(train_df, test_b, full_cols)[0]["auc"]
        deltas[i] = auc_full - auc_base
    deltas = deltas[~np.isnan(deltas)]
    return tuple(np.percentile(deltas, [2.5, 97.5]))


def run(candidate_metric: str) -> None:
    baseline_cols = list(BASELINE_PEAKS.keys())
    candidate_col = f"{candidate_metric}_pct"
    full_cols = baseline_cols + [candidate_col]
    rng = np.random.default_rng(RNG_SEED)

    full = build_matrix(CORPUS_FILES, candidate_metric)
    print(f"===== Stage 2 — {candidate_metric} (percentile-rank) =====")
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
    stages = {"baseline (evol+vpmv+cvd_level)": baseline_cols, f"+{candidate_col}": full_cols}

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


def run_stability(candidate_metric: str, n_folds: int = 8) -> None:
    """body_size_pct'in tek başına Stage 2 sonucunun (ΔAUC=+0.0086, bkz.
    CONTEXT_LAB_STATUS.md) zaman içinde İSTİKRARLI mı yoksa tek bir
    dönemin sürüklediği bir ortalama mı olduğunu kontrol eder. 5 yerine
    daha ince (n_folds) dilimlere bölünür, her dilim kendi tarih aralığı
    ve kendi bootstrap CI'siyle raporlanır."""
    baseline_cols = list(BASELINE_PEAKS.keys())
    candidate_col = f"{candidate_metric}_pct"
    full_cols = baseline_cols + [candidate_col]
    rng = np.random.default_rng(RNG_SEED)

    full = build_matrix(CORPUS_FILES, candidate_metric)
    print(f"===== Stage 2 Stability — {candidate_metric} (percentile-rank), {n_folds} dilim =====")
    print(f"Toplam sinyal: {len(full)}, Tarih aralığı: {full['t0'].min()} -> {full['t0'].max()}\n")

    fold_idx = walk_forward_folds(full, n_folds)

    print("--- DİLİM BAZINDA ΔAUC (expanding window: train=[0..i-1], test=dilim i) ---")
    deltas = []
    for i in range(1, n_folds):
        train_idx = np.concatenate(fold_idx[:i])
        test_idx = fold_idx[i]
        train_df, test_df = full.iloc[train_idx], full.iloc[test_idx]
        auc_base = fit_eval_lgbm(train_df, test_df, baseline_cols)[0]["auc"]
        auc_full = fit_eval_lgbm(train_df, test_df, full_cols)[0]["auc"]
        delta = auc_full - auc_base
        lo, hi = bootstrap_delta_ci(train_df, test_df, baseline_cols, candidate_col, N_BOOTSTRAP, rng)
        deltas.append(delta)
        sig = "pozitif" if lo > 0 else ("negatif" if hi < 0 else "belirsiz(0 CI içinde)")
        print(
            f"  Dilim {i} [{test_df['t0'].min()} -> {test_df['t0'].max()}] n={len(test_df)} "
            f"(train n={len(train_df)}): AUC {auc_base:.4f}->{auc_full:.4f} ΔAUC={delta:+.4f} "
            f"CI=[{lo:+.4f},{hi:+.4f}] [{sig}]"
        )

    deltas = np.array(deltas)
    n_pos = int((deltas > 0).sum())
    print(
        f"\n  Özet: {n_pos}/{len(deltas)} dilim pozitif, ortalama ΔAUC={deltas.mean():+.4f}, "
        f"std={deltas.std():.4f}, min={deltas.min():+.4f}, max={deltas.max():+.4f}\n"
    )


def run_combo(candidate_metrics: list) -> None:
    baseline_cols = list(BASELINE_PEAKS.keys())
    candidate_cols = [f"{m}_pct" for m in candidate_metrics]
    full_cols = baseline_cols + candidate_cols
    rng = np.random.default_rng(RNG_SEED)

    full = build_matrix(CORPUS_FILES, candidate_metrics)
    print(f"===== Stage 2 — {' + '.join(candidate_metrics)} (BİRLİKTE, percentile-rank) =====")
    print(f"Toplam sinyal (winner+loser, havuzlanmış): {len(full)}")
    print(f"Sınıf dağılımı: {full['y'].value_counts().to_dict()}")
    print(f"Kaynak dağılımı: {full['source'].value_counts().to_dict()}")
    print(f"Tarih aralığı: {full['t0'].min()} -> {full['t0'].max()}\n")

    print("--- KORELASYON MATRİSİ ---")
    print(full[full_cols + ["y"]].corr().round(3))
    print()

    print(f"--- REDUNDANCY: {candidate_cols[0]} <-> {candidate_cols[1]} (birbirleriyle) ---")
    raw_corr = full[candidate_cols[0]].corr(full[candidate_cols[1]])
    print(f"  ham korelasyon = {raw_corr:.4f}")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(full[baseline_cols])
    lr = LogisticRegression().fit(x_scaled, full["y"])
    residual = full["y"] - lr.predict_proba(x_scaled)[:, 1]
    for c in candidate_cols:
        print(f"  corr({c}, residual[y ~ {'+'.join(baseline_cols)}]) = {np.corrcoef(residual, full[c])[0, 1]:.4f}")
    print()

    fold_idx = walk_forward_folds(full, N_WALK_FORWARD_FOLDS)
    stages = {
        "baseline (evol+vpmv+cvd_level)": baseline_cols,
        f"+{candidate_cols[0]}": baseline_cols + [candidate_cols[0]],
        f"+{candidate_cols[1]}": baseline_cols + [candidate_cols[1]],
        f"+{candidate_cols[0]}+{candidate_cols[1]}": full_cols,
    }

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

    print(f"--- {N_PLACEBO_PERMUTATIONS}-PERMÜTASYON PLACEBO TESTİ (ikisi BİRLİKTE) ---")
    placebo = incremental_placebo_test_multi(
        final_train, final_test, baseline_cols, candidate_cols, N_PLACEBO_PERMUTATIONS, rng
    )
    print(f"  AUC(baseline)={placebo['base_auc']:.4f} -> AUC(+ikisi)={placebo['obs_auc']:.4f}, "
          f"ΔAUC={placebo['obs_delta']:+.4f}")
    print(f"  Ampirik p-değeri = {placebo['p_value']:.4f}\n")

    print("--- BOOTSTRAP %95 CI (ΔAUC, ikisi BİRLİKTE) ---")
    lo, hi = bootstrap_delta_ci_multi(final_train, final_test, baseline_cols, candidate_cols, N_BOOTSTRAP, rng)
    print(f"  ΔAUC %95 CI = [{lo:+.4f}, {hi:+.4f}]\n")

    print("--- HER AİLEDE AYRI DOĞRULAMA (ikisi BİRLİKTE) ---")
    for source in full["source"].unique():
        sub = full[full["source"] == source].sort_values("t0").reset_index(drop=True)
        split = int(len(sub) * 0.7)
        train_s, test_s = sub.iloc[:split], sub.iloc[split:]
        auc_base = fit_eval_lgbm(train_s, test_s, baseline_cols)[0]["auc"]
        auc_full = fit_eval_lgbm(train_s, test_s, full_cols)[0]["auc"]
        print(f"  {source} (n={len(sub)}): AUC(baseline)={auc_base:.4f} -> "
              f"AUC(+ikisi)={auc_full:.4f}, ΔAUC={auc_full - auc_base:+.4f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", required=True, nargs="+", choices=["body_size", "range_size", "volume_level", "roc"]
    )
    parser.add_argument("--stability", action="store_true", help="Dilim-bazında zaman içi kararlılık testi")
    parser.add_argument("--folds", type=int, default=8, help="--stability için dilim sayısı")
    args = parser.parse_args()
    if args.stability:
        if len(args.candidate) != 1:
            parser.error("--stability tek adayla kullanılmalı")
        run_stability(args.candidate[0], args.folds)
    elif len(args.candidate) == 1:
        run(args.candidate[0])
    else:
        run_combo(args.candidate)
