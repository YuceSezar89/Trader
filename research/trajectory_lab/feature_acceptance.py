"""
Trajectory Lab — Feature Acceptance Pipeline (bkz. README.md).

Bir aday metriğin (CANDIDATE), mevcut kabul edilmiş özellik setine
(BASELINE_FEATURES) GERÇEKTEN yeni bilgi katıp katmadığını ölçer.
CVD Level'in 3 Ağustos'ta bu bataryadan geçirilip active=True yapılmasıyla
doğrulandı (bkz. README "Neden CVD Slope reddedildi" ve "Feature
Acceptance Pipeline" bölümleri) — bu dosya o çalışmanın genelleştirilmiş,
tekrar kullanılabilir hâli.

Kullanım: aşağıdaki "AYARLAR" bölümünü yeni aday için güncelleyip
`python -m research.trajectory_lab.feature_acceptance` çalıştır.

Karar kuralı: 1000-permütasyon placebo testi p<0.05 VE bootstrap %95 CI
sıfırı dışlıyor VE katkı yönü TÜM corpus'larda (indikatörlerde) tutarlıysa
→ config.py'de candidate active=True yapılabilir. Aksi halde provider
olarak kalır, corpus'a dahil edilmez.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ============================================================
# AYARLAR — yeni bir aday test etmek için burayı güncelle
# ============================================================
RNG_SEED = 42

CORPUS_FILES = {
    "HA_Cross_Long": "research/trajectory_corpus/HA_Cross_Long.parquet",
    "RSI_Cross_Long": "research/trajectory_corpus/RSI_Cross_9,24_Long.parquet",
    "Supertrend_Long": "research/trajectory_corpus/Supertrend_10,3.0_Long.parquet",
}

# metric -> divergence-peak t_offset (viz.compute_divergence ile önceden bulunmuş olmalı)
FEATURE_PEAK_OFFSETS = {"evol": 4, "vpmv": 0, "cvd_level": 5}

BASELINE_FEATURES = ["evol", "vpmv"]  # zaten kabul edilmiş / active=True olan set
CANDIDATE_FEATURE = "cvd_level"  # test edilen aday

N_WALK_FORWARD_FOLDS = 5  # expanding window, ilk fold hariç (N-1) adım test edilir
N_PLACEBO_PERMUTATIONS = 1000
N_BOOTSTRAP = 300

LGB_PARAMS = dict(
    n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
    random_state=RNG_SEED, verbose=-1,
)

# ============================================================
# Yardımcı fonksiyonlar
# ============================================================


def build_feature_matrix(corpus_files: dict, peak_offsets: dict) -> pd.DataFrame:
    """Her corpus'tan, her metriğin kendi divergence-peak bar'ındaki değerini
    tek skaler özellik olarak çıkarır. Winner/loser dışındaki sinyaller
    (neutral) atılır — Trajectory Lab'ın divergence çerçevesiyle tutarlı."""
    rows = []
    for name, path in corpus_files.items():
        df = pd.read_parquet(path)
        df = df[df["outcome"].isin(["winner", "loser"])]
        meta = df[["signal_id", "t0", "outcome"]].drop_duplicates().set_index("signal_id")
        feat = pd.DataFrame(index=meta.index)
        for metric, t_off in peak_offsets.items():
            sub = df[(df["metric"] == metric) & (df["t_offset"] == t_off)].set_index("signal_id")["value"]
            feat[metric] = sub
        feat["t0"] = meta["t0"]
        feat["outcome"] = meta["outcome"]
        feat["source"] = name
        rows.append(feat)
    full = pd.concat(rows).dropna()
    full["y"] = (full["outcome"] == "winner").astype(int)
    return full.sort_values("t0").reset_index(drop=True)


def cohens_d(df: pd.DataFrame, metric: str) -> float:
    w = df.loc[df["y"] == 1, metric]
    l = df.loc[df["y"] == 0, metric]
    pooled_std = np.sqrt(((len(w) - 1) * w.var(ddof=1) + (len(l) - 1) * l.var(ddof=1)) / (len(w) + len(l) - 2))
    return (w.mean() - l.mean()) / pooled_std


def fit_eval_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list) -> tuple:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(train_df[cols], train_df["y"])
    proba = model.predict_proba(test_df[cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(test_df["y"], proba),
        "precision": precision_score(test_df["y"], pred, zero_division=0),
        "recall": recall_score(test_df["y"], pred, zero_division=0),
        "f1": f1_score(test_df["y"], pred, zero_division=0),
        "brier": brier_score_loss(test_df["y"], proba),
    }
    return metrics, model, proba


def walk_forward_folds(full: pd.DataFrame, n_folds: int) -> list:
    n = len(full)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_folds)]


def incremental_placebo_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, base_cols: list, new_col: str,
    n_perm: int, rng: np.random.Generator,
) -> dict:
    base_auc = fit_eval_lgbm(train_df, test_df, base_cols)[0]["auc"]
    full_cols = base_cols + [new_col]
    obs_auc = fit_eval_lgbm(train_df, test_df, full_cols)[0]["auc"]
    obs_delta = obs_auc - base_auc

    null_deltas = np.empty(n_perm)
    train_shuf = train_df.copy()
    for i in range(n_perm):
        train_shuf[new_col] = rng.permutation(train_shuf[new_col].to_numpy())
        m_shuf = fit_eval_lgbm(train_shuf, test_df, full_cols)[0]
        null_deltas[i] = m_shuf["auc"] - base_auc
    p_value = (null_deltas >= obs_delta).mean()
    return {
        "base_auc": base_auc, "obs_auc": obs_auc, "obs_delta": obs_delta,
        "null_mean": null_deltas.mean(), "null_std": null_deltas.std(), "p_value": p_value,
    }


def bootstrap_delta_ci(
    train_df: pd.DataFrame, test_df: pd.DataFrame, base_cols: list, new_col: str,
    n_boot: int, rng: np.random.Generator,
) -> tuple:
    full_cols = base_cols + [new_col]
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


# ============================================================
# Ana rapor
# ============================================================


def run(
    corpus_files: dict | None = None,
    peak_offsets: dict | None = None,
    baseline_features: list | None = None,
    candidate_feature: str = CANDIDATE_FEATURE,
) -> None:
    corpus_files = CORPUS_FILES if corpus_files is None else corpus_files
    peak_offsets = FEATURE_PEAK_OFFSETS if peak_offsets is None else peak_offsets
    baseline_features = BASELINE_FEATURES if baseline_features is None else baseline_features

    rng = np.random.default_rng(RNG_SEED)
    full_cols = baseline_features + [candidate_feature]

    full = build_feature_matrix(corpus_files, peak_offsets)
    print(f"Toplam sinyal (winner+loser, havuzlanmış): {len(full)}")
    print(f"Sınıf dağılımı: {full['y'].value_counts().to_dict()}")
    print(f"Tarih aralığı: {full['t0'].min()} -> {full['t0'].max()}\n")

    print("=" * 70)
    print("COHEN'S D (tek başına, TÜM havuz)")
    print("=" * 70)
    for metric in full_cols:
        print(f"  {metric}: d = {cohens_d(full, metric):.3f}")
    print()

    print("=" * 70)
    print("MUTUAL INFORMATION (outcome ile, TÜM havuz)")
    print("=" * 70)
    mi = mutual_info_classif(full[full_cols].to_numpy(), full["y"].to_numpy(), random_state=RNG_SEED)
    for metric, val in zip(full_cols, mi):
        print(f"  {metric}: MI = {val:.4f}")
    print()

    print("=" * 70)
    print("KORELASYON MATRİSİ (Pearson, TÜM havuz)")
    print("=" * 70)
    print(full[full_cols + ["y"]].corr().round(3))
    print()

    print("=" * 70)
    print(f"REDUNDANCY: {candidate_feature}, {'+'.join(baseline_features)} residual'iyla korele mi?")
    print("=" * 70)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(full[baseline_features])
    lr = LogisticRegression().fit(x_scaled, full["y"])
    residual = full["y"] - lr.predict_proba(x_scaled)[:, 1]
    partial_corr = np.corrcoef(residual, full[candidate_feature])[0, 1]
    print(f"  corr({candidate_feature}, residual[y ~ {'+'.join(baseline_features)}]) = {partial_corr:.4f}\n")

    fold_idx = walk_forward_folds(full, N_WALK_FORWARD_FOLDS)
    stages = {"+".join(baseline_features[:i + 1]): baseline_features[:i + 1] for i in range(len(baseline_features))}
    stages["+".join(full_cols)] = full_cols

    print("=" * 70)
    print(f"WALK-FORWARD VALIDATION ({N_WALK_FORWARD_FOLDS} dilim, expanding window)")
    print("=" * 70)
    for stage_name, cols in stages.items():
        fold_metrics = []
        for i in range(1, N_WALK_FORWARD_FOLDS):
            train_idx = np.concatenate(fold_idx[:i])
            test_idx = fold_idx[i]
            m, _, _ = fit_eval_lgbm(full.iloc[train_idx], full.iloc[test_idx], cols)
            fold_metrics.append(m)
        dfm = pd.DataFrame(fold_metrics)
        print(f"\n--- {stage_name} ---")
        print(dfm.round(4))
        print(f"ORTALAMA: AUC={dfm['auc'].mean():.4f}±{dfm['auc'].std():.4f}, "
              f"Precision={dfm['precision'].mean():.4f}, Recall={dfm['recall'].mean():.4f}, "
              f"F1={dfm['f1'].mean():.4f}, Brier={dfm['brier'].mean():.4f}")
    print()

    final_train = full.iloc[np.concatenate(fold_idx[:-1])]
    final_test = full.iloc[fold_idx[-1]]

    print("=" * 70)
    print("PERMUTATION IMPORTANCE + LGBM GAIN IMPORTANCE (son dilim, tam model)")
    print("=" * 70)
    for stage_name, cols in stages.items():
        m, model, _ = fit_eval_lgbm(final_train, final_test, cols)
        perm = permutation_importance(
            model, final_test[cols], final_test["y"], n_repeats=30, random_state=RNG_SEED, scoring="roc_auc",
        )
        gain = model.booster_.feature_importance(importance_type="gain")
        print(f"\n--- {stage_name} (test AUC={m['auc']:.4f}) ---")
        for j, c in enumerate(cols):
            print(f"  {c}: perm.imp(ΔAUC)={perm.importances_mean[j]:+.4f}±{perm.importances_std[j]:.4f}, "
                  f"gain=%{100 * gain[j] / gain.sum():.1f}")
    print()

    print("=" * 70)
    print(f"{N_PLACEBO_PERMUTATIONS}-PERMÜTASYON PLACEBO TESTİ ({candidate_feature} artan katkısı)")
    print("=" * 70)
    placebo = incremental_placebo_test(
        final_train, final_test, baseline_features, candidate_feature, N_PLACEBO_PERMUTATIONS, rng,
    )
    print(f"  AUC({'+'.join(baseline_features)})={placebo['base_auc']:.4f} -> "
          f"AUC(+{candidate_feature})={placebo['obs_auc']:.4f}, ΔAUC={placebo['obs_delta']:+.4f}")
    print(f"  Placebo null ΔAUC: ortalama={placebo['null_mean']:+.4f}, std={placebo['null_std']:.4f}")
    print(f"  Ampirik p-değeri = {placebo['p_value']:.4f}\n")

    print("=" * 70)
    print("BOOTSTRAP %95 CI (ΔAUC, son test dilimi resample)")
    print("=" * 70)
    lo, hi = bootstrap_delta_ci(final_train, final_test, baseline_features, candidate_feature, N_BOOTSTRAP, rng)
    print(f"  ΔAUC({candidate_feature} katkısı) %95 CI = [{lo:+.4f}, {hi:+.4f}]\n")

    print("=" * 70)
    print(f"HER CORPUS'TA AYRI DOĞRULAMA ({candidate_feature} artan katkısı)")
    print("=" * 70)
    for name, path in corpus_files.items():
        sub = build_feature_matrix({name: path}, peak_offsets)
        split = int(len(sub) * 0.7)
        train_s, test_s = sub.iloc[:split], sub.iloc[split:]
        auc_base = fit_eval_lgbm(train_s, test_s, baseline_features)[0]["auc"]
        auc_full = fit_eval_lgbm(train_s, test_s, full_cols)[0]["auc"]
        print(f"  {name} (n={len(sub)}): AUC({'+'.join(baseline_features)})={auc_base:.4f} -> "
              f"AUC(+{candidate_feature})={auc_full:.4f}, ΔAUC={auc_full - auc_base:+.4f}")


if __name__ == "__main__":
    run()
