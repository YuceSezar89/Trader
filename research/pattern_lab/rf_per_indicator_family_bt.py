"""
Random Forest — indikatör ailesi başına AYRI model (20 Tem 2026, kullanıcı
isteği: "bunu diğer indikatör ailelerinde de dene").

rf_signal_quality_bt.py TÜM aileleri (RSI_Cross, HA_Cross, Supertrend,
MA200_Cross) tek modelde karıştırıp "indicators" alanını bir one-hot feature
olarak vermişti. Sorun: "hangi aileden geldiği" bilgisi TEK BAŞINA çok güçlü
bir ipucu olabilir (ör. "zaten MA200_Cross zayıftır") — bu da VPMV/RSI/
z-score gibi diğer metriklerin HER AİLENİN KENDİ İÇİNDE gerçekten işe
yarayıp yaramadığını gölgeleyebilir.

Bu script aynı 17 metriği, aynı walk-forward (4-fold, kronolojik) yöntemi,
4 aile × 2 yön = 8 AYRI modelde tekrarlar. "indicators" artık her alt
kümede sabit olduğu için feature listesinden çıkarılır.

Kullanım: python -m research.pattern_lab.rf_per_indicator_family_bt
"""

import warnings

import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.rf_signal_quality_bt import (
    _BOOL_FEATURES,
    _MIN_INITIAL_TRAIN_FRAC,
    _N_FOLDS,
    _NUMERIC_FEATURES,
    _TOP_QUANTILE,
    _fetch,
    _prep,
    _stats,
)

_FAMILIES = ["RSI_Cross(9,24)", "HA_Cross", "Supertrend(10,3.0)", "MA200_Cross"]


def _run_family(df: pd.DataFrame, indicators: str, sig_type: str) -> dict | None:
    sub = df[(df["indicators"] == indicators) & (df["signal_type"] == sig_type)]
    sub = sub.sort_values("opened_at").reset_index(drop=True)
    feature_cols = _NUMERIC_FEATURES + _BOOL_FEATURES
    sub = sub.dropna(subset=feature_cols)
    sub = pd.get_dummies(sub, columns=["interval"], prefix="interval")
    dummy_cols = [c for c in sub.columns if c.startswith("interval_")]
    all_features = feature_cols + dummy_cols

    n = len(sub)
    initial_train_end = int(n * _MIN_INITIAL_TRAIN_FRAC)
    remaining = n - initial_train_end
    fold_size = remaining // _N_FOLDS
    if fold_size < 100:
        print(f"\n{indicators} / {sig_type}: yetersiz örnek (n={n}), atlanıyor")
        return None

    print(f"\n{'='*70}\n{indicators} / {sig_type} — n={n}\n{'='*70}")

    fold_results = []
    importances_acc = pd.Series(0.0, index=all_features)

    for fold in range(_N_FOLDS):
        train_end = initial_train_end + fold * fold_size
        test_end = n if fold == _N_FOLDS - 1 else initial_train_end + (fold + 1) * fold_size
        train, test = sub.iloc[:train_end], sub.iloc[train_end:test_end]
        if len(train) < 100 or len(test) < 50:
            continue

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[all_features], train["is_win"])

        test_auc = roc_auc_score(test["is_win"], clf.predict_proba(test[all_features])[:, 1])
        baseline = _stats(test["realized_pnl"])

        test = test.copy()
        test["proba_win"] = clf.predict_proba(test[all_features])[:, 1]
        cutoff = test["proba_win"].quantile(1 - _TOP_QUANTILE)
        top_stats = _stats(test[test["proba_win"] >= cutoff]["realized_pnl"])

        print(f"  Fold {fold+1}: AUC={test_auc:.3f}  baseline={baseline}  üst-%25={top_stats}")

        fold_results.append({"fold": fold + 1, "test_auc": test_auc,
                              "baseline_pf": baseline.get("pf"), "top_pf": top_stats.get("pf")})
        importances_acc += pd.Series(clf.feature_importances_, index=all_features)

    if not fold_results:
        print("  hiçbir fold yeterli örneğe sahip değildi")
        return None

    res_df = pd.DataFrame(fold_results)
    avg_auc = res_df["test_auc"].mean()
    avg_baseline = res_df["baseline_pf"].mean()
    avg_top = res_df["top_pf"].mean()
    n_positive = (res_df["top_pf"] > res_df["baseline_pf"]).sum()
    print(f"  ÖZET: ort_AUC={avg_auc:.3f}  ort_baseline_PF={avg_baseline:.3f}  "
          f"ort_üst25_PF={avg_top:.3f}  ({n_positive}/{len(fold_results)} fold'da iyileşme)")

    importances_avg = (importances_acc / len(fold_results)).sort_values(ascending=False)
    print("  En önemli 5 feature:", ", ".join(importances_avg.head(5).index))

    return {
        "indicators": indicators, "signal_type": sig_type, "n": n,
        "avg_auc": avg_auc, "avg_baseline_pf": avg_baseline, "avg_top_pf": avg_top,
        "n_positive_folds": n_positive, "n_folds": len(fold_results),
    }


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    df = _fetch(cur)
    conn.close()
    print(f"[fetch] {len(df)} kapalı sinyal")
    df = _prep(df)

    summary = []
    for indicators in _FAMILIES:
        for sig_type in ("Long", "Short"):
            result = _run_family(df, indicators, sig_type)
            if result:
                summary.append(result)

    print(f"\n{'='*70}\nGENEL ÖZET (tüm aileler)\n{'='*70}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
