"""
Random Forest ile sinyal kalitesi denemesi (20 Tem 2026, kullanıcı isteği:
"random forest ile bir dene").

Bugüne kadarki derste (bkz. memory/project_ml_vision.md, "Tanım → Ölçüm →
Özellik → ML"): Adım 1-2 bitti, Adım 3 (özellik seçimi) bugün büyük ölçüde
yapıldı — bu script Adım 4'ün ilk, DENEYSEL/dürüst versiyonu.

Kurallar (geçmiş hatalardan — bkz. DevisSoTrader/vpmv_cons XGBoost
eleştirileri):
  - Label = getiri (realized_pnl > 0), MUTLAK fiyat DEĞİL
  - Feature'lar sadece sinyal AÇILIŞ anında bilinen değerler (signals
    tablosu, look-ahead yok)
  - Train/test bölünmesi KRONOLOJİK (rastgele shuffle DEĞİL) — geleceği
    geçmişten tahmin etmeye çalışmak gerçekçi OOS testi taklit eder
  - alpha/beta DIŞARIDA BIRAKILDI — bu iki metriğin hesap penceresi bugün
    (20 Tem) düzeltildi, eski satırlar bozuk formülle hesaplanmış, dataset'e
    karışırsa gürültü ekler (bkz. project_sdf_ve_diger_filtreler.md)
  - Long/Short AYRI modellenir (bugün tekrar tekrar görülen asimetri deseni)

Kullanım: python -m research.pattern_lab.rf_signal_quality_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from config import Config

_NUMERIC_FEATURES = [
    "vpms_score", "mtf_score", "rsi", "atr_pct",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "z_score_entry", "devisso_score", "cvd_slope",
    "vp_score", "vp_score_real",
    "vpmv_pre_avg", "vpmv_ratio", "vpmv_slope",
    "strength",
]
# hour_utc/day_of_week ÇIKARILDI (20 Tem 2026, kullanıcı denetimi) — ilk
# denemede feature importance'ın en tepesindeydi ama tek-bölünme test seti
# sadece ~2.5 gün kapsıyordu (birkaç takvim günü) — gerçek haftalık döngü
# yerine "test penceresine özgü rejimi ezberleme" riski yüksekti.
_BOOL_FEATURES = ["st_confirmed"]
_CATEGORICAL_FEATURES = ["interval", "indicators"]
_TOP_QUANTILE = 0.25
_N_FOLDS = 4
_MIN_INITIAL_TRAIN_FRAC = 0.40  # ilk fold'dan önce en az bu kadar veri train'de olsun


def _fetch(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, signal_type, interval, indicators, opened_at, open_price, atr,
               vpms_score, mtf_score, rsi, sharpe_ratio, sortino_ratio, calmar_ratio,
               z_score_entry, devisso_score, cvd_slope, vp_score, vp_score_real,
               vpmv_pre_avg, vpmv_ratio, vpmv_slope, st_confirmed, strength,
               realized_pnl
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY opened_at ASC
        """
    )
    cols = [
        "symbol", "signal_type", "interval", "indicators", "opened_at", "open_price", "atr",
        "vpms_score", "mtf_score", "rsi", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "z_score_entry", "devisso_score", "cvd_slope", "vp_score", "vp_score_real",
        "vpmv_pre_avg", "vpmv_ratio", "vpmv_slope", "st_confirmed", "strength",
        "realized_pnl",
    ]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_pct"] = df["atr"] / df["open_price"] * 100.0
    df["st_confirmed"] = df["st_confirmed"].fillna(False).astype(int)
    df["is_win"] = (df["realized_pnl"] > 0).astype(int)
    for col in _NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stats(rets: pd.Series) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    g = rets[rets > 0].sum()
    l = -rets[rets < 0].sum()
    return {
        "n": len(rets),
        "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _run_side(df: pd.DataFrame, sig_type: str) -> None:
    sub = df[df["signal_type"] == sig_type].sort_values("opened_at").reset_index(drop=True)
    feature_cols = _NUMERIC_FEATURES + _BOOL_FEATURES
    sub = sub.dropna(subset=feature_cols)
    sub = pd.get_dummies(sub, columns=_CATEGORICAL_FEATURES, prefix=_CATEGORICAL_FEATURES)
    dummy_cols = [c for c in sub.columns if any(c.startswith(f"{p}_") for p in _CATEGORICAL_FEATURES)]
    all_features = feature_cols + dummy_cols

    n = len(sub)
    initial_train_end = int(n * _MIN_INITIAL_TRAIN_FRAC)
    remaining = n - initial_train_end
    fold_size = remaining // _N_FOLDS
    if fold_size < 200:
        print(f"\n{sig_type}: yetersiz örnek (n={n}), atlanıyor")
        return

    print(f"\n{'='*70}\n{sig_type} — walk-forward, {_N_FOLDS} fold (n={n}, "
          f"ilk train={initial_train_end}, fold başına test~{fold_size})\n{'='*70}")

    fold_results = []
    importances_acc = pd.Series(0.0, index=all_features)

    for fold in range(_N_FOLDS):
        train_end = initial_train_end + fold * fold_size
        test_end = n if fold == _N_FOLDS - 1 else initial_train_end + (fold + 1) * fold_size
        train, test = sub.iloc[:train_end], sub.iloc[train_end:test_end]
        if len(train) < 200 or len(test) < 100:
            continue

        X_train, y_train = train[all_features], train["is_win"]
        X_test, y_test = test[all_features], test["is_win"]

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        baseline = _stats(test["realized_pnl"])

        test = test.copy()
        test["proba_win"] = clf.predict_proba(X_test)[:, 1]
        cutoff = test["proba_win"].quantile(1 - _TOP_QUANTILE)
        top = test[test["proba_win"] >= cutoff]
        top_stats = _stats(top["realized_pnl"])
        bottom = test[test["proba_win"] < test["proba_win"].quantile(_TOP_QUANTILE)]
        bottom_stats = _stats(bottom["realized_pnl"])

        print(f"\n  --- Fold {fold+1}/{_N_FOLDS} | test: {test['opened_at'].min()} → "
              f"{test['opened_at'].max()} (n={len(test)}) ---")
        print(f"    AUC: train={train_auc:.3f} test={test_auc:.3f}")
        print(f"    Baseline:      {baseline}")
        print(f"    Üst %{_TOP_QUANTILE*100:.0f} güven: {top_stats}")
        print(f"    Alt %{_TOP_QUANTILE*100:.0f} güven:  {bottom_stats}")

        fold_results.append(
            {"fold": fold + 1, "test_auc": test_auc, "baseline_pf": baseline.get("pf"),
             "top_pf": top_stats.get("pf"), "bottom_pf": bottom_stats.get("pf")}
        )
        importances_acc += pd.Series(clf.feature_importances_, index=all_features)

    if not fold_results:
        print("  hiçbir fold yeterli örneğe sahip değildi")
        return

    print(f"\n  === {sig_type} ÖZET ({len(fold_results)} fold) ===")
    res_df = pd.DataFrame(fold_results)
    print(res_df.to_string(index=False))
    print(f"\n  Ortalama test AUC: {res_df['test_auc'].mean():.3f}")
    print(f"  Ortalama baseline PF: {res_df['baseline_pf'].mean():.3f}  |  "
          f"Ortalama üst-%25 PF: {res_df['top_pf'].mean():.3f}  |  "
          f"Ortalama alt-%25 PF: {res_df['bottom_pf'].mean():.3f}")
    n_positive_folds = (res_df["top_pf"] > res_df["baseline_pf"]).sum()
    print(f"  Üst-%25 PF > baseline PF olan fold sayısı: {n_positive_folds}/{len(fold_results)}")

    importances_avg = (importances_acc / len(fold_results)).sort_values(ascending=False)
    print("\n  Ortalama feature importance (üst 10, fold'lar arası):")
    for feat, imp in importances_avg.head(10).items():
        print(f"    {feat:30} {imp:.4f}")


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

    for sig_type in ("Long", "Short"):
        _run_side(df, sig_type)


if __name__ == "__main__":
    main()
