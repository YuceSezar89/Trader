"""
ML değerlendirmesi (2. adım) — elle kurduğumuz eşik yığınının (all_up
booleanları + TA-percentile>=55/90 + HA boolean) yerine, aynı ham
özelliklerin sürekli değerlerini bir XGBoost modeline verip modelin kendi
ağırlıklarını öğrenmesine izin versek daha mı iyi ayırt eder? (24 Tem 2026,
kullanıcı isteği — önce basit kalabalık eşiği test edildi
[[project_combo_clustering_threshold_24tem]], şimdi ML sırası)

Disiplin: KRONOLOJİK train/test ayrımı (ilk %65 train, son %35 test) —
rastgele shuffle YOK, bu proje boyunca kullandığımız IS/OOS mantığının
aynısı. Model test döneminde hiç görülmemiş veriyle değerlendiriliyor.

Long ve Short için AYRI modeller (RSI_Cross+HA_Cross havuzlanmış, indicator
one-hot ile). Özellikler: pct_1h/4h, slope_1h/4h, ha_bull_1h/4h,
d_vol/d_mom/d_volat/d_price (all_up bileşenlerinin ham diff'leri —
booleana indirgemeden), concurrent_count (kalabalık özelliği,
[[project_combo_clustering_feature_24tem]]). Hedef: fwd_ret>0 (ikili).

Kullanım: python -m research.pattern_lab.combo_ml_score_bt
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up

_BASE = os.path.dirname(__file__)
_FEATURES = ["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h",
             "d_vol", "d_mom", "d_volat", "d_price", "concurrent_count", "is_ha_cross"]
_TRAIN_FRAC = 0.65


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


def _concurrent_count(times: np.ndarray, window_hours: float = 4.0) -> np.ndarray:
    window_ns = window_hours * 3600 * 1e9
    counts = np.zeros(len(times), dtype=int)
    left = 0
    for i in range(len(times)):
        while times[i] - times[left] > window_ns:
            left += 1
        counts[i] = i - left
    return counts


def _load_long() -> pd.DataFrame:
    rsi = pd.read_parquet(os.path.join(_BASE, "_cache_rsi_cross_ta_ha.parquet"))
    rsi["is_ha_cross"] = 0
    ha = pd.read_parquet(os.path.join(_BASE, "_cache_ha_cross_ta_ha.parquet"))
    ha["is_ha_cross"] = 1
    df = pd.concat([rsi, ha], ignore_index=True)
    return df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"])


def _load_short() -> pd.DataFrame:
    rsi = pd.read_parquet(os.path.join(_BASE, "_cache_rsi_cross_ta_triple_short.parquet"))
    rsi = _add_all_up(rsi)
    rsi["is_ha_cross"] = 0
    ha = pd.read_parquet(os.path.join(_BASE, "_cache_ha_cross_ta_triple_short.parquet"))
    ha = _add_all_up(ha)
    ha["is_ha_cross"] = 1
    df = pd.concat([rsi, ha], ignore_index=True)
    df = df[df["all_up"]]
    return df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"])


def _run(label: str, df: pd.DataFrame) -> None:
    df = df.sort_values("opened_at").reset_index(drop=True)
    times = df["opened_at"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    df["concurrent_count"] = _concurrent_count(times)
    df["y"] = (df["fwd_ret"] > 0).astype(int)

    n_train = int(len(df) * _TRAIN_FRAC)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    print(f"\n{'='*78}\n{label} — n_train={len(train)} n_test={len(test)} "
          f"(train: {train['opened_at'].min()}..{train['opened_at'].max()}, "
          f"test: {test['opened_at'].min()}..{test['opened_at'].max()})\n{'='*78}")

    X_train, y_train = train[_FEATURES], train["y"]
    X_test, y_test = test[_FEATURES], test["y"]

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=2.0, eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  Test AUC: {auc:.3f} (0.5=rastgele, 1.0=mükemmel)")

    test = test.copy()
    test["proba"] = proba

    print("\n  Özellik önemi (gain bazlı):")
    importances = pd.Series(model.feature_importances_, index=_FEATURES).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"    {feat:18}: {imp:.3f}")

    print("\n  Model skoruna göre çeyreklik kırılım (test döneminde):")
    q = pd.qcut(test["proba"], 4, labels=["Q1(zayıf)", "Q2", "Q3", "Q4(güçlü)"], duplicates="drop")
    for name, grp in test.groupby(q, observed=True):
        s = _stats(grp["fwd_ret"].to_numpy())
        print(f"    {name:10}: n={s['n']:>4}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")

    top_decile = test.nlargest(max(10, len(test) // 10), "proba")
    print(f"\n  ML top-%10 (n={len(top_decile)}): {_stats(top_decile['fwd_ret'].to_numpy())}")

    print("\n  KARŞILAŞTIRMA: elle kurduğumuz kural (aynı test döneminde)")
    # TA-base + kovalama (yöne göre), HA sadece Long'da
    if label == "LONG":
        ta_base = (test["pct_1h"] >= 55) & (test["pct_4h"] >= 55)
        kovalama = ((test["pct_1h"] >= 90) & (test["slope_1h"] > 0)) | ((test["pct_4h"] >= 90) & (test["slope_4h"] > 0))
        ha_ok = (test["ha_bull_1h"] > 0.5) & (test["ha_bull_4h"] > 0.5)
        hand_mask = ta_base & kovalama & ha_ok
    else:
        ta_base = (test["pct_1h"] <= 45) & (test["pct_4h"] <= 45)
        kovalama = ((test["pct_1h"] <= 20) & (test["slope_1h"] < 0)) | ((test["pct_4h"] <= 20) & (test["slope_4h"] < 0))
        hand_mask = ta_base & kovalama
    hand_group = test[hand_mask]
    print(f"    elle kural (n={len(hand_group)}): {_stats(hand_group['fwd_ret'].to_numpy())}")
    print(f"    ML top-%10   (n={len(top_decile)}): {_stats(top_decile['fwd_ret'].to_numpy())}")
    overlap = len(set(hand_group.index) & set(top_decile.index))
    print(f"    örtüşme: elle kuralın %{overlap/max(1,len(hand_group))*100:.1f}'i ML top-%10'da da var")


def main() -> None:
    _run("LONG", _load_long())
    _run("SHORT", _load_short())


if __name__ == "__main__":
    main()
