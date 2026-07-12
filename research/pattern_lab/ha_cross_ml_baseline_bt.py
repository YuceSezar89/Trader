"""
HA_Cross için ilk (küçük, kasıtlı sade) ML denemesi — bugün tek tek test edilen
sinyal-anı metriklerin (alpha, beta, vp_buy_avg, vp_sell_avg, cvd_slope,
vpmv_pre_avg, vpmv_ratio, vpmv_slope, mtf_score, rank_score, vs_btc) HEPSİNİ
BİRLİKTE bir XGBoost sınıflandırıcıya veriyor — tek değişken taramasının
kaçırabileceği etkileşimleri yakalayabilir mi diye.

Disiplin bugünkünle AYNI, sadece "eşik arama" yerine "model eğitimi":
1. Kronolojik IS/OOS (%50/%50) — SADECE IS'te eğitiliyor.
2. OOS'ta model olasılığına göre üst tercil (en güvenilir %33) seçilip
   baseline'la (tüm OOS sinyalleri) karşılaştırılıyor.
3. OOS'un kendi içinde split-period sağlamlığı.
4. Placebo: IS etiketleri (kazanç/kayıp) karıştırılıp model YENİDEN eğitiliyor,
   aynı seçim OOS'ta tekrarlanıyor — gerçek modelin şanstan anlamlı şekilde
   iyi olup olmadığını ölçer (30 tekrar, XGBoost yeniden eğitim maliyeti
   nedeniyle bugünkü 200'lük placebo sayısından daha az).

Kasıtlı sade tutuldu: küçük ağaç sayısı/derinliği (max_depth=3, n_estimators=100),
market_structure/candle_pattern gibi kategorik/çok-değerli alanlar bu ilk
denemede DAHIL EDİLMEDİ (sadece sayısal özellikler) — overfitting riskini
düşük tutmak için.
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

INDICATOR = "HA_Cross"
FEATURES = [
    "alpha",
    "beta",
    "vp_buy_avg",
    "vp_sell_avg",
    "cvd_slope",
    "vpmv_pre_avg",
    "vpmv_ratio",
    "vpmv_slope",
    "mtf_score",
    "rank_score",
    "vs_btc",
]
N_PLACEBO = 30
TOP_FRACTION = 0.33


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cols = ", ".join(FEATURES)
    q = f"""
        SELECT {cols}, signal_type, realized_pnl, opened_at
        FROM signals
        WHERE indicators = %s AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(INDICATOR,))
    conn.close()
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df["is_long"] = (df["signal_type"] == "Long").astype(int)
    df["y"] = (df["realized_pnl"] > 0).astype(int)
    return df


def _train(train_df: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    model.fit(train_df[FEATURES + ["is_long"]], y)
    return model


def _top_tercile_pf(oos_df: pd.DataFrame, proba: np.ndarray, frac: float = TOP_FRACTION) -> dict:
    thr = np.quantile(proba, 1 - frac)
    sub = oos_df[proba >= thr]
    return _stats(sub["realized_pnl"].to_numpy() / 100)


def run() -> None:
    raw = _fetch()
    df = _prepare(raw)
    print(
        f"HA_Cross (Long+Short) — ham sinyal: {len(raw):,} | özellik-tam (NaN atıldı): {len(df):,}\n"
    )

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_df = df[df["opened_at"] < mid].reset_index(drop=True)
    oos_df = df[df["opened_at"] >= mid].reset_index(drop=True)
    print(f"dönem: {t_min} .. {t_max} | IS: {len(is_df)} | OOS: {len(oos_df)}\n")

    if len(is_df) < 200 or len(oos_df) < 100:
        print("Örneklem ML için çok küçük, atlanıyor.")
        return

    model = _train(is_df, is_df["y"].to_numpy())
    proba_oos = model.predict_proba(oos_df[FEATURES + ["is_long"]])[:, 1]

    baseline = _stats(oos_df["realized_pnl"].to_numpy() / 100)
    top = _top_tercile_pf(oos_df, proba_oos)

    print(f"{'grup':28} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(
        f"{'OOS baseline (tümü)':28} {baseline.get('n',0):>6} {baseline.get('wr',0):>6} "
        f"{baseline.get('ort_%',0):>8} {baseline.get('pf',0):>7}"
    )
    print(
        f"{'OOS üst tercil (model)':28} {top.get('n',0):>6} {top.get('wr',0):>6} "
        f"{top.get('ort_%',0):>8} {top.get('pf',0):>7}"
    )

    # split-period (üst tercil OOS'un kendi içinde)
    oos_mid = mid + (t_max - mid) / 2
    thr = np.quantile(proba_oos, 1 - TOP_FRACTION)
    sel_mask = proba_oos >= thr
    sel_df = oos_df[sel_mask]
    s1 = _stats(sel_df[sel_df["opened_at"] < oos_mid]["realized_pnl"].to_numpy() / 100)
    s2 = _stats(sel_df[sel_df["opened_at"] >= oos_mid]["realized_pnl"].to_numpy() / 100)
    print(
        f"\nsplit-period (üst tercil): ilk_yari n={s1.get('n',0)} PF={s1.get('pf',0)} | "
        f"ikinci_yari n={s2.get('n',0)} PF={s2.get('pf',0)}"
    )

    # Placebo: IS etiketlerini karıştır, yeniden eğit, aynı seçimi OOS'ta tekrarla
    rng = np.random.default_rng(42)
    y_is = is_df["y"].to_numpy().copy()
    placebo_pfs = []
    for _ in range(N_PLACEBO):
        y_shuf = rng.permutation(y_is)
        m = _train(is_df, y_shuf)
        p = m.predict_proba(oos_df[FEATURES + ["is_long"]])[:, 1]
        s = _top_tercile_pf(oos_df, p)
        if s.get("n", 0) >= 20:
            placebo_pfs.append(s.get("pf", 0.0))

    real_pf = top.get("pf", 0.0)
    if placebo_pfs:
        arr = np.array(placebo_pfs)
        rank = float((arr < real_pf).mean() * 100)
        print(
            f"\nplacebo (n={len(arr)}): ort={arr.mean():.3f} p90={np.percentile(arr,90):.3f} "
            f"max={arr.max():.3f} | gerçek PF={real_pf:.3f} (placebo'nun %{rank:.0f}'ini geçiyor)"
        )

    print("\nÖzellik önem sırası (gerçek model):")
    importances = sorted(
        zip(FEATURES + ["is_long"], model.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        print(f"  {name:14} {imp:.3f}")


if __name__ == "__main__":
    run()
