"""
"Önceki sinyale göre iyileşti mi" konsensüs testi (20 Tem 2026, kullanıcı
fikri): elimizdeki TÜM sinyal-anı metriklerini HAM DEĞER olarak değil,
"aynı sembol+TF+indikatör+yöndeki BİR ÖNCEKİ sinyale göre yükseldi mi (1)
yoksa düşmedi mi (0)" ikili bayrağına çevirip test eder.

Felsefe Madde 4 (Confluence/Divergence, bugünün en güçlü bulgusu) ve
13'lü konsensüs fikriyle AYNI aile — kompozit/ham skor yerine "kaçı aynı
yönde" sayımı. Devisso panelinin Δ/Ratio sütunlarının TÜM metriklere
genelleştirilmiş hali.

İki test:
  A) confluence_count (kaç metrik iyileşti) — realized_pnl ile korelasyon,
     placebo, split-period (Madde-4 stiliyle)
  B) Aynı ikili bayraklar Random Forest'a feature olarak verilir, ham-değer
     modeliyle (rf_signal_quality_bt.py) karşılaştırılır

"Önceki sinyal" = aynı (symbol, interval, indicators, signal_type) grubunda
opened_at'a göre bir önceki satır — SQL LAG() ile tek sorguda, N+1 yok.
alpha/beta bilerek DIŞARIDA (bugün pencere düzeltmesi oldu, eski veri
kirli — bkz. alpha_beta_fixed_window_bt.py).

Kullanım: python -m research.pattern_lab.prev_signal_delta_consensus_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from config import Config

_METRICS = [
    "vpms_score", "mtf_score", "rsi", "atr_pct",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "z_score_entry", "devisso_score", "cvd_slope",
    "vp_score", "vp_score_real",
    "vpmv_pre_avg", "vpmv_ratio", "vpmv_slope", "strength",
]
_N_FOLDS = 4
_MIN_INITIAL_TRAIN_FRAC = 0.40
_TOP_QUANTILE = 0.25
_PLACEBO_ITER = 300


def _fetch(cur) -> pd.DataFrame:
    lag_cols = ", ".join(
        f"LAG({m}) OVER w AS prev_{m}" for m in _METRICS if m != "atr_pct"
    )
    cur.execute(
        f"""
        SELECT id, symbol, interval, indicators, signal_type, opened_at, open_price, atr,
               vpms_score, mtf_score, rsi, sharpe_ratio, sortino_ratio, calmar_ratio,
               z_score_entry, devisso_score, cvd_slope, vp_score, vp_score_real,
               vpmv_pre_avg, vpmv_ratio, vpmv_slope, strength, realized_pnl,
               LAG(atr) OVER w AS prev_atr, LAG(open_price) OVER w AS prev_open_price,
               {lag_cols}
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL
        WINDOW w AS (PARTITION BY symbol, interval, indicators, signal_type ORDER BY opened_at)
        ORDER BY opened_at ASC
        """
    )
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_pct"] = df["atr"] / df["open_price"] * 100.0
    df["prev_atr_pct"] = df["prev_atr"] / df["prev_open_price"] * 100.0
    df["is_win"] = (df["realized_pnl"] > 0).astype(int)

    flag_cols = []
    for m in _METRICS:
        cur_col, prev_col, flag_col = m, f"prev_{m}", f"up_{m}"
        valid = df[cur_col].notna() & df[prev_col].notna()
        df[flag_col] = np.where(valid, (df[cur_col] > df[prev_col]).astype(float), np.nan)
        flag_cols.append(flag_col)

    df["confluence_count"] = df[flag_cols].sum(axis=1, skipna=True)
    df["confluence_valid_n"] = df[flag_cols].notna().sum(axis=1)
    return df, flag_cols


def _stats(rets: pd.Series) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    g = rets[rets > 0].sum()
    l = -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _placebo(sub: pd.DataFrame, n_iter: int = _PLACEBO_ITER) -> float:
    real_rho, _ = spearmanr(sub["confluence_count"], sub["realized_pnl"])
    rng = np.random.default_rng(42)
    vals = sub["confluence_count"].to_numpy()
    target = sub["realized_pnl"].to_numpy()
    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(vals)
        rho, _ = spearmanr(shuffled, target)
        if abs(rho) >= abs(real_rho):
            count_ge += 1
    return count_ge / n_iter


def _test_a_confluence(df: pd.DataFrame, sig_type: str) -> None:
    print(f"\n{'='*70}\nTEST A — confluence_count (kaç metrik iyileşti), {sig_type}\n{'='*70}")
    sub = df[(df["signal_type"] == sig_type) & (df["confluence_valid_n"] >= len(_METRICS) - 2)].copy()
    print(f"  n={len(sub)} (en az {len(_METRICS)-2}/{len(_METRICS)} metrik geçerli bir önceki sinyale sahip)")
    if len(sub) < 100:
        print("  yetersiz örnek")
        return

    rho, p = spearmanr(sub["confluence_count"], sub["realized_pnl"])
    print(f"\n  Korelasyon: rho={rho:+.3f} (p={p:.4f})")
    g = sub.groupby("confluence_count")["realized_pnl"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(g.to_string())

    tercile = pd.qcut(sub["confluence_count"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
    gt = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print("\n  Tercile:")
    print(gt.to_string())

    p_val = _placebo(sub)
    print(f"\n  Placebo: gerçek rho={rho:+.3f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{p_val*100:.1f}")

    med = sub["opened_at"].median()
    first, second = sub[sub["opened_at"] < med], sub[sub["opened_at"] >= med]
    r1, p1 = spearmanr(first["confluence_count"], first["realized_pnl"])
    r2, p2 = spearmanr(second["confluence_count"], second["realized_pnl"])
    print(f"  Split-period: ilk yarı n={len(first)} rho={r1:+.3f}(p={p1:.3f}) | "
          f"ikinci yarı n={len(second)} rho={r2:+.3f}(p={p2:.3f})")


def _test_b_rf(df: pd.DataFrame, flag_cols: list, sig_type: str) -> None:
    print(f"\n{'='*70}\nTEST B — ikili bayraklar RF feature'ı olarak, {sig_type}\n{'='*70}")
    sub = df[(df["signal_type"] == sig_type)].dropna(subset=flag_cols).sort_values("opened_at").reset_index(drop=True)
    n = len(sub)
    initial_train_end = int(n * _MIN_INITIAL_TRAIN_FRAC)
    remaining = n - initial_train_end
    fold_size = remaining // _N_FOLDS
    if fold_size < 200:
        print(f"  yetersiz örnek (n={n})")
        return

    fold_results = []
    for fold in range(_N_FOLDS):
        train_end = initial_train_end + fold * fold_size
        test_end = n if fold == _N_FOLDS - 1 else initial_train_end + (fold + 1) * fold_size
        train, test = sub.iloc[:train_end], sub.iloc[train_end:test_end]
        if len(train) < 200 or len(test) < 100:
            continue

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[flag_cols], train["is_win"])
        test_auc = roc_auc_score(test["is_win"], clf.predict_proba(test[flag_cols])[:, 1])

        baseline = _stats(test["realized_pnl"])
        test = test.copy()
        test["proba_win"] = clf.predict_proba(test[flag_cols])[:, 1]
        cutoff = test["proba_win"].quantile(1 - _TOP_QUANTILE)
        top_stats = _stats(test[test["proba_win"] >= cutoff]["realized_pnl"])

        fold_results.append({"fold": fold + 1, "test_auc": test_auc,
                              "baseline_pf": baseline.get("pf"), "top_pf": top_stats.get("pf")})

    if not fold_results:
        print("  hiçbir fold yeterli örneğe sahip değildi")
        return
    res_df = pd.DataFrame(fold_results)
    print(res_df.to_string(index=False))
    print(f"  Ortalama test AUC: {res_df['test_auc'].mean():.3f} | "
          f"Ortalama baseline PF: {res_df['baseline_pf'].mean():.3f} | "
          f"Ortalama üst-%25 PF: {res_df['top_pf'].mean():.3f}")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    df = _fetch(cur)
    conn.close()
    print(f"[fetch] {len(df)} kapalı sinyal (LAG ile önceki-sinyal karşılaştırması dahil)")

    df, flag_cols = _prep(df)

    for sig_type in ("Long", "Short"):
        _test_a_confluence(df, sig_type)
        _test_b_rf(df, flag_cols, sig_type)


if __name__ == "__main__":
    main()
