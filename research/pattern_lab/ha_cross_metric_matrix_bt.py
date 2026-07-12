"""
HA_Cross (paper_trades'te gerçek veride en kârlı strateji, [[project_paper_trading]])
için elimizdeki TÜM sinyal-anı metriklerini (alpha, beta, vp_buy_avg, vp_sell_avg,
cvd_slope, vpmv_pre_avg, vpmv_ratio, vpmv_slope) TEK TEK filtre olarak test eder.

Kullanıcı isteği (12 Tem 2026): "elimizdeki tüm metrikler ile bir matris kursak" —
TAM kombinatoryal (2^N) matris KURULMUYOR bilinçli olarak; her metrik BAĞIMSIZ
test ediliyor (threshold_optimizer.py::_run_single_var_on_df, aynı IS/OOS+
split-period+placebo disiplini). Tekli testte hayatta kalanlar olursa
regime_matrix_bt.py'deki greedy-stepwise ile kombine edilmesi bir sonraki adım —
bugünkü do_open_streak eşik-optimizasyonu denemesinde görülen overfitting
riskinden kaçınmak için ([[project_pattern_lab]]).

Metrikler `signals` tablosunda sinyal AÇILIŞ anında kaydediliyor (look-ahead
yok) — database/models.py::Signal.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.threshold_optimizer import (  # pylint: disable=wrong-import-position
    MIN_N, _apply_rule, _pf, _rule_text, _single_var_placebo_distribution, _single_var_search,
)

INDICATOR = "HA_Cross"
DIRECTIONS = ["Long", "Short"]
METRICS = [
    "alpha", "beta", "vp_buy_avg", "vp_sell_avg",
    "cvd_slope", "vpmv_pre_avg", "vpmv_ratio", "vpmv_slope",
]


def _fetch(indicator: str, direction: str, col: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT {col}, realized_pnl, opened_at
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
          AND {col} IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    return df


def _test_metric(direction: str, col: str) -> dict:
    df = _fetch(INDICATOR, direction, col)
    n = len(df)
    if n < MIN_N * 4:
        print(f"  {col:14} n={n:>5}  örnek yetersiz, atlanıyor")
        return {"metric": col, "direction": direction, "n": n, "verdict": "yetersiz"}

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_df = df[df["opened_at"] < mid].reset_index(drop=True)
    oos_df = df[df["opened_at"] >= mid].reset_index(drop=True)

    rule, is_pf, is_n = _single_var_search(is_df, col)
    oos_baseline = _pf(oos_df)
    oos_filtered = _apply_rule(oos_df, rule)
    oos_stats = _pf(oos_filtered) if len(oos_filtered) > 0 else {"pf": 0, "n": 0}

    oos_mid = mid + (t_max - mid) / 2
    oos_first = _apply_rule(oos_df[oos_df["opened_at"] < oos_mid], rule)
    oos_second = _apply_rule(oos_df[oos_df["opened_at"] >= oos_mid], rule)
    s1 = _pf(oos_first) if len(oos_first) > 0 else {"pf": 0, "n": 0}
    s2 = _pf(oos_second) if len(oos_second) > 0 else {"pf": 0, "n": 0}

    placebo_pfs = _single_var_placebo_distribution(is_df, col)
    rank = 0.0
    if placebo_pfs:
        arr = np.array(placebo_pfs)
        rank = float((arr < is_pf).mean() * 100)

    verdict = "GEÇERLİ ADAY" if (
        rank >= 90 and oos_stats.get("pf", 0) > oos_baseline.get("pf", 0)
        and s1.get("pf", 0) > 1 and s2.get("pf", 0) > 1 and s1.get("n", 0) >= 20 and s2.get("n", 0) >= 20
    ) else "zayıf"

    print(f"  {col:14} n={n:>5}  kural={_rule_text(rule)}")
    print(f"                 OOS: baseline_PF={oos_baseline.get('pf',0):.3f} → "
          f"filtreli_PF={oos_stats.get('pf',0):.3f} (n={oos_stats.get('n',0)}) | "
          f"split ilk={s1.get('pf',0):.3f}(n={s1.get('n',0)}) ikinci={s2.get('pf',0):.3f}(n={s2.get('n',0)}) | "
          f"placebo=%{rank:.0f} → {verdict}")

    return {
        "metric": col, "direction": direction, "n": n,
        "is_pf": is_pf, "oos_baseline_pf": oos_baseline.get("pf", 0),
        "oos_filtered_pf": oos_stats.get("pf", 0), "oos_n": oos_stats.get("n", 0),
        "s1_pf": s1.get("pf", 0), "s2_pf": s2.get("pf", 0),
        "placebo_rank": rank, "verdict": verdict,
    }


def run() -> None:
    print(f"HA_Cross — {len(METRICS)} metrik × {len(DIRECTIONS)} yön = "
          f"{len(METRICS)*len(DIRECTIONS)} tekli test\n")

    results = []
    for direction in DIRECTIONS:
        print(f"\n{'='*70}\n{INDICATOR} — {direction}\n{'='*70}")
        for col in METRICS:
            results.append(_test_metric(direction, col))

    print(f"\n\n{'='*70}\nÖZET (placebo sırasına göre)\n{'='*70}")
    print(f"{'yön':6} {'metrik':14} {'n':>6} {'OOS_taban':>10} {'OOS_filtre':>11} {'plasebo%':>9} {'sonuç':>14}")
    for r in sorted(results, key=lambda x: -x.get("placebo_rank", 0)):
        if r.get("verdict") == "yetersiz":
            continue
        print(f"{r['direction']:6} {r['metric']:14} {r['n']:>6} "
              f"{r.get('oos_baseline_pf',0):>10.3f} {r.get('oos_filtered_pf',0):>11.3f} "
              f"{r.get('placebo_rank',0):>9.0f} {r['verdict']:>14}")

    survivors = [r for r in results if r.get("verdict") == "GEÇERLİ ADAY"]
    print(f"\n>>> {len(survivors)} metrik tam disiplini geçti: "
          f"{[(r['direction'], r['metric']) for r in survivors]}")


if __name__ == "__main__":
    run()
