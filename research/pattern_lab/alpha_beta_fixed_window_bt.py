"""
Alpha/Beta — DÜZELTİLMİŞ pencereyle 4-kapılı doğrulama (20 Tem 2026).

Arka plan: eski formül (9 Tem'de test edilmiş) Beta'yı ~1 aylık pencereden,
Alpha'yı sabit 20 bardan hesaplıyordu — sinyalin kendi ömrüyle (birkaç saat)
alakasız bir zaman ölçeğiydi, sonuç anlamsızdı (Beta rho≈0, Alpha zayıf/
tutarsız). Bugün `indicators/financial_metrics.py`'de düzeltildi:
`_LOOKBACK_SIGNAL_BY_TF` artık her TF'nin GERÇEK medyan sinyal tutulma
süresinin ~3 katını kullanıyor (Beta/Alpha için), Sharpe/Sortino/Calmar/Omega
ayrı bir "rejim" penceresinde (~30 gün) kalmaya devam ediyor.

BU SCRIPT SADECE düzeltmeden SONRA açılan sinyalleri kullanır — eski
(bozuk pencereli) alpha/beta değerleriyle karışmasın diye `_FIX_CUTOFF`
ile filtreleniyor (dosya değişim zamanı + restart payı).

4 kapı: korelasyon, gerçek $ (paper_trades kesişimi), placebo (metrik
karıştırma), split-period (kronolojik ikiye böl — DAR pencere, sadece
bugünün verisi, dürüstçe not düşülüyor).

Kullanım: python -m research.pattern_lab.alpha_beta_fixed_window_bt
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_FIX_CUTOFF = datetime(2026, 7, 20, 13, 0, 0)  # dosya değişimi 12:34 + restart payı
_METRICS = ("alpha", "beta")
_MIN_N = 30
_PLACEBO_ITER = 300


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, signal_type, opened_at, alpha, beta, realized_pnl
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL
          AND alpha IS NOT NULL AND beta IS NOT NULL
          AND opened_at > %s
        """,
        (_FIX_CUTOFF,),
    )
    rows = cur.fetchall()
    return pd.DataFrame(
        rows, columns=["id", "symbol", "signal_type", "opened_at", "alpha", "beta", "realized_pnl"]
    )


def _fetch_real_trades(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT signal_id, pnl_usd FROM paper_trades
        WHERE status = 'closed' AND signal_id IS NOT NULL
          AND opened_at > %s
        """,
        (_FIX_CUTOFF,),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["signal_id", "pnl_usd"])


def _tercile_report(sub: pd.DataFrame, metric: str, target: str, label: str) -> None:
    if len(sub) < _MIN_N:
        print(f"    {label}: yetersiz örnek (n={len(sub)})")
        return
    rho, p = spearmanr(sub[metric], sub[target])
    tercile = pd.qcut(sub[metric], 3, labels=["alt", "orta", "üst"], duplicates="drop")
    g = sub.groupby(tercile, observed=True)[target].agg(
        ort=lambda s: s.mean(), n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(f"    {label}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")
    print(g.to_string().replace("\n", "\n      "))


def _placebo(sub: pd.DataFrame, metric: str, target: str, n_iter: int = _PLACEBO_ITER) -> float:
    """Metrik değerleri sinyal-target eşleşmesinden BAĞIMSIZ karıştırılır —
    gerçek |rho| rastgele karıştırmalardan daha büyük mü sıklıkla?"""
    real_rho, _ = spearmanr(sub[metric], sub[target])
    rng = np.random.default_rng(42)
    vals = sub[metric].to_numpy()
    target_vals = sub[target].to_numpy()
    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(vals)
        rho, _ = spearmanr(shuffled, target_vals)
        if abs(rho) >= abs(real_rho):
            count_ge += 1
    return count_ge / n_iter


def _split_period(sub: pd.DataFrame, metric: str, target: str) -> None:
    if len(sub) < _MIN_N * 2:
        print(f"    yetersiz örnek (n={len(sub)}), split yapılmadı")
        return
    med = sub["opened_at"].median()
    first = sub[sub["opened_at"] < med]
    second = sub[sub["opened_at"] >= med]
    r1, p1 = spearmanr(first[metric], first[target]) if len(first) >= _MIN_N else (float("nan"), float("nan"))
    r2, p2 = spearmanr(second[metric], second[target]) if len(second) >= _MIN_N else (float("nan"), float("nan"))
    print(f"    bölünme: {med} | ilk yarı n={len(first)} rho={r1:+.3f}(p={p1:.3f}) "
          f"| ikinci yarı n={len(second)} rho={r2:+.3f}(p={p2:.3f})")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()

    df = _fetch_signals(cur)
    print(f"[fetch] düzeltme sonrası ({_FIX_CUTOFF}) {len(df)} kapalı sinyal (alpha+beta dolu)\n")
    if df.empty:
        print("Yeterli veri yok — düzeltmeden bu yana çok az sinyal kapanmış olabilir.")
        return

    real_trades = _fetch_real_trades(cur)
    conn.close()
    print(f"[fetch] gerçek $ kesişimi için {len(real_trades)} paper_trade\n")

    for metric in _METRICS:
        print(f"{'='*70}\n{metric.upper()}\n{'='*70}")
        for sig_type in ("Long", "Short"):
            sub = df[df["signal_type"] == sig_type]
            print(f"\n--- {sig_type} ---")

            print("  [1] KORELASYON (realized_pnl)")
            _tercile_report(sub, metric, "realized_pnl", "signals.realized_pnl")

            print("  [2] GERÇEK $ (paper_trades.pnl_usd)")
            merged = sub.merge(real_trades, left_on="id", right_on="signal_id", how="inner")
            _tercile_report(merged, metric, "pnl_usd", "paper_trades.pnl_usd")

            print("  [3] PLACEBO (metrik karıştırma)")
            if len(sub) >= _MIN_N:
                p_val = _placebo(sub, metric, "realized_pnl")
                real_rho, real_p = spearmanr(sub[metric], sub["realized_pnl"])
                print(f"    gerçek rho={real_rho:+.3f} (p={real_p:.4f}) — "
                      f"rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{p_val*100:.1f}")
            else:
                print(f"    yetersiz örnek (n={len(sub)})")

            print("  [4] SPLIT-PERIOD (DAR pencere — sadece bugünkü veri, dikkatli yorumla)")
            _split_period(sub, metric, "realized_pnl")


if __name__ == "__main__":
    main()
