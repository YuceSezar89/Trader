"""
Finansal metrik korelasyonu testi (BEKLEYEN_ANALIZLER'de eski madde, eşik
20 Tem 2026'da fazlasıyla aşıldı: 102.295 sinyalde Sharpe/Sortino/Calmar/IR
hepsi dolu, 100+ isteniyordu).

Soru: sinyal açılış anındaki Sharpe/Sortino/Calmar/Information ratio
(indicators/financial_metrics.py, "rejim" penceresi ~30 gün) realized_pnl'i
öngörüyor mu?

ÖNEMLİ UYARI: calmar_ratio'nun payda formülü (current_dd → max_dd) BUGÜN
(20 Tem 2026) düzeltildi — eski satırlar farklı/bozuk formülle hesaplanmış.
Bu yüzden calmar_ratio AYRI olarak hem TÜM veri hem SADECE düzeltme-sonrası
veriyle test edilir (karşılaştırma için). Sharpe/Sortino/Information ratio
formülü değişmedi (sadece 1m/5m'de pencere kısaltıldı, maliyet için) — tüm
veriyle test edilir.

4 kapı: korelasyon+tercile, gerçek $, placebo, split-period. Her metrik
Long/Short ayrı.

Kullanım: python -m research.pattern_lab.financial_ratio_correlation_bt
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_FIX_CUTOFF = datetime(2026, 7, 20, 13, 0, 0)
_METRICS = ["sharpe_ratio", "sortino_ratio", "information_ratio"]
_MIN_N = 100
_PLACEBO_ITER = 300


def _fetch(cur) -> pd.DataFrame:
    # close_reason='timeout' HARİÇ TUTULUYOR (20 Tem 2026) — sweep_timeouts()
    # düzeltilmeden önce (bkz. signal_lifecycle_manager.py) timeout kapanışı
    # HER ZAMAN open_price'ı close_price olarak kullanıyordu, yani bu 6.782
    # sinyalin realized_pnl'i her zaman sahte %0.000'dı — korelasyonu
    # gürültüye boğuyordu.
    cur.execute(
        """
        SELECT id, symbol, signal_type, opened_at,
               sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio,
               realized_pnl
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL
          AND close_reason != 'timeout'
        """
    )
    cols = ["id", "symbol", "signal_type", "opened_at", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "information_ratio", "realized_pnl"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_real_trades(cur) -> pd.DataFrame:
    cur.execute(
        "SELECT signal_id, pnl_usd FROM paper_trades WHERE status='closed' AND signal_id IS NOT NULL"
    )
    return pd.DataFrame(cur.fetchall(), columns=["signal_id", "pnl_usd"])


def _tercile_report(sub: pd.DataFrame, metric: str, target: str) -> tuple:
    sub = sub.dropna(subset=[metric, target])
    if len(sub) < _MIN_N:
        return None, None, len(sub)
    rho, p = spearmanr(sub[metric], sub[target])
    try:
        tercile = pd.qcut(sub[metric], 3, labels=["alt", "orta", "üst"], duplicates="drop")
    except ValueError:
        # Çok sayıda tekrarlayan değer (ör. eski ±50 yapışma kalıntısı) qcut'un
        # 3 ayrı dilim oluşturmasını engelleyebiliyor — bin sayısını otomatik
        # düşürüyoruz, çökmek yerine ne kadar mümkünse o kadar dilim gösteriyoruz.
        try:
            tercile = pd.qcut(sub[metric], 2, labels=["alt", "üst"], duplicates="drop")
        except ValueError:
            print("      (çok fazla tekrarlayan değer, dilim oluşturulamadı)")
            return rho, p, len(sub)
    g = sub.groupby(tercile, observed=True)[target].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(g.to_string().replace("\n", "\n      "))
    return rho, p, len(sub)


def _placebo(sub: pd.DataFrame, metric: str, target: str, n_iter: int = _PLACEBO_ITER) -> float:
    sub = sub.dropna(subset=[metric, target])
    real_rho, _ = spearmanr(sub[metric], sub[target])
    rng = np.random.default_rng(42)
    vals = sub[metric].to_numpy()
    t = sub[target].to_numpy()
    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(vals)
        rho, _ = spearmanr(shuffled, t)
        if abs(rho) >= abs(real_rho):
            count_ge += 1
    return count_ge / n_iter


def _run_metric(df: pd.DataFrame, real_trades: pd.DataFrame, metric: str, sig_type: str) -> None:
    sub = df[df["signal_type"] == sig_type].dropna(subset=[metric])
    print(f"\n--- {metric} / {sig_type} (n={len(sub)}) ---")
    if len(sub) < _MIN_N:
        print("  yetersiz örnek")
        return

    print("  [1] Korelasyon (realized_pnl):")
    rho, p, n = _tercile_report(sub, metric, "realized_pnl")
    print(f"      rho={rho:+.3f} (p={p:.4f})" if rho is not None else "  yetersiz")

    merged = sub.merge(real_trades, left_on="id", right_on="signal_id", how="inner")
    print(f"  [2] Gerçek $ (n={len(merged)}):")
    if len(merged) >= _MIN_N:
        rho_r, p_r, _ = _tercile_report(merged, metric, "pnl_usd")
        print(f"      rho={rho_r:+.3f} (p={p_r:.4f})")
    else:
        print("      yetersiz örnek")

    p_val = _placebo(sub, metric, "realized_pnl")
    print(f"  [3] Placebo: gerçek rho={rho:+.3f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{p_val*100:.1f}")

    med = sub["opened_at"].median()
    first, second = sub[sub["opened_at"] < med], sub[sub["opened_at"] >= med]
    r1, p1 = spearmanr(first[metric], first["realized_pnl"])
    r2, p2 = spearmanr(second[metric], second["realized_pnl"])
    print(f"  [4] Split-period: ilk yarı n={len(first)} rho={r1:+.3f}(p={p1:.3f}) | "
          f"ikinci yarı n={len(second)} rho={r2:+.3f}(p={p2:.3f})")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    df = _fetch(cur)
    real_trades = _fetch_real_trades(cur)
    conn.close()
    print(f"[fetch] {len(df)} kapalı sinyal, {len(real_trades)} gerçek paper_trade\n")

    print(f"{'='*70}\nSHARPE / SORTINO / INFORMATION RATIO (TÜM veri, formül değişmedi)\n{'='*70}")
    for metric in _METRICS:
        print(f"\n{'#'*70}\n{metric}\n{'#'*70}")
        for sig_type in ("Long", "Short"):
            _run_metric(df, real_trades, metric, sig_type)

    print(f"\n\n{'='*70}\nCALMAR_RATIO — TÜM veri (formül BUGÜN düzeltildi, eski satırlar kirli olabilir)\n{'='*70}")
    for sig_type in ("Long", "Short"):
        _run_metric(df, real_trades, "calmar_ratio", sig_type)

    df_fixed = df[df["opened_at"] > _FIX_CUTOFF]
    print(f"\n\n{'='*70}\nCALMAR_RATIO — SADECE düzeltme-sonrası ({_FIX_CUTOFF}, n={len(df_fixed)})\n{'='*70}")
    for sig_type in ("Long", "Short"):
        _run_metric(df_fixed, real_trades, "calmar_ratio", sig_type)


if __name__ == "__main__":
    main()
