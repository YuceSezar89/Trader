"""
Z-score üst sınırı testi (BEKLEYEN_ANALIZLER'de eski madde, eşik 20 Tem 2026'da
aşıldı: 1845 |z_score_entry|>3 sinyal, 50+ isteniyordu).

Soru: z_score_entry = (close-EMA200)/StdDev200 (Ayrışma panelinin ölçtüğü
şey, sinyal açılış anındaki dondurulmuş hali) çok EKSTREM (|Z|>3) olduğunda
sinyal kalitesi değişiyor mu — "çok gerilmiş, dönüşe yakın" (kötü) mü yoksa
"çok güçlü trend" (iyi) mü?

4 kapı: bant analizi (|Z| dörtte bir: <1/1-2/2-3/>3), korelasyon, gerçek $,
placebo, split-period. Yön-ayarlı DEĞİL bilerek — z_score_entry ham/yönsüz
bir konum ölçüsü (Long/Short ayrı test edilir, ama işaretin kendisi ham).

Kullanım: python -m research.pattern_lab.zscore_extreme_band_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_BAND_EDGES = [0, 1, 2, 3, np.inf]
_BAND_LABELS = ["<1", "1-2", "2-3", ">3"]
_PLACEBO_ITER = 300


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, signal_type, opened_at, z_score_entry, realized_pnl
        FROM signals
        WHERE status = 'closed' AND realized_pnl IS NOT NULL AND z_score_entry IS NOT NULL
        """
    )
    return pd.DataFrame(cur.fetchall(), columns=["id", "symbol", "signal_type", "opened_at", "z_score_entry", "realized_pnl"])


def _fetch_real_trades(cur) -> pd.DataFrame:
    cur.execute(
        "SELECT signal_id, pnl_usd FROM paper_trades WHERE status='closed' AND signal_id IS NOT NULL"
    )
    return pd.DataFrame(cur.fetchall(), columns=["signal_id", "pnl_usd"])


def _band_report(sub: pd.DataFrame, target: str, label: str) -> None:
    abs_z = sub["z_score_entry"].abs()
    band = pd.cut(abs_z, bins=_BAND_EDGES, labels=_BAND_LABELS, right=False)
    g = sub.groupby(band, observed=True)[target].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(f"    [{label}] |Z| bant analizi:")
    print(g.to_string().replace("\n", "\n      "))


def _placebo(sub: pd.DataFrame, target: str, n_iter: int = _PLACEBO_ITER) -> float:
    real_rho, _ = spearmanr(sub["z_score_entry"].abs(), sub[target])
    rng = np.random.default_rng(42)
    vals = sub["z_score_entry"].abs().to_numpy()
    t = sub[target].to_numpy()
    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(vals)
        rho, _ = spearmanr(shuffled, t)
        if abs(rho) >= abs(real_rho):
            count_ge += 1
    return count_ge / n_iter


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    df = _fetch_signals(cur)
    real_trades = _fetch_real_trades(cur)
    conn.close()
    print(f"[fetch] {len(df)} kapalı sinyal (z_score_entry dolu), {len(real_trades)} gerçek paper_trade\n")

    for sig_type in ("Long", "Short"):
        sub = df[df["signal_type"] == sig_type]
        print(f"{'='*70}\n{sig_type} (n={len(sub)})\n{'='*70}")

        rho, p = spearmanr(sub["z_score_entry"].abs(), sub["realized_pnl"])
        print(f"\n  [1] Korelasyon (|Z| vs realized_pnl): rho={rho:+.3f} (p={p:.4f})")
        _band_report(sub, "realized_pnl", "signals.realized_pnl")

        merged = sub.merge(real_trades, left_on="id", right_on="signal_id", how="inner")
        print(f"\n  [2] Gerçek $ (n={len(merged)})")
        if len(merged) >= 100:
            rho_r, p_r = spearmanr(merged["z_score_entry"].abs(), merged["pnl_usd"])
            print(f"    rho={rho_r:+.3f} (p={p_r:.4f})")
            _band_report(merged, "pnl_usd", "paper_trades.pnl_usd")
        else:
            print(f"    yetersiz örnek (n={len(merged)})")

        p_val = _placebo(sub, "realized_pnl")
        print(f"\n  [3] Placebo: gerçek rho={rho:+.3f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{p_val*100:.1f}")

        med = sub["opened_at"].median()
        first, second = sub[sub["opened_at"] < med], sub[sub["opened_at"] >= med]
        r1, p1 = spearmanr(first["z_score_entry"].abs(), first["realized_pnl"])
        r2, p2 = spearmanr(second["z_score_entry"].abs(), second["realized_pnl"])
        print(f"\n  [4] Split-period: ilk yarı n={len(first)} rho={r1:+.3f}(p={p1:.3f}) | "
              f"ikinci yarı n={len(second)} rho={r2:+.3f}(p={p2:.3f})")
        print()


if __name__ == "__main__":
    main()
