"""
HAM RSI_Cross(9,24) Long sinyalleri (5m, HİÇBİR filtre YOK — all_up/TA-
percentile/kovalama/HA-hizalanma dahil değil) — giriş anındaki Shannon
Entropy'nin (pencere=50, kutu=5) trade sonucunu öngörüp öngörmediği.
(24 Tem 2026, kullanıcı isteği — ha_cross_raw_entropy_bt.py'nin RSI_Cross
eşleniği, aynı genelleme testi.)

Kullanım: python -m research.pattern_lab.rsi_cross_raw_entropy_bt
"""

import numpy as np

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.ha_cross_raw_entropy_bt import _run_replay_with_entropy
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import _stats
import pandas as pd

_MAX_SIGNALS = 4000


def _fetch_rsi_cross_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = 'Long' AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    return pd.DataFrame(cur.fetchall(), columns=["symbol", "opened_at", "open_price"])


def main() -> None:
    print("=" * 78)
    print("HAM RSI_Cross(9,24) Long (filtresiz) — giriş anı Shannon Entropy ile sonuç ilişkisi")
    print("=" * 78)

    conn = _conn()
    cur = conn.cursor()
    signals = _fetch_rsi_cross_signals(cur)
    conn.close()

    df = _run_replay_with_entropy(signals)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"entropy dağılımı: min={df['entropy'].min():.3f} medyan={df['entropy'].median():.3f} "
          f"max={df['entropy'].max():.3f}")

    print("\n-- Kartil bucket'ları (entropy) --")
    df["q"] = pd.qcut(df["entropy"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"])
    for q in ["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"]:
        sub = df[df["q"] == q]
        rng = f"{sub['entropy'].min():.3f}-{sub['entropy'].max():.3f}"
        print(f"  {q:14s} E={rng:16s} {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: entropy <= X olan popülasyon --")
    for th in [0.6, 0.7, 0.8, 0.9]:
        sub = df[df["entropy"] <= th]
        print(f"  <= {th:.1f}: n={len(sub):5d}  {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["entropy"], df["pnl_usd"])[0, 1]
    print(f"\nentropy vs pnl_usd korelasyonu: {corr:.3f}")

    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["entropy"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"placebo (karıştırmada eşit/büyük |rho| sıklığı): %{pct_ge:.1f}")

    d_sorted = df.sort_values("opened_at")
    mid = d_sorted["opened_at"].iloc[len(d_sorted) // 2]
    fh = _stats(d_sorted[d_sorted["opened_at"] < mid]["pnl_usd"].to_numpy())
    sh = _stats(d_sorted[d_sorted["opened_at"] >= mid]["pnl_usd"].to_numpy())
    print(f"\nsplit-period (tüm popülasyon): ilk yarı {fh} | ikinci yarı {sh}")

    for th in [0.6, 0.7, 0.8]:
        sub = df[df["entropy"] <= th].sort_values("opened_at")
        if len(sub) < 20:
            continue
        mid = sub["opened_at"].iloc[len(sub) // 2]
        fh = _stats(sub[sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(sub[sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"entropy<={th} split-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
