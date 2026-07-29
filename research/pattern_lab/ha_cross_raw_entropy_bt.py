"""
HAM HA_Cross Long sinyalleri (5m, HİÇBİR filtre YOK — all_up/TA-percentile/
kovalama/HA-hizalanma dahil değil) — giriş anındaki Shannon Entropy'nin
(pencere=50, kutu=5) trade sonucunu öngörüp öngörmediği. (24 Tem 2026,
kullanıcı isteği — entropy bulgusunun rsi_cross_ta_triple_entropy_bt.py'deki
GİBİ heavily-filtered bir kombinasyona değil, HAM sinyale de genellenip
genellenmediğini görmek.)

Aynı gerçekçi replay motoru (SL=1.5xATR, TP=3xATR, trailing) ve aynı
entropy formülü (entropy_clean_forward_return_bt.py::_entropy) kullanılıyor.

Kullanım: python -m research.pattern_lab.ha_cross_raw_entropy_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.entropy_clean_forward_return_bt import _entropy
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _fetch_5m_full,
    _pnl_usd,
    _simulate,
    _stats,
)

_WINDOW = 50
_BINS = 5
_MAX_SIGNALS = 4000


def _fetch_ha_cross_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'HA_Cross' AND signal_type = 'Long' AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    return pd.DataFrame(cur.fetchall(), columns=["symbol", "opened_at", "open_price"])


def _run_replay_with_entropy(signals: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    print(f"ham HA_Cross Long örneklem n={len(signals)}")

    results = []
    symbols = signals["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_full(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()
        log_ret = np.diff(np.log(c))

        sub = signals[signals["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, _WINDOW) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            window_returns = log_ret[idx - _WINDOW : idx]
            ent = _entropy(window_returns, _BINS)
            if ent is None:
                continue

            entry_price = float(row["open_price"])
            atr_entry = float(atr[idx])
            exit_price, reason, bars_held = _simulate(
                h, l, c, idx + 1, "Long", entry_price, atr_entry
            )
            pnl = _pnl_usd("Long", entry_price, exit_price)
            results.append(
                {
                    "symbol": sym,
                    "opened_at": row["opened_at"],
                    "pnl_usd": pnl,
                    "reason": reason,
                    "entropy": ent,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("HAM HA_Cross Long (filtresiz) — giriş anı Shannon Entropy ile sonuç ilişkisi")
    print(f"(pencere={_WINDOW}, kutu={_BINS})")
    print("=" * 78)

    conn = _conn()
    cur = conn.cursor()
    signals = _fetch_ha_cross_signals(cur)
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
    real_corr = corr
    shuffled = [
        np.corrcoef(rng.permutation(df["entropy"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(real_corr)) * 100)
    print(f"placebo (karıştırmada eşit/büyük |rho| sıklığı): %{pct_ge:.1f}")

    d_sorted = df.sort_values("opened_at")
    mid = d_sorted["opened_at"].iloc[len(d_sorted) // 2]
    fh = _stats(d_sorted[d_sorted["opened_at"] < mid]["pnl_usd"].to_numpy())
    sh = _stats(d_sorted[d_sorted["opened_at"] >= mid]["pnl_usd"].to_numpy())
    print(f"\nsplit-period (tüm popülasyon): ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
