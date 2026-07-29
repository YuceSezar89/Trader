"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki Kaufman
Efficiency Ratio'nun (ER) trade sonucunu öngörüp öngörmediği. (24 Tem 2026,
kullanıcı isteği — Hurst/Entropy/Amihud'un yanına, "fiyat A'dan B'ye ne kadar
verimli gitti" açısından.)

ER = |close[t] - close[t-N]| / Σ|close[i]-close[i-1]| (i=t-N+1..t), 0-1 arası.
Yüksek ER = verimli/düz trend, düşük ER = dolambaçlı/gürültülü yol (aynı net
mesafeyi kat etmek için çok daha fazla ileri-geri hareket).

Amihud'da öğrenilen ders: her bucket için AYRI subset-placebo testi şart —
sadece genel korelasyon placebo'su yanıltıcı olabilir (Q4'te görüldüğü gibi).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_kaufman_er_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı — cache kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _BASE_TH_LONG,
    _fetch_5m_full,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH

_LIVE_KOVALAMA_TH_LONG = 90
_WINDOW = 50


def _kaufman_er_at(close: np.ndarray, idx: int) -> float:
    if idx < _WINDOW:
        return float("nan")
    seg = close[idx - _WINDOW : idx + 1]
    net = abs(seg[-1] - seg[0])
    path = np.abs(np.diff(seg)).sum()
    if path <= 0:
        return float("nan")
    return float(net / path)


def _run_replay_with_er(qualifying: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Long")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

    results = []
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_full(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, _WINDOW) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            er = _kaufman_er_at(c, idx)
            if np.isnan(er):
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
                    "er": er,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def _subset_placebo(df: pd.DataFrame, sub: pd.DataFrame, iters: int = 1000) -> float:
    rng = np.random.default_rng(42)
    pool = df["pnl_usd"].to_numpy()
    real_mean = sub["pnl_usd"].mean()
    n = len(sub)
    ge = sum(1 for _ in range(iters) if rng.choice(pool, size=n, replace=False).mean() >= real_mean)
    return ge / iters * 100.0


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı Kaufman ER ile trade sonucu ilişkisi (gerçekçi replay)")
    print(f"(kovalama eşiği={_LIVE_KOVALAMA_TH_LONG}, pencere={_WINDOW})")
    print("=" * 78)

    long_df = pd.read_parquet(_HA_CACHE_PATH)
    long_df = long_df.dropna(
        subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"]
    )
    ta_base = (long_df["pct_1h"] >= _BASE_TH_LONG) & (long_df["pct_4h"] >= _BASE_TH_LONG)
    kovalama = (
        (long_df["pct_1h"] >= _LIVE_KOVALAMA_TH_LONG) & (long_df["slope_1h"] > 0)
    ) | ((long_df["pct_4h"] >= _LIVE_KOVALAMA_TH_LONG) & (long_df["slope_4h"] > 0))
    ha = (long_df["ha_bull_1h"] > 0.5) & (long_df["ha_bull_4h"] > 0.5)
    qualifying = long_df[ta_base & kovalama & ha][["symbol", "opened_at"]].reset_index(drop=True)

    df = _run_replay_with_er(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"ER dağılımı: min={df['er'].min():.3f} medyan={df['er'].median():.3f} max={df['er'].max():.3f}")

    print("\n-- Kartil bucket'ları (ER, düşük=dolambaçlı, yüksek=verimli/düz) --")
    df["q"] = pd.qcut(df["er"], 4, labels=["Q1(en dolambaçlı)", "Q2", "Q3", "Q4(en verimli)"])
    for q in ["Q1(en dolambaçlı)", "Q2", "Q3", "Q4(en verimli)"]:
        sub = df[df["q"] == q]
        rng_s = f"{sub['er'].min():.3f}-{sub['er'].max():.3f}"
        pb = _subset_placebo(df, sub)
        print(f"  {q:18s} ER={rng_s:14s} {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    print("\n-- Eşik sweep: ER >= X olan popülasyon (verimli taraf) --")
    for pct in [25, 50, 75]:
        th = df["er"].quantile(pct / 100)
        sub = df[df["er"] >= th]
        pb = _subset_placebo(df, sub)
        print(f"  >= p{pct} ({th:.3f}): n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    corr = np.corrcoef(df["er"], df["pnl_usd"])[0, 1]
    print(f"\nER vs pnl_usd korelasyonu: {corr:.3f}")

    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["er"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"genel korelasyon placebo: %{pct_ge:.1f}")

    for pct in [50, 75]:
        th = df["er"].quantile(pct / 100)
        sub = df[df["er"] >= th].sort_values("opened_at")
        if len(sub) < 20:
            continue
        mid = sub["opened_at"].iloc[len(sub) // 2]
        fh = _stats(sub[sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(sub[sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"ER>=p{pct} split-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
