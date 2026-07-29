"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki
Choppiness Index'in (CI) trade sonucunu öngörüp öngörmediği. (24 Tem 2026,
kullanıcı isteği — Kaufman ER'in klasik TA karşılığı, True Range tabanlı.)

CI = 100 * log10( Σ TrueRange(N) / (max(High,N) - min(Low,N)) ) / log10(N)
0-100 arası. YÜKSEK CI = dolambaçlı/yatay (çok hareket ama net mesafe küçük),
DÜŞÜK CI = verimli/düz trend (True Range toplamı net menzile yakın).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_choppiness_bt
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


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))


def _choppiness_at(h: np.ndarray, l: np.ndarray, tr: np.ndarray, idx: int) -> float:
    if idx < _WINDOW:
        return float("nan")
    seg_tr = tr[idx - _WINDOW + 1 : idx + 1]
    seg_h = h[idx - _WINDOW + 1 : idx + 1]
    seg_l = l[idx - _WINDOW + 1 : idx + 1]
    rng = seg_h.max() - seg_l.min()
    if rng <= 0:
        return float("nan")
    ratio = seg_tr.sum() / rng
    if ratio <= 0:
        return float("nan")
    return float(100.0 * np.log10(ratio) / np.log10(_WINDOW))


def _run_replay_with_ci(qualifying: pd.DataFrame) -> pd.DataFrame:
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
        tr = _true_range(h, l, c)

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, _WINDOW) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            ci = _choppiness_at(h, l, tr, idx)
            if np.isnan(ci):
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
                    "ci": ci,
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
    print("Long ÜÇLÜ — giriş anı Choppiness Index ile trade sonucu ilişkisi (gerçekçi replay)")
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

    df = _run_replay_with_ci(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"CI dağılımı: min={df['ci'].min():.1f} medyan={df['ci'].median():.1f} max={df['ci'].max():.1f}")

    print("\n-- Kartil bucket'ları (CI, düşük=verimli/düz, yüksek=dolambaçlı) --")
    df["q"] = pd.qcut(df["ci"], 4, labels=["Q1(en verimli)", "Q2", "Q3", "Q4(en dolambaçlı)"])
    for q in ["Q1(en verimli)", "Q2", "Q3", "Q4(en dolambaçlı)"]:
        sub = df[df["q"] == q]
        rng_s = f"{sub['ci'].min():.1f}-{sub['ci'].max():.1f}"
        pb = _subset_placebo(df, sub)
        print(f"  {q:18s} CI={rng_s:12s} {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    print("\n-- Eşik sweep: CI <= X olan popülasyon (verimli taraf) --")
    for pct in [25, 50, 75]:
        th = df["ci"].quantile(pct / 100)
        sub = df[df["ci"] <= th]
        pb = _subset_placebo(df, sub)
        print(f"  <= p{pct} ({th:.1f}): n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    corr = np.corrcoef(df["ci"], df["pnl_usd"])[0, 1]
    print(f"\nCI vs pnl_usd korelasyonu: {corr:.3f}")

    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["ci"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"genel korelasyon placebo: %{pct_ge:.1f}")

    for pct in [25, 50]:
        th = df["ci"].quantile(pct / 100)
        sub = df[df["ci"] <= th].sort_values("opened_at")
        if len(sub) < 20:
            continue
        mid = sub["opened_at"].iloc[len(sub) // 2]
        fh = _stats(sub[sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(sub[sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"CI<=p{pct} split-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
