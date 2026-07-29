"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki Hurst
exponent'i (Ehlers fraktal boyut yöntemi, ~/Desktop/pine/hurst_ehlers_fractal.pine
ile BİREBİR aynı formül, N=16, 5m bar) trade sonucunu öngörüyor mu? (24 Tem
2026, kullanıcı isteği — ATR%/BTC-trend testleriyle aynı disiplin.)

D = (log(N1+N2) - log(N3)) / log(2), H = 2 - D, giriş barının 5m serisinde
hesaplanıyor (kovalamanın "yerel" fiyat yapısı — trend mi, range mi).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_hurst_bt
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

_LIVE_KOVALAMA_TH_LONG = 90  # signals/ta_kovalama_gate.py::_LONG_KOVALAMA_TH
_N = 16  # hurst_ehlers_fractal.pine varsayılan pencere
_N2 = _N // 2


def _hurst_at(high: np.ndarray, low: np.ndarray, idx: int) -> float:
    """idx barındaki Hurst exponent — Ehlers fraktal boyut, pine dosyasıyla birebir."""
    if idx < _N - 1:
        return float("nan")
    n1 = (high[idx - _N2 + 1 : idx + 1].max() - low[idx - _N2 + 1 : idx + 1].min()) / _N2
    n2 = (
        high[idx - _N + 1 : idx - _N2 + 1].max() - low[idx - _N + 1 : idx - _N2 + 1].min()
    ) / _N2
    n3 = (high[idx - _N + 1 : idx + 1].max() - low[idx - _N + 1 : idx + 1].min()) / _N
    if n1 + n2 <= 0 or n3 <= 0:
        return float("nan")
    d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    d = min(2.0, max(1.0, d))
    return 2.0 - d


def _run_replay_with_hurst(qualifying: pd.DataFrame) -> pd.DataFrame:
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
            if idx < max(14, _N - 1) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            hurst = _hurst_at(h, l, idx)
            if np.isnan(hurst):
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
                    "bars_held": bars_held,
                    "hurst": hurst,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı Hurst exponent'i ile trade sonucu ilişkisi (gerçekçi replay)")
    print(f"(kovalama eşiği={_LIVE_KOVALAMA_TH_LONG}, N={_N}, canlı gate ile birebir popülasyon)")
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

    df = _run_replay_with_hurst(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"hurst dağılımı: min={df['hurst'].min():.3f} medyan={df['hurst'].median():.3f} "
          f"max={df['hurst'].max():.3f}")

    print("\n-- Kartil bucket'ları (hurst) --")
    df["q"] = pd.qcut(df["hurst"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"])
    for q in ["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"]:
        sub = df[df["q"] == q]
        rng = f"{sub['hurst'].min():.3f}-{sub['hurst'].max():.3f}"
        print(f"  {q:14s} H={rng:16s} {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: hurst >= X olan popülasyon --")
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sub = df[df["hurst"] >= th]
        print(f"  >= {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: hurst < X olan popülasyon --")
    for th in [0.4, 0.5, 0.6]:
        sub = df[df["hurst"] < th]
        print(f"  <  {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["hurst"], df["pnl_usd"])[0, 1]
    print(f"\nhurst vs pnl_usd korelasyonu: {corr:.3f}")


if __name__ == "__main__":
    main()
