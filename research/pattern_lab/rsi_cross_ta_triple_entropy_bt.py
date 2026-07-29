"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki Shannon
Entropy'nin (pencere=50, kutu=5 — entropy_clean_forward_return_bt.py'de EN
İYİ çıkan kombinasyon, orada rho=+0.032/p=0.0137/placebo=%1 ile GERÇEK
bulunmuştu) trade sonucunu öngörüp öngörmediği. (24 Tem 2026, kullanıcı
isteği — Hurst'te yaptığımız aynı "koşullu test" mantığı: ham/evrensel
korelasyon yerine, zaten yönlü bir sinyalin (kovalama) üstüne ekleyince
etki güçleniyor mu.)

Entropy formülü entropy_clean_forward_return_bt.py::_entropy ile birebir:
log-return'leri eşit-genişlik kutulara ayır, H=-Σ(p×log2 p)/log2(kutu).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_entropy_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı — cache kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.entropy_clean_forward_return_bt import _entropy
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
_WINDOW = 50  # entropy_clean_forward_return_bt.py'de en umut verici kombinasyon
_BINS = 5


def _run_replay_with_entropy(qualifying: pd.DataFrame) -> pd.DataFrame:
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
        log_ret = np.diff(np.log(c))  # log_ret[i] = c[i+1] barının getirisi (c[i]->c[i+1])

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < 14 or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            # entropy_clean_forward_return_bt._fetch_closes_before ile eşdeğer:
            # giriş barından ÖNCEKİ _WINDOW+1 close'un log-return'leri (son _WINDOW tanesi)
            if idx < _WINDOW:
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
                    "bars_held": bars_held,
                    "entropy": ent,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı Shannon Entropy ile trade sonucu ilişkisi (gerçekçi replay)")
    print(f"(kovalama eşiği={_LIVE_KOVALAMA_TH_LONG}, pencere={_WINDOW}, kutu={_BINS})")
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

    df = _run_replay_with_entropy(qualifying)
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

    print("\n-- Eşik sweep: entropy <= X olan popülasyon (düşük entropi = sıkışık/yönlü) --")
    for th in [0.6, 0.7, 0.8, 0.9]:
        sub = df[df["entropy"] <= th]
        print(f"  <= {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: entropy >= X olan popülasyon (yüksek entropi = dağınık/rastgele) --")
    for th in [0.6, 0.7, 0.8, 0.9]:
        sub = df[df["entropy"] >= th]
        print(f"  >= {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["entropy"], df["pnl_usd"])[0, 1]
    print(f"\nentropy vs pnl_usd korelasyonu: {corr:.3f}")


if __name__ == "__main__":
    main()
