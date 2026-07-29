"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki ATR%
(atr/entry_price*100) trade sonucunu öngörüyor mu? (24 Tem 2026, canlıda
ta_kovalama_live'ın ilk 12 saatlik Long serisinde 10/12 işlemin SL mesafesi
<%1 idi ve çoğu 5-30dk içinde stop oldu — bkz. memory
project_rsi_cross_ta_triple_combo_24tem — bu şüphenin gerçekçi replay
üzerinde testi.)

rsi_cross_combo_realistic_replay_bt.py ile AYNI simülasyon motoru
(SL=1.5xATR, TP=3xATR, trailing sadece TP'de aktif) — tek fark her işlem
için atr_pct = atr_entry/entry_price*100 kaydediliyor ve sonuçlar bu
değere göre bucket'lanıyor.

Kovalama eşiği canlı gate'le (signals/ta_kovalama_gate.py) birebir: 90
(realistic_replay_bt.py'deki 80 DEĞİL — o script threshold-sweep'ten
ÖNCE yazılmıştı, güncellenmedi).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_atrpct_bt
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


def _run_replay_with_atrpct(qualifying: pd.DataFrame) -> pd.DataFrame:
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
            if idx < 14 or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
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
                    "atr_pct": atr_entry / entry_price * 100.0,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş ATR% ile trade sonucu ilişkisi (gerçekçi replay)")
    print(f"(kovalama eşiği={_LIVE_KOVALAMA_TH_LONG}, canlı gate ile birebir)")
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

    df = _run_replay_with_atrpct(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"atr_pct dağılımı: min={df['atr_pct'].min():.3f} medyan={df['atr_pct'].median():.3f} "
          f"max={df['atr_pct'].max():.3f}")

    print("\n-- Kartil bucket'ları (atr_pct) --")
    df["q"] = pd.qcut(df["atr_pct"], 4, labels=["Q1(en dar)", "Q2", "Q3", "Q4(en geniş)"])
    for q in ["Q1(en dar)", "Q2", "Q3", "Q4(en geniş)"]:
        sub = df[df["q"] == q]
        rng = f"{sub['atr_pct'].min():.3f}-{sub['atr_pct'].max():.3f}%"
        print(f"  {q:14s} atr_pct={rng:16s} {_stats(sub['pnl_usd'].to_numpy())} "
              f"reason={sub['reason'].value_counts().to_dict()}")

    print("\n-- Eşik sweep: atr_pct >= X olan popülasyon --")
    for th in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        sub = df[df["atr_pct"] >= th]
        print(f"  >= {th:>4.1f}%: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: atr_pct < X olan popülasyon (dışlanacak taraf) --")
    for th in [0.3, 0.5, 0.8, 1.0, 1.5]:
        sub = df[df["atr_pct"] < th]
        print(f"  <  {th:>4.1f}%: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    # IS/OOS: en iyi görünen eşiğin split-period tutarlılığı
    best_sub = df[df["atr_pct"] >= 0.8].sort_values("opened_at")
    if len(best_sub) >= 40:
        mid = best_sub["opened_at"].iloc[len(best_sub) // 2]
        fh = _stats(best_sub[best_sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(best_sub[best_sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"\n-- atr_pct>=0.8 split-period: ilk yarı {fh} | ikinci yarı {sh} --")
    else:
        print(f"\n-- atr_pct>=0.8: n={len(best_sub)} split-period için yetersiz (<40) --")


if __name__ == "__main__":
    main()
