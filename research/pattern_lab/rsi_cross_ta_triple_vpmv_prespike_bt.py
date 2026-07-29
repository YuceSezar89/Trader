"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki VPMV
"sıçrama" ölçülerinin (vpmv_ratio, vpmv_slope) trade sonucunu öngörüp
öngörmediği. (25 Tem 2026, kullanıcı isteği — röntgen incelemesinde VPMV'nin
giriş öncesi sakin, SADECE tetikleyici barda sıçradığı bulundu; bu sıçramanın
BÜYÜKLÜĞÜ/öncesi-eğimi ayırt edici mi?)

vpmv_ratio = giriş barının VPMV'si / önceki 5 barın ortalaması (sıçrama oranı)
vpmv_slope = önceki 5 barın kendi içindeki eğimi (giriş ÖNCESİ zaten
             yükseliyor muydu, düşüyor muydu)

Bu ikisi `signals` tablosunda ZATEN her sinyalde hesaplanıp kayıtlı
(utils/vpmv.py::compute_pre, signal_engine.py'de sinyal üretilirken) — yeni
kline hesaplaması gerekmiyor, doğrudan JOIN.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_vpmv_prespike_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı — cache kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _BASE_TH_LONG,
    _fetch_5m_full,
    _pnl_usd,
    _simulate,
    _stats,
)
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH

_LIVE_KOVALAMA_TH_LONG = 90


def _fetch_signals_with_vpmv(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price, vpmv_ratio, vpmv_slope, vpmv_pre_avg
        FROM signals
        WHERE indicators IN ('RSI_Cross(9,24)','HA_Cross') AND interval='5m'
          AND signal_type='Long' AND open_price IS NOT NULL
          AND vpmv_ratio IS NOT NULL AND vpmv_slope IS NOT NULL
        """
    )
    rows = cur.fetchall()
    return pd.DataFrame(
        rows, columns=["symbol", "opened_at", "open_price", "vpmv_ratio", "vpmv_slope", "vpmv_pre_avg"]
    )


def _subset_placebo(df: pd.DataFrame, sub: pd.DataFrame, direction: str, iters: int = 1000) -> float:
    rng = np.random.default_rng(42)
    pool = df["pnl_usd"].to_numpy()
    real_mean = sub["pnl_usd"].mean()
    n = len(sub)
    if direction == "ge":
        cnt = sum(1 for _ in range(iters) if rng.choice(pool, size=n, replace=False).mean() >= real_mean)
    else:
        cnt = sum(1 for _ in range(iters) if rng.choice(pool, size=n, replace=False).mean() <= real_mean)
    return cnt / iters * 100.0


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı VPMV sıçrama ölçüleri ile trade sonucu ilişkisi")
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

    conn = _conn()
    cur = conn.cursor()
    sig_df = _fetch_signals_with_vpmv(cur)
    merged = qualifying.merge(sig_df, on=["symbol", "opened_at"], how="inner")
    print(f"VPMV verisiyle eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

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
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Long", entry_price, atr_entry)
            pnl = _pnl_usd("Long", entry_price, exit_price)
            results.append(
                {
                    "symbol": sym,
                    "opened_at": row["opened_at"],
                    "pnl_usd": pnl,
                    "reason": reason,
                    "vpmv_ratio": row["vpmv_ratio"],
                    "vpmv_slope": row["vpmv_slope"],
                    "vpmv_pre_avg": row["vpmv_pre_avg"],
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    df = pd.DataFrame(results)
    if df.empty:
        print("Sonuç yok.")
        return
    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")

    for col in ["vpmv_ratio", "vpmv_slope", "vpmv_pre_avg"]:
        vals = df[col].to_numpy()
        corr = np.corrcoef(vals, df["pnl_usd"])[0, 1]
        rng = np.random.default_rng(42)
        shuffled = [np.corrcoef(rng.permutation(vals), df["pnl_usd"].to_numpy())[0, 1] for _ in range(300)]
        pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
        print(f"\n{'='*70}\n{col} — korelasyon={corr:.3f}  genel-placebo=%{pct_ge:.1f}")
        print(f"  dağılım: min={vals.min():.2f} medyan={np.median(vals):.2f} max={vals.max():.2f}")

        df["q"] = pd.qcut(df[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        for q in df["q"].cat.categories:
            sub = df[df["q"] == q]
            rng_s = f"{sub[col].min():.2f}-{sub[col].max():.2f}"
            pb_ge = _subset_placebo(df, sub, "ge")
            pb_le = _subset_placebo(df, sub, "le")
            print(f"    {str(q)}: {rng_s:16s} {_stats(sub['pnl_usd'].to_numpy())} "
                  f"placebo(iyi>=)=%{pb_ge:.1f} placebo(kötü<=)=%{pb_le:.1f}")


if __name__ == "__main__":
    main()
