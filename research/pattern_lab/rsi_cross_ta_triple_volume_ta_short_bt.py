"""
"Volume TA" — Short taraf eşleniği (24 Tem 2026). Long'daki bulgunun
(pct_volta düşükse fiyat "kovalama" desin bile işlem güvenilir değil,
placebo %0.6-2.6) Short'ta simetrik/tersi mi çıktığı test ediliyor.

Short'ta HA-hizalanma kullanılmıyor (ta_kovalama_gate.py'de zaten HA'sız —
[[project_ha_cross_short_24tem]]), popülasyon sadece all_up+TA-base+kovalama.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_volume_ta_short_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _SHORT_CACHE_PATH,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)
from research.pattern_lab.rsi_cross_ta_triple_volume_ta_bt import (
    _fetch_5m_with_dirvol,
    _net_volume_ta_series,
    _subset_placebo,
)
from signals.ta_kovalama_gate import _percentile_now

_BASE_TH_SHORT = 45.0
_EXTREME_TH_SHORT = 20.0  # signals/ta_kovalama_gate.py::_SHORT_KOVALAMA_TH


def _run_replay_with_volta_short(qualifying: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Short")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

    results = []
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_with_dirvol(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()
        vol_ta = _net_volume_ta_series(
            b, df5["volume"].to_numpy(), df5["buy_volume"].to_numpy(), df5["sell_volume"].to_numpy()
        )

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, 50) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            pct_volta = _percentile_now(vol_ta[: idx + 1])
            if pct_volta != pct_volta:
                continue

            entry_price = float(row["open_price"])
            atr_entry = float(atr[idx])
            exit_price, reason, bars_held = _simulate(
                h, l, c, idx + 1, "Short", entry_price, atr_entry
            )
            pnl = _pnl_usd("Short", entry_price, exit_price)
            results.append(
                {
                    "symbol": sym,
                    "opened_at": row["opened_at"],
                    "pnl_usd": pnl,
                    "reason": reason,
                    "pct_volta": pct_volta,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def _placebo_direction(df: pd.DataFrame, sub: pd.DataFrame, direction: str, iters: int = 1000) -> float:
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
    print("Short İKİLİ — giriş anı Volume TA percentile ile trade sonucu ilişkisi")
    print(f"(kovalama eşiği={_EXTREME_TH_SHORT}, HA'sız)")
    print("=" * 78)

    short_df = pd.read_parquet(_SHORT_CACHE_PATH)
    short_df = _add_all_up(short_df)
    short_df = short_df[short_df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"])
    ta_base_s = (short_df["pct_1h"] <= _BASE_TH_SHORT) & (short_df["pct_4h"] <= _BASE_TH_SHORT)
    kovalama_s = ((short_df["pct_1h"] <= _EXTREME_TH_SHORT) & (short_df["slope_1h"] < 0)) | (
        (short_df["pct_4h"] <= _EXTREME_TH_SHORT) & (short_df["slope_4h"] < 0)
    )
    qualifying = short_df[ta_base_s & kovalama_s][["symbol", "opened_at"]].reset_index(drop=True)

    df = _run_replay_with_volta_short(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"pct_volta dağılımı: min={df['pct_volta'].min():.1f} medyan={df['pct_volta'].median():.1f} "
          f"max={df['pct_volta'].max():.1f}")

    print("\n-- Kartil bucket'ları (volume TA percentile) --")
    df["q"] = pd.qcut(df["pct_volta"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"], duplicates="drop")
    for q in df["q"].cat.categories:
        sub = df[df["q"] == q]
        rng_s = f"{sub['pct_volta'].min():.1f}-{sub['pct_volta'].max():.1f}"
        pb_ge = _placebo_direction(df, sub, "ge")
        pb_le = _placebo_direction(df, sub, "le")
        print(f"  {str(q):14s} pct={rng_s:12s} {_stats(sub['pnl_usd'].to_numpy())}  "
              f"placebo(iyi-yön>=)=%{pb_ge:.1f}  placebo(kötü-yön<=)=%{pb_le:.1f}")

    print("\n-- Eşik sweep: pct_volta >= X --")
    for th in [50, 60, 70, 80, 90]:
        sub = df[df["pct_volta"] >= th]
        if len(sub) < 10:
            continue
        pb = _placebo_direction(df, sub, "ge")
        print(f"  >= {th}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  placebo=%{pb:.1f}")

    print("\n-- Eşik sweep: pct_volta <= X --")
    for th in [10, 20, 30, 40, 50]:
        sub = df[df["pct_volta"] <= th]
        if len(sub) < 10:
            continue
        pb = _placebo_direction(df, sub, "le")
        print(f"  <= {th}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  placebo=%{pb:.1f}")

    corr = np.corrcoef(df["pct_volta"], df["pnl_usd"])[0, 1]
    print(f"\npct_volta vs pnl_usd korelasyonu: {corr:.3f}")
    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["pct_volta"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"genel korelasyon placebo: %{pct_ge:.1f}")


if __name__ == "__main__":
    main()
