"""
Shannon Entropy — Short taraf paritey testi (24 Tem 2026). Long'daki güçlü
bulgunun (placebo %4.7, kartiller düzgün merdiven) Short'ta tutup tutmadığı
— Volume TA/Amihud'da yapılan aynı disiplin.

Short'ta HA-hizalanma kullanılmıyor (ta_kovalama_gate.py'de zaten HA'sız).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_entropy_short_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.entropy_clean_forward_return_bt import _entropy
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _SHORT_CACHE_PATH,
    _fetch_5m_full,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)

_BASE_TH_SHORT = 45.0
_EXTREME_TH_SHORT = 20.0
_WINDOW = 50
_BINS = 5


def _run_replay_with_entropy_short(qualifying: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Short")
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
        log_ret = np.diff(np.log(c))

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, _WINDOW) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            ent = _entropy(log_ret[idx - _WINDOW : idx], _BINS)
            if ent is None:
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
                    "entropy": ent,
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
    print("Short İKİLİ — giriş anı Shannon Entropy ile trade sonucu ilişkisi")
    print(f"(kovalama eşiği={_EXTREME_TH_SHORT}, pencere={_WINDOW}, kutu={_BINS}, HA'sız)")
    print("=" * 78)

    short_df = pd.read_parquet(_SHORT_CACHE_PATH)
    short_df = _add_all_up(short_df)
    short_df = short_df[short_df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"])
    ta_base_s = (short_df["pct_1h"] <= _BASE_TH_SHORT) & (short_df["pct_4h"] <= _BASE_TH_SHORT)
    kovalama_s = ((short_df["pct_1h"] <= _EXTREME_TH_SHORT) & (short_df["slope_1h"] < 0)) | (
        (short_df["pct_4h"] <= _EXTREME_TH_SHORT) & (short_df["slope_4h"] < 0)
    )
    qualifying = short_df[ta_base_s & kovalama_s][["symbol", "opened_at"]].reset_index(drop=True)

    df = _run_replay_with_entropy_short(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"entropy dağılımı: min={df['entropy'].min():.3f} medyan={df['entropy'].median():.3f} "
          f"max={df['entropy'].max():.3f}")

    print("\n-- Kartil bucket'ları (entropy) --")
    df["q"] = pd.qcut(df["entropy"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"], duplicates="drop")
    for q in df["q"].cat.categories:
        sub = df[df["q"] == q]
        rng_s = f"{sub['entropy'].min():.3f}-{sub['entropy'].max():.3f}"
        pb_le = _placebo_direction(df, sub, "le")
        print(f"  {str(q):14s} E={rng_s:16s} {_stats(sub['pnl_usd'].to_numpy())}  placebo(kötü-yön<=)=%{pb_le:.1f}")

    print("\n-- Eşik sweep: entropy <= X (düşük=iyi hipotezi) --")
    for th in [0.6, 0.7, 0.8, 0.9]:
        sub = df[df["entropy"] <= th]
        if len(sub) < 10:
            continue
        pb = _placebo_direction(df, sub, "ge")
        print(f"  <= {th:.1f}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  placebo(iyi-yön>=)=%{pb:.1f}")

    corr = np.corrcoef(df["entropy"], df["pnl_usd"])[0, 1]
    print(f"\nentropy vs pnl_usd korelasyonu: {corr:.3f}")
    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["entropy"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"genel korelasyon placebo: %{pct_ge:.1f}")

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
