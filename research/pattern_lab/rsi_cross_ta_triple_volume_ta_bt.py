"""
"Volume TA" — signals/ta_kovalama_gate.py::_net_ta_series'in BİREBİR
tarifini (12 saatte bir UTC 00:00/12:00'de sıfırlanan kümülatif toplam,
son 200 bara göre percentile) fiyat yerine GERÇEK yönlü hacme uygular.
(24 Tem 2026, kullanıcı fikri.)

net_ta:     Σ (close[i]-close[i-1])/close[i-1]*100     (bar-bar % fiyat değişimi)
volume_ta:  Σ (buy_volume[i]-sell_volume[i])/volume[i]*100   (bar'ın net-alıcı
            payı, yüzde puanı — fiyat % değişimiyle AYNI ölçekte, doğrudan
            karşılaştırılabilir)

signals/market_context.py::compute_cvd_slope'tan farkı: o sürekli kümülatif +
10-bar eğim; bu 12h döngü-sıfırlamalı + percentile (net_ta'nın tam disiplini).

Test: ta_kovalama Long kombosu (n=266) üzerinde volume_ta percentile'ının
trade sonucuyla ilişkisi — kartil, eşik, subset-placebo, split-period.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_volume_ta_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı — cache kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _BASE_TH_LONG,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH
from signals.ta_kovalama_gate import _percentile_now
from indicators.core import calculate_atr

_LIVE_KOVALAMA_TH_LONG = 90
_CYCLE_MS = 12 * 3600 * 1000
_TA_LOOKBACK = 200
_TA_MIN_LOOKBACK = 50


def _fetch_5m_with_dirvol(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT bucket, open, high, low, close, volume, buy_volume, sell_volume
        FROM cagg_5m WHERE symbol=%s ORDER BY bucket ASC
        """,
        (symbol,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(
        rows, columns=["bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    )
    for c in ("open", "high", "low", "close", "volume", "buy_volume", "sell_volume"):
        df[c] = df[c].astype(float)
    df["atr"] = calculate_atr(df)
    return df


def _net_volume_ta_series(bucket: np.ndarray, volume: np.ndarray, buy_v: np.ndarray, sell_v: np.ndarray) -> np.ndarray:
    """net_ta ile BİREBİR aynı iskelet — sadece bar-bar bileşen fiyat yerine
    net-alıcı yüzdesi (buy-sell)/toplam*100."""
    t_ms = bucket.astype("datetime64[ms]").astype("int64")
    cycle = (t_ms // _CYCLE_MS) * _CYCLE_MS
    n = len(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_imbalance = np.where(volume > 0, (buy_v - sell_v) / volume * 100.0, 0.0)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            net[i] = net[i - 1] + bar_imbalance[i]
    return net


def _run_replay_with_volta(qualifying: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Long")
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
            if idx < max(14, _TA_MIN_LOOKBACK) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            pct_volta = _percentile_now(vol_ta[: idx + 1])
            if pct_volta != pct_volta:  # NaN kontrolü
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
                    "pct_volta": pct_volta,
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
    print("Long ÜÇLÜ — giriş anı Volume TA percentile ile trade sonucu ilişkisi")
    print(f"(kovalama eşiği={_LIVE_KOVALAMA_TH_LONG})")
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

    df = _run_replay_with_volta(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"pct_volta dağılımı: min={df['pct_volta'].min():.1f} medyan={df['pct_volta'].median():.1f} "
          f"max={df['pct_volta'].max():.1f}")

    print("\n-- Kartil bucket'ları (volume TA percentile) --")
    df["q"] = pd.qcut(df["pct_volta"], 4, labels=["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"])
    for q in ["Q1(en düşük)", "Q2", "Q3", "Q4(en yüksek)"]:
        sub = df[df["q"] == q]
        rng_s = f"{sub['pct_volta'].min():.1f}-{sub['pct_volta'].max():.1f}"
        pb = _subset_placebo(df, sub)
        print(f"  {q:14s} pct={rng_s:12s} {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    print("\n-- Eşik sweep: pct_volta >= X olan popülasyon --")
    for th in [50, 60, 70, 80, 90]:
        sub = df[df["pct_volta"] >= th]
        if len(sub) < 10:
            continue
        pb = _subset_placebo(df, sub)
        print(f"  >= {th}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    print("\n-- Eşik sweep: pct_volta <= X olan popülasyon --")
    for th in [10, 20, 30, 40, 50]:
        sub = df[df["pct_volta"] <= th]
        if len(sub) < 10:
            continue
        pb = _subset_placebo(df, sub)
        print(f"  <= {th}: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

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
