"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki Amihud
Illiquidity ölçüsünün trade sonucunu öngörüp öngörmediği. (24 Tem 2026,
kullanıcı isteği — Hurst/Entropy'nin yanına, likidite-mikroyapı açısından.)

Amihud ILLIQ_t = |getiri_t| / dolar_hacim_t (× 1e6, okunabilirlik için) —
akademik literatürdeki standart formül, günlük yerine 5m bar'da uygulanıyor.
Trailing pencerede (50 bar, entropy testiyle aynı disiplin) ortalaması alınıp
giriş anındaki değeri kaydediliyor. YÜKSEK ILLIQ = düşük likidite (fiyat
dolar-hacme göre çok oynuyor) — bugünkü adli incelemede bulduğumuz "düşük
likiditeli altcoinlerde SL gürültü seviyesine iniyor" sorununu doğrudan
ölçen bir aday.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_amihud_bt
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
from indicators.core import calculate_atr

_LIVE_KOVALAMA_TH_LONG = 90  # signals/ta_kovalama_gate.py::_LONG_KOVALAMA_TH
_WINDOW = 50  # entropy testiyle aynı pencere, karşılaştırılabilir olsun


def _fetch_5m_with_volume(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, open, high, low, close, volume FROM cagg_5m WHERE symbol=%s ORDER BY bucket ASC",
        (symbol,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["atr"] = calculate_atr(df)
    return df


def _amihud_at(close: np.ndarray, volume: np.ndarray, idx: int) -> float:
    if idx < _WINDOW:
        return float("nan")
    seg_c = close[idx - _WINDOW : idx + 1]
    seg_v = volume[idx - _WINDOW : idx]
    ret = np.abs(np.diff(seg_c) / seg_c[:-1])
    dollar_vol = seg_v * seg_c[:-1]
    valid = dollar_vol > 0
    if valid.sum() < _WINDOW * 0.5:
        return float("nan")
    illiq = ret[valid] / dollar_vol[valid]
    return float(illiq.mean() * 1e6)


def _run_replay_with_amihud(qualifying: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Long")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

    results = []
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_with_volume(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        v = df5["volume"].to_numpy()
        atr = df5["atr"].to_numpy()

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, _WINDOW) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            illiq = _amihud_at(c, v, idx)
            if np.isnan(illiq):
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
                    "amihud": illiq,
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı Amihud Illiquidity ile trade sonucu ilişkisi (gerçekçi replay)")
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

    df = _run_replay_with_amihud(qualifying)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")
    print(f"amihud dağılımı: min={df['amihud'].min():.4f} medyan={df['amihud'].median():.4f} "
          f"max={df['amihud'].max():.4f}")

    print("\n-- Kartil bucket'ları (amihud, düşük=likit, yüksek=illikit) --")
    df["q"] = pd.qcut(df["amihud"], 4, labels=["Q1(en likit)", "Q2", "Q3", "Q4(en illikit)"])
    for q in ["Q1(en likit)", "Q2", "Q3", "Q4(en illikit)"]:
        sub = df[df["q"] == q]
        rng = f"{sub['amihud'].min():.4f}-{sub['amihud'].max():.4f}"
        print(f"  {q:16s} A={rng:22s} {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik sweep: amihud <= X olan popülasyon (likit taraf) --")
    for pct in [25, 50, 75, 90]:
        th = df["amihud"].quantile(pct / 100)
        sub = df[df["amihud"] <= th]
        print(f"  <= p{pct} ({th:.4f}): n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["amihud"], df["pnl_usd"])[0, 1]
    print(f"\namihud vs pnl_usd korelasyonu: {corr:.3f}")

    rng = np.random.default_rng(42)
    shuffled = [
        np.corrcoef(rng.permutation(df["amihud"].to_numpy()), df["pnl_usd"].to_numpy())[0, 1]
        for _ in range(300)
    ]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"placebo (karıştırmada eşit/büyük |rho| sıklığı): %{pct_ge:.1f}")

    for pct in [25, 50]:
        th = df["amihud"].quantile(pct / 100)
        sub = df[df["amihud"] <= th].sort_values("opened_at")
        if len(sub) < 20:
            continue
        mid = sub["opened_at"].iloc[len(sub) // 2]
        fh = _stats(sub[sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(sub[sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"amihud<=p{pct} split-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
