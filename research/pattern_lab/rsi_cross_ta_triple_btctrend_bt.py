"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — giriş anındaki BTC
trend'i (z-score EMA200/std200, signals/signal_processor.py::btc_trend ile
BİREBİR aynı formül) trade sonucunu öngörüyor mu? (24 Tem 2026, kullanıcı
gözlemi: canlıda ta_kovalama_live Long serisi kötüye gidiyor, "piyasanın
düşüş yönlü olmasıyla bağlantılı olabilir mi" — canlı örneklemde bearish/
neutral BTC rejiminde WR %8.3/%0, bullish'te %23.8 görüldü, bu ilişkinin
gerçekçi replay'de de var olup olmadığının testi.)

BTC z-score formülü signal_processor.py:711-728 ile birebir:
  z = (close - EMA200) / (rolling_std200 + 1e-12), 5m bar bazında
  bullish: z>0.5, bearish: z<-0.5, neutral: aradaki

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_btctrend_bt
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


def _fetch_btc_5m(cur) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, close FROM cagg_5m WHERE symbol='BTCUSDT' ORDER BY bucket ASC"
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "close"])
    df["close"] = df["close"].astype(float)
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["std200"] = df["close"].rolling(200).std()
    df["z"] = (df["close"] - df["ema200"]) / (df["std200"] + 1e-12)
    df["trend"] = np.where(df["z"] > 0.5, "bullish", np.where(df["z"] < -0.5, "bearish", "neutral"))
    return df


def _run_replay_with_btctrend(qualifying: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Long")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

    b_btc = btc_df["bucket"].to_numpy()
    z_btc = btc_df["z"].to_numpy()
    trend_btc = btc_df["trend"].to_numpy()

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

            btc_idx = np.searchsorted(b_btc, np.datetime64(row["opened_at"]), side="right") - 1
            if btc_idx < 200 or btc_idx >= len(z_btc) or np.isnan(z_btc[btc_idx]):
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
                    "btc_z": float(z_btc[btc_idx]),
                    "btc_trend": str(trend_btc[btc_idx]),
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("Long ÜÇLÜ — giriş anı BTC trend'i ile trade sonucu ilişkisi (gerçekçi replay)")
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

    conn = _conn()
    cur = conn.cursor()
    btc_df = _fetch_btc_5m(cur)
    conn.close()
    print(f"BTC 5m bar sayısı: {len(btc_df)}, trend dağılımı: {btc_df['trend'].value_counts().to_dict()}")

    df = _run_replay_with_btctrend(qualifying, btc_df)
    if df.empty:
        print("Sonuç yok.")
        return

    print(f"\nGenel: {_stats(df['pnl_usd'].to_numpy())}")

    print("\n-- BTC trend bucket'ları --")
    for trend in ["bearish", "neutral", "bullish"]:
        sub = df[df["btc_trend"] == trend]
        print(f"  {trend:10s} n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())} "
              f"reason={sub['reason'].value_counts().to_dict()}")

    print("\n-- BTC z-score kartilleri --")
    df["zq"] = pd.qcut(df["btc_z"], 4, labels=["Q1(en dip)", "Q2", "Q3", "Q4(en tepe)"])
    for q in ["Q1(en dip)", "Q2", "Q3", "Q4(en tepe)"]:
        sub = df[df["zq"] == q]
        rng = f"{sub['btc_z'].min():.2f}..{sub['btc_z'].max():.2f}"
        print(f"  {q:12s} z={rng:14s} {_stats(sub['pnl_usd'].to_numpy())}")

    corr = np.corrcoef(df["btc_z"], df["pnl_usd"])[0, 1]
    print(f"\nbtc_z vs pnl_usd korelasyonu: {corr:.3f}")


if __name__ == "__main__":
    main()
