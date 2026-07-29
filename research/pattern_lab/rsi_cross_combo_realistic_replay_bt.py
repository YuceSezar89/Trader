"""
Gerçekçi SL/TP/trailing replay — "didik didik" serisinin şu ana kadarki
TÜM sonuçları sabit 24-bar ileri getiri (teorik) üzerinden hesaplandı.
do_open_streak'te teorik Gauss-eşiği ile gerçekçi bar-bar replay arasında
%98.5'lik bir uçurum bulmuştuk — aynı kontrol burada da yapılıyor (24 Tem
2026, kullanıcı isteği).

Risk kuralları CANLI sistemle (signals/risk_policy.py + signals/trailing.py)
BİREBİR aynı:
  - SL = giriş ATR(14) × 1.5, TP = giriş ATR(14) × 3.0 (baz, VPMV/MTF/15m
    bonusları burada hesaplanmıyor — basit/muhafazakâr varsayım)
  - TP'ye değince pozisyon KAPANMAZ, trailing_stop aktive olur (mesafe=SL
    mesafesiyle aynı, static fallback — trailing.py::update_trailing)
  - Aynı bar içinde SL VE TP ikisi de menzildeyse SL ÖNCE kontrol edilir
    (muhafazakâr kural, tp_sl_lever_sim.py ile aynı disiplin)
  - rsi_cross_live'ın CANLIDA max_hold_hours YOK (sınırsız) — backtest'te
    güvenlik için 500 bar (~41.7 saat) zaman aşımı sınırı konuldu, bu SADECE
    backtest güvenlik önlemi, canlı davranışın bir parçası değil

Kaldıraç 3x, POSITION_USD=$25, FEE_RATE=%0.05 (giriş+çıkış) — rsi_cross_live
ile birebir (paper_trade_manager.py).

Test edilen popülasyonlar:
  - Long ÜÇLÜ: all_up + TA-base(pct>=55) + kovalama + HA-hizalı (n=428 teorik)
  - Short İKİLİ: all_up + TA-base(pct<=45) + kovalama (HA'sız, n=253 teorik)

Kullanım: python -m research.pattern_lab.rsi_cross_combo_realistic_replay_bt
(önce rsi_cross_ta_triple_combo_bt.py VE rsi_cross_ta_triple_combo_short_bt.py
çalıştırılmış olmalı — cache'lerini kullanır)
"""

import os

import numpy as np
import pandas as pd

from indicators.core import calculate_atr
from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH

_SHORT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_ta_triple_short.parquet")

_SL_MULT = 1.5
_TP_MULT = 3.0
_MAX_HOLD_BARS = 500
_POSITION_USD = 25.0
_LEVERAGE = 3.0
_FEE_RATE = 0.0005
_PLACEBO_ITER = 300
_BASE_TH_LONG = 55
_EXTREME_TH_LONG = 80
_BASE_TH_SHORT = 45
_EXTREME_TH_SHORT = 20


def _fetch_signal_open_prices(cur, direction: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = %s AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        """,
        (direction,),
    )
    return pd.DataFrame(cur.fetchall(), columns=["symbol", "opened_at", "open_price"])


def _fetch_5m_full(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, open, high, low, close FROM cagg_5m WHERE symbol=%s ORDER BY bucket ASC",
        (symbol,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["atr"] = calculate_atr(df)
    return df


def _simulate(h: np.ndarray, l: np.ndarray, c: np.ndarray, start: int, direction: str,
              entry_price: float, atr_entry: float) -> tuple[float, str, int]:
    sl_dist = atr_entry * _SL_MULT
    tp_dist = atr_entry * _TP_MULT
    trail_dist = sl_dist
    if direction == "Long":
        sl, tp = entry_price - sl_dist, entry_price + tp_dist
    else:
        sl, tp = entry_price + sl_dist, entry_price - tp_dist

    trail = None
    end = min(len(h), start + _MAX_HOLD_BARS)
    for i in range(start, end):
        hi, lo = h[i], l[i]
        if direction == "Long":
            if trail is None:
                if lo <= sl:
                    return sl, "stop_loss", i - start + 1
                if hi >= tp:
                    trail = tp - trail_dist
            else:
                new_trail = hi - trail_dist
                if new_trail > trail:
                    trail = new_trail
                if lo <= trail:
                    return trail, "trailing_stop", i - start + 1
        else:
            if trail is None:
                if hi >= sl:
                    return sl, "stop_loss", i - start + 1
                if lo <= tp:
                    trail = tp + trail_dist
            else:
                new_trail = lo + trail_dist
                if new_trail < trail:
                    trail = new_trail
                if hi >= trail:
                    return trail, "trailing_stop", i - start + 1
    end_idx = end - 1
    return c[end_idx], "timeout_cap", end_idx - start + 1


def _pnl_usd(direction: str, entry: float, exit_p: float) -> float:
    pnl_pct = (exit_p - entry) / entry * 100.0 if direction == "Long" else (entry - exit_p) / entry * 100.0
    fee = _POSITION_USD * _FEE_RATE * 2
    return (pnl_pct / 100.0) * _POSITION_USD * _LEVERAGE - fee


def _run_replay(label: str, qualifying: pd.DataFrame, direction: str) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, direction)
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"[{label}] açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

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
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, direction, entry_price, float(atr[idx]))
            pnl_usd = _pnl_usd(direction, entry_price, exit_price)
            results.append({
                "symbol": sym, "opened_at": row["opened_at"], "pnl_usd": pnl_usd,
                "reason": reason, "bars_held": bars_held,
            })
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def _stats(pnl: np.ndarray) -> dict:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return {"n": 0}
    g, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "n": len(pnl), "wr": round(float((pnl > 0).mean() * 100), 1),
        "toplam_$": round(float(pnl.sum()), 2), "medyan_$": round(float(np.median(pnl)), 3),
        "ort_$": round(float(pnl.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _deep_validate_usd(label: str, df: pd.DataFrame) -> None:
    print(f"\n  -- {label} (n={len(df)}) --")
    print(f"    {_stats(df['pnl_usd'].to_numpy())}")
    print(f"    kapanış nedeni dağılımı: {df['reason'].value_counts().to_dict()}")
    print(f"    ort. tutulan bar sayısı: {df['bars_held'].mean():.1f} (5dk bar ~ {df['bars_held'].mean()*5/60:.1f} saat)")

    if len(df) >= 40:
        d_sorted = df.sort_values("opened_at")
        mid = d_sorted["opened_at"].iloc[len(d_sorted)//2]
        fh = _stats(d_sorted[d_sorted["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(d_sorted[d_sorted["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"    split-period: ilk yarı {fh} | ikinci yarı {sh}")

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    summarize(label, df["pnl_usd"].to_numpy(), df["opened_at"], days_span)


def main() -> None:
    print("=" * 78)
    print("LONG ÜÇLÜ (all_up + TA-base>=55 + kovalama + HA-hizalı) — gerçekçi replay")
    print("=" * 78)
    long_df = pd.read_parquet(_HA_CACHE_PATH)
    long_df = long_df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"])
    ta_base = (long_df["pct_1h"] >= _BASE_TH_LONG) & (long_df["pct_4h"] >= _BASE_TH_LONG)
    kovalama = ((long_df["pct_1h"] >= _EXTREME_TH_LONG) & (long_df["slope_1h"] > 0)) | \
               ((long_df["pct_4h"] >= _EXTREME_TH_LONG) & (long_df["slope_4h"] > 0))
    ha = (long_df["ha_bull_1h"] > 0.5) & (long_df["ha_bull_4h"] > 0.5)
    long_qualifying = long_df[ta_base & kovalama & ha][["symbol", "opened_at"]].reset_index(drop=True)

    long_result = _run_replay("Long üçlü", long_qualifying, "Long")
    if not long_result.empty:
        long_result.to_parquet(os.path.join(os.path.dirname(__file__), "_cache_replay_long.parquet"))
        _deep_validate_usd("Long üçlü — gerçekçi $ PnL", long_result)

    print("\n" + "=" * 78)
    print("SHORT İKİLİ (all_up + TA-base<=45 + kovalama, HA'sız) — gerçekçi replay")
    print("=" * 78)
    short_df = pd.read_parquet(_SHORT_CACHE_PATH)
    short_df = _add_all_up(short_df)
    short_df = short_df[short_df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"])
    ta_base_s = (short_df["pct_1h"] <= _BASE_TH_SHORT) & (short_df["pct_4h"] <= _BASE_TH_SHORT)
    kovalama_s = ((short_df["pct_1h"] <= _EXTREME_TH_SHORT) & (short_df["slope_1h"] < 0)) | \
                 ((short_df["pct_4h"] <= _EXTREME_TH_SHORT) & (short_df["slope_4h"] < 0))
    short_qualifying = short_df[ta_base_s & kovalama_s][["symbol", "opened_at"]].reset_index(drop=True)

    short_result = _run_replay("Short ikili", short_qualifying, "Short")
    if not short_result.empty:
        short_result.to_parquet(os.path.join(os.path.dirname(__file__), "_cache_replay_short.parquet"))
        _deep_validate_usd("Short ikili — gerçekçi $ PnL", short_result)


if __name__ == "__main__":
    main()
