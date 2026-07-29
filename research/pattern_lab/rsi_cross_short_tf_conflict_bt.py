"""
Short kovalama popülasyonunda (n=253) TÜM TF varyasyonları (5m/15m/1h/4h)
arasındaki EĞİM UYUMU/ÇELİŞKİSİ test ediliyor. (25 Tem 2026, kullanıcı
isteği — PRLUSDT vakasının genellemesi: "1h zaten dönmüş ama 4h'den geçiyor"
tipi çelişkiler ne kadar yaygın ve ne kadar zararlı, TÜM TF çiftlerinde.)

REPAINT DİSİPLİNİ (kullanıcının özellikle istediği): her TF'nin barı
GERÇEKTEN kapanmış olmalı — bucket + interval_dakika <= giriş_anı. 15m/1h/4h
için bu kontrol açıkça uygulanıyor (bugün ta_kovalama_gate.py'de bulduğumuz
forming-bar bug'ıyla AYNI kategoriden bir hataya düşmemek için). 5m zaten
sinyalin kendi (tetikleyici) barı, backtest'te tarihsel veri olduğu için
doğal olarak kapanmış.

Kullanım: python -m research.pattern_lab.rsi_cross_short_tf_conflict_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _SHORT_CACHE_PATH,
    _fetch_5m_full,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)
from signals.ta_kovalama_gate import _net_ta_series, _percentile_now, _slope_now

_BASE_TH_SHORT = 45.0
_EXTREME_TH_SHORT = 20.0
_INTERVAL_MIN = {"15m": 15, "1h": 60, "4h": 240}


def _fetch_closed_tf(cur, symbol: str, view: str, interval_min: int, cutoff, limit: int = 220) -> pd.DataFrame:
    """view'dan (cagg_15m/1h/4h) GERÇEKTEN kapanmış barları çeker — bucket +
    interval_min <= cutoff. get_cagg_klines(closed_only=True)'ın backtest
    (psycopg2, senkron) eşleniği."""
    from datetime import timedelta

    closed_cutoff = cutoff - timedelta(minutes=interval_min)
    cur.execute(
        f"SELECT bucket, open, high, low, close FROM {view} "
        f"WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
        (symbol, closed_cutoff, limit),
    )
    rows = sorted(cur.fetchall(), key=lambda r: r[0])
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["open_time"] = [int(b.timestamp() * 1000) for b in df["bucket"]]
    return df


def main() -> None:
    short_df = pd.read_parquet(_SHORT_CACHE_PATH)
    short_df = _add_all_up(short_df)
    short_df = short_df[short_df["all_up"]].dropna(
        subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"]
    )
    ta_base = (short_df["pct_1h"] <= _BASE_TH_SHORT) & (short_df["pct_4h"] <= _BASE_TH_SHORT)
    kovalama = ((short_df["pct_1h"] <= _EXTREME_TH_SHORT) & (short_df["slope_1h"] < 0)) | (
        (short_df["pct_4h"] <= _EXTREME_TH_SHORT) & (short_df["slope_4h"] < 0)
    )
    qualifying = short_df[ta_base & kovalama].copy().reset_index(drop=True)
    print(f"Short kovalama popülasyonu: n={len(qualifying)}")

    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Short")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen: n={len(merged)}")

    results = []
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_full(cur, sym)
        if df5.empty:
            continue
        b5 = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()
        df5c = df5.copy()
        df5c["open_time"] = [int(x.timestamp() * 1000) for x in df5["bucket"]]

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            opened_at = row["opened_at"]
            idx = np.searchsorted(b5, np.datetime64(opened_at), side="right") - 1
            if idx < 14 or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue

            # 5m: sinyalin kendi barına kadar (dahil) — zaten tetikleyici bar
            net5 = _net_ta_series(df5c.iloc[: idx + 1])
            pct_5m = _percentile_now(net5)
            slope_5m = _slope_now(net5)

            # 15m: GERÇEKTEN kapanmış barlarla
            df15 = _fetch_closed_tf(cur, sym, "cagg_15m", 15, opened_at)
            if len(df15) < 53:
                continue
            net15 = _net_ta_series(df15)
            pct_15m = _percentile_now(net15)
            slope_15m = _slope_now(net15)

            if any(v != v for v in (pct_5m, slope_5m, pct_15m, slope_15m)):
                continue

            entry_price = float(row["open_price"])
            atr_entry = float(atr[idx])
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Short", entry_price, atr_entry)
            pnl = _pnl_usd("Short", entry_price, exit_price)

            results.append(
                {
                    "symbol": sym,
                    "opened_at": opened_at,
                    "pnl_usd": pnl,
                    "reason": reason,
                    "pct_5m": pct_5m, "slope_5m": slope_5m,
                    "pct_15m": pct_15m, "slope_15m": slope_15m,
                    "pct_1h": row["pct_1h"], "slope_1h": row["slope_1h"],
                    "pct_4h": row["pct_4h"], "slope_4h": row["slope_4h"],
                }
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    df = pd.DataFrame(results)
    print(f"\nTüm TF'lerde geçerli veri olan işlem: {len(df)}")
    print(f"Baseline: {_stats(df['pnl_usd'].to_numpy())}")

    for tf in ["5m", "15m", "1h", "4h"]:
        df[f"agree_{tf}"] = df[f"slope_{tf}"] < 0  # Short yönüyle uyumlu mu

    df["agree_count"] = df[[f"agree_{tf}" for tf in ["5m", "15m", "1h", "4h"]]].sum(axis=1)

    print("\n-- Kaç TF eğimi Short yönüyle uyumlu (0-4) --")
    for k in range(5):
        sub = df[df["agree_count"] == k]
        if len(sub) < 5:
            print(f"  {k}/4 uyumlu: n={len(sub)} (yetersiz)")
            continue
        print(f"  {k}/4 uyumlu: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}")

    print("\n-- Eşik: en az K TF uyumlu olsun --")
    pool = df["pnl_usd"].to_numpy()
    rng = np.random.default_rng(42)
    for k in [2, 3, 4]:
        sub = df[df["agree_count"] >= k]
        if len(sub) < 10:
            continue
        real_mean = sub["pnl_usd"].mean()
        ge = sum(1 for _ in range(1000) if rng.choice(pool, size=len(sub), replace=False).mean() >= real_mean)
        print(f"  >= {k}/4: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  placebo(iyi)=%{ge/10:.1f}")

    print("\n-- >=3/4 filtresi için split-period (IS/OOS) kontrolü --")
    for k in [3, 4]:
        sub = df[df["agree_count"] >= k].sort_values("opened_at")
        if len(sub) < 20:
            print(f"  >= {k}/4: n={len(sub)} yetersiz")
            continue
        mid = sub["opened_at"].iloc[len(sub) // 2]
        fh = _stats(sub[sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(sub[sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"  >= {k}/4: ilk yarı {fh} | ikinci yarı {sh}")

    # Karşı-doğrulama: 2/4 (kötü) grubun kötülüğü de iki yarıda tutarlı mı
    bad = df[df["agree_count"] == 2].sort_values("opened_at")
    if len(bad) >= 10:
        mid = bad["opened_at"].iloc[len(bad) // 2]
        fh = _stats(bad[bad["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(bad[bad["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"\n  2/4 (kötü) grup split-period: ilk yarı {fh} | ikinci yarı {sh}")

    print("\n-- Her TF çifti için ÇELİŞKİ testi (biri Short'u onaylıyor, diğeri ZATEN ters dönmüş) --")
    tfs = ["5m", "15m", "1h", "4h"]
    for i, tf_a in enumerate(tfs):
        for tf_b in tfs[i + 1 :]:
            conflict = (~df[f"agree_{tf_a}"]) & df[f"agree_{tf_b}"] | (~df[f"agree_{tf_b}"]) & df[f"agree_{tf_a}"]
            sub = df[conflict]
            if len(sub) < 8:
                continue
            real_mean = sub["pnl_usd"].mean()
            le = sum(1 for _ in range(1000) if rng.choice(pool, size=len(sub), replace=False).mean() <= real_mean)
            print(f"  {tf_a} vs {tf_b} çelişkili: n={len(sub):4d}  {_stats(sub['pnl_usd'].to_numpy())}  placebo(kötü)=%{le/10:.1f}")


if __name__ == "__main__":
    main()
