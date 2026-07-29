"""
BTC rejim filtresi (btc_trend: bullish/bearish/neutral) — TEMİZ yöntemle
yeniden test (21 Tem 2026, kullanıcı isteği).

Eski `rsi_cross_btc_regime_bt.py` LOOK-AHEAD içeriyordu (jump=post_v-pre_v,
sinyal SONRASI barı kullanıyordu — bu oturumda erken saatlerde tüm VPMV
sıçraması ailesi bu yüzden geçersiz ilan edilmişti). Bu script SADECE
sinyal ANINA kadarki BTC verisini kullanır, hedef sabit 24-bar ileri getiri.

Formül `signal_processor.py:711-728` ile BİREBİR AYNI: BTC 15m kapanışının
EMA200'den kaç std saptığı (z), bullish>0.5 / bearish<-0.5 / neutral arası.

Hipotez: Long, BTC 'bearish' DEĞİLKEN (bullish/neutral) daha mı iyi;
Short, BTC 'bullish' DEĞİLKEN daha mı iyi (rejime-karşı bahis cezalı mı).

Kullanım: python -m research.pattern_lab.rsi_cross_btc_regime_clean_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_FORWARD_BARS = 24
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")
_BULLISH_Z = 0.5
_BEARISH_Z = -0.5


def _fetch_btc_regime(cur) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, close FROM cagg_15m WHERE symbol='BTCUSDT' ORDER BY bucket ASC"
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    ema = df["close"].ewm(span=200, adjust=False).mean()
    std = df["close"].rolling(200).std()
    z = (df["close"] - ema) / (std + 1e-12)
    df["btc_trend"] = np.where(z > _BULLISH_Z, "bullish", np.where(z < _BEARISH_Z, "bearish", "neutral"))
    return df[["bucket", "btc_trend"]].dropna()


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)'
          AND interval IN ('5m','15m','1h','4h')
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY opened_at ASC
        """
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "open_price"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_forward_price(cur, symbol: str, interval: str, after, n_bars: int) -> float | None:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket >= %s ORDER BY bucket ASC LIMIT 1 OFFSET %s",
        (symbol, after, n_bars - 1),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _deep_validate(label: str, group: pd.DataFrame, rest: pd.DataFrame) -> None:
    print(f"\n  -- {label} (n={len(group)}) vs geri kalan (n={len(rest)}) --")
    print(f"    {label}: {_stats(group['fwd_ret'].to_numpy())}")
    print(f"    geri kalan: {_stats(rest['fwd_ret'].to_numpy())}")
    if len(group) == 0 or len(rest) == 0:
        return
    real_gap = group["fwd_ret"].mean() - rest["fwd_ret"].mean()
    combo = pd.concat([group.assign(_g=True), rest.assign(_g=False)])
    rng = np.random.default_rng(42)
    labels = combo["_g"].to_numpy()
    target = combo["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"    placebo: gerçek ort% farkı ({real_gap:+.4f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")
    if len(group) >= 40:
        g_sorted = group.sort_values("opened_at")
        mid = g_sorted["opened_at"].iloc[len(g_sorted)//2]
        fh = _stats(g_sorted[g_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(g_sorted[g_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"    split-period: ilk yarı {fh} | ikinci yarı {sh}")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    btc = _fetch_btc_regime(cur)
    print(f"[btc] {len(btc)} adet 15m BTC rejim barı")
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) sinyali")

    btc_buckets = btc["bucket"].to_numpy()
    btc_trend_arr = btc["btc_trend"].to_numpy()

    records = []
    for i, row in signals.iterrows():
        idx = np.searchsorted(btc_buckets, np.datetime64(row["opened_at"]), side="right") - 1
        if idx < 0:
            continue
        trend = btc_trend_arr[idx]
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        records.append({
            "signal_type": row["signal_type"], "btc_trend": trend,
            "fwd_ret": fwd_ret, "opened_at": row["opened_at"],
        })
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(signals)}")
    conn.close()

    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için BTC rejim + {_FORWARD_BARS}-bar ileri getiri\n")
    if df.empty:
        return

    print("=== [1] Sinyal yönü × BTC rejimi — ham tablo ===")
    g = df.groupby(["signal_type", "btc_trend"])["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    long_df = df[df["signal_type"] == "Long"]
    short_df = df[df["signal_type"] == "Short"]

    print("\n=== [2] Long: BTC bearish DEĞİLKEN vs bearish İKEN ===")
    long_ok = long_df[long_df["btc_trend"] != "bearish"]
    long_bad = long_df[long_df["btc_trend"] == "bearish"]
    _deep_validate("Long (BTC bearish DEĞİL)", long_ok, long_bad)

    print("\n=== [3] Short: BTC bullish DEĞİLKEN vs bullish İKEN ===")
    short_ok = short_df[short_df["btc_trend"] != "bullish"]
    short_bad = short_df[short_df["btc_trend"] == "bullish"]
    _deep_validate("Short (BTC bullish DEĞİL)", short_ok, short_bad)

    print("\n=== [4] Rejim dönemi split (çöküş/toparlanma) — Long-BTC-uyum grubu ===")
    crash = long_ok[long_ok["opened_at"] < _REGIME_SPLIT]
    recovery = long_ok[long_ok["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 50:
            print(f"  {label} (n={len(sub)}): {_stats(sub['fwd_ret'].to_numpy())}")
        else:
            print(f"  {label}: yetersiz örnek (n={len(sub)})")


if __name__ == "__main__":
    main()
