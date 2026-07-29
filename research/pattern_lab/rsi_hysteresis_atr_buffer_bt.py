"""
RSI9/24 Hysteresis (Seviye 1) + SignalFilter ATR-tampon (Seviye 2) —
TEMİZ yöntemle, sıfırdan kendi-kendine simülasyon (21 Tem 2026).

`signal_filter_events` tablosuna güvenilmedi: diagnostik scriptler
(scripts/compare_filter_output.py, scripts/visualize_filter.py) de aynı
tabloya "passed" event yazıyor (canlı sinyal sayısının ~8 katı kadar) —
gerçek `signals` tablosuyla join denemesi 446.837 passed event'ten sadece
196'sını eşleştirdi. Bu yüzden RSI9/24 kesişimlerini VE SignalFilter'ın
referans-zinciri mantığını (signal_filter.py ile BİREBİR aynı state
machine — her deneme, geçti/geçmedi fark etmeksizin referansı günceller)
ham fiyat verisinden SIFIRDAN kendimiz simüle ediyoruz.

Seviye 1: kesişim anındaki |RSI9-RSI24| ayrımı (sep) → temiz ileri getiriyle
  korelasyon (büyük sep = hysteresis bandının rahat geçtiği, "gerçek" cross).
Seviye 2: k=0 (mevcut/baseline) filtre + ATR-tamponlu varyantlar (k=0.25/
  0.5/1.0) — baseline'da geçen ama tampon varyantında ELENECEK olanların
  ileri getirisi, TUTULACAK olanlarla karşılaştırılıyor.

Kullanım: python -m research.pattern_lab.rsi_hysteresis_atr_buffer_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_atr, calculate_rsi

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m"}
_FAST, _SLOW = 9, 24
_WARMUP = 60
_FORWARD_BARS = 24
_K_LEVELS = (0.25, 0.5, 1.0)
_PLACEBO_ITER = 200
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _fetch_symbols(cur, interval: str) -> list[str]:
    table = _INTERVAL_TABLE[interval]
    cur.execute(f"SELECT DISTINCT symbol FROM {table}")
    return [r[0] for r in cur.fetchall()]


def _fetch_series(cur, symbol: str, interval: str) -> pd.DataFrame:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT bucket, high, low, close FROM {table} WHERE symbol=%s ORDER BY bucket ASC",
        (symbol,),
    )
    rows = cur.fetchall()
    if len(rows) < _WARMUP + _FORWARD_BARS + 5:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["bucket", "high", "low", "close"]).astype(
        {"high": float, "low": float, "close": float}
    )


def _simulate_symbol(df: pd.DataFrame) -> list[dict]:
    rsi_fast = calculate_rsi(df, period=_FAST, price_col="close")
    rsi_slow = calculate_rsi(df, period=_SLOW, price_col="close")
    atr = calculate_atr(df, period=14)
    diff = (rsi_fast - rsi_slow).to_numpy()
    sign = np.sign(diff)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = atr.to_numpy()
    buckets = df["bucket"].to_numpy()
    n = len(df)

    last_long = None  # (high, low)
    last_short = None
    out = []
    for i in range(_WARMUP, n - _FORWARD_BARS):
        if np.isnan(sign[i]) or np.isnan(sign[i - 1]) or sign[i - 1] == 0 or sign[i] == 0:
            continue
        if sign[i] == sign[i - 1]:
            continue
        sig_type = "Long" if sign[i] > 0 else "Short"
        h, l, sep, a = highs[i], lows[i], abs(diff[i]), atrs[i]
        opposite = last_short if sig_type == "Long" else last_long

        passed_flags = {}
        if opposite is not None and not np.isnan(a):
            for k in (0.0,) + _K_LEVELS:
                buf = k * a
                if sig_type == "Long":
                    passed_flags[k] = h > opposite[0] + buf
                else:
                    passed_flags[k] = l < opposite[1] - buf

        if sig_type == "Long":
            last_long = (h, l)
        else:
            last_short = (h, l)

        if not passed_flags or not passed_flags[0.0]:
            continue  # baseline'da geçmedi -> zaten canlıda sinyal olmazdı

        side = 1.0 if sig_type == "Long" else -1.0
        fwd_ret = (closes[i + _FORWARD_BARS - 1] - closes[i]) / closes[i] * 100.0 * side
        rec = {
            "sig_type": sig_type, "sep": sep, "fwd_ret": fwd_ret,
            "bar_time": pd.Timestamp(buckets[i]),
        }
        for k in _K_LEVELS:
            rec[f"pass_k{k}"] = passed_flags[k]
        out.append(rec)
    return out


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


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()

    records = []
    for interval in _INTERVAL_TABLE:
        symbols = _fetch_symbols(cur, interval)
        print(f"[{interval}] {len(symbols)} sembol taranacak")
        for si, symbol in enumerate(symbols):
            df = _fetch_series(cur, symbol, interval)
            if df.empty:
                continue
            recs = _simulate_symbol(df)
            for r in recs:
                r["interval"] = interval
                r["symbol"] = symbol
            records.extend(recs)
            if (si + 1) % 100 == 0:
                print(f"  ... {si+1}/{len(symbols)} sembol ({len(records)} baseline-geçen kesişim)")
    conn.close()

    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} baseline-geçen (k=0) RSI9/24 kesişimi, {_FORWARD_BARS}-bar ileri getiri\n")
    if df.empty:
        return

    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SEVİYE 1 — Kesişim anındaki |RSI9-RSI24| ayrımı (hysteresis sinyali)")
    print("=" * 70)
    rho, p = spearmanr(df["sep"], df["fwd_ret"])
    print(f"[1] Korelasyon: rho={rho:+.4f} (p={p:.4f})")

    tercile = pd.qcut(df["sep"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    rng = np.random.default_rng(42)
    vals = df["sep"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
    )
    print(f"[2] Placebo (korelasyon karıştırma): %{count_ge/_PLACEBO_ITER*100:.1f}")

    crash = df[df["bar_time"] < _REGIME_SPLIT]
    recovery = df[df["bar_time"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            r, p2 = spearmanr(sub["sep"], sub["fwd_ret"])
            print(f"[3] Split — {label} (n={len(sub)}): rho={r:+.4f} (p={p2:.4f})")
        else:
            print(f"[3] Split — {label}: yetersiz örnek (n={len(sub)})")

    print("\n  En yüksek sep çeyreği içinde kronolojik yarı-yarı:")
    top_q = df[df["sep"] >= df["sep"].quantile(0.75)].sort_values("bar_time")
    if len(top_q) >= 50:
        mid = top_q["bar_time"].iloc[len(top_q)//2]
        fh = _stats(top_q[top_q["bar_time"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(top_q[top_q["bar_time"] >= mid]["fwd_ret"].to_numpy())
        print(f"    ilk yarı: {fh} | ikinci yarı: {sh}")

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SEVİYE 2 — ATR-tamponlu SignalFilter varyantları")
    print("=" * 70)
    print(f"  Baseline (k=0, mevcut canlı davranış): {_stats(df['fwd_ret'].to_numpy())}")

    for k in _K_LEVELS:
        col = f"pass_k{k}"
        kept = df[df[col]]
        cut = df[~df[col]]
        print(f"\n  --- k={k}×ATR tampon ---")
        print(f"  Tutulan  (n={len(kept)}): {_stats(kept['fwd_ret'].to_numpy())}")
        print(f"  Kesilen  (n={len(cut)}): {_stats(cut['fwd_ret'].to_numpy())}")
        if len(kept) == 0 or len(cut) == 0:
            continue

        real_diff = kept["fwd_ret"].mean() - cut["fwd_ret"].mean()
        labels = df[col].to_numpy()
        target = df["fwd_ret"].to_numpy()
        count_ge = 0
        for _ in range(_PLACEBO_ITER):
            shuffled = rng.permutation(labels)
            fake_kept = target[shuffled]
            fake_cut = target[~shuffled]
            fake_diff = fake_kept.mean() - fake_cut.mean()
            if abs(fake_diff) >= abs(real_diff):
                count_ge += 1
        print(f"  Placebo (tutulan-kesilen farkı karıştırma): gerçek fark={real_diff:+.4f} "
              f"— rastgelede aynı/daha büyük sıklık: %{count_ge/_PLACEBO_ITER*100:.1f}")

        kept_sorted = kept.sort_values("bar_time")
        if len(kept_sorted) >= 50:
            mid = kept_sorted["bar_time"].iloc[len(kept_sorted)//2]
            fh = _stats(kept_sorted[kept_sorted["bar_time"] < mid]["fwd_ret"].to_numpy())
            sh = _stats(kept_sorted[kept_sorted["bar_time"] >= mid]["fwd_ret"].to_numpy())
            print(f"  Tutulan grubu içi kronolojik yarı-yarı: ilk={fh} | ikinci={sh}")


if __name__ == "__main__":
    main()
