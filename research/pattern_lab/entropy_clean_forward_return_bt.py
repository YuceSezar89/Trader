"""
Shannon Entropy — Hurst/FDI/ADX ile AYNI temiz yöntemle, KAPSAMLI test (21 Tem 2026).

Getiri serisini eşit-genişlikte kutulara ayırıp (kutu sayısı parametre),
her kutunun olasılığından H = -Σ(p_i × log2(p_i)) hesaplanır, log2(kutu
sayısı)'na bölünüp 0-1'e normalize edilir. Eşit-genişlik kutulama bilinçli
seçildi (quantile-kutulama yapısal olarak her zaman ~max entropiye yakın
çıkar, ayrım gücü taşımaz) — sıkışık/yönlü getiriler DÜŞÜK, dağınık/rastgele
getiriler YÜKSEK entropi vermeli.

"Kapsamlı" — 3 pencere (50/100/200 bar) × 2 kutu sayısı (5/10) = 6 kombinasyon,
her biri sinyal açılış anında hesaplanıp çıkış mekanizmasından bağımsız sabit
24-bar ileri getiriyle test edilir. En umut verici kombinasyon için permütasyon
+ split-period doğrulaması da DAHİL (ADX'te sonradan eklemek zorunda kalmıştık).

Kullanım: python -m research.pattern_lab.entropy_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_WINDOWS = (50, 100, 200)
_BIN_COUNTS = (5, 10)
_FORWARD_BARS = 24
_MAX_SIGNALS = 6000
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _entropy(returns: np.ndarray, n_bins: int) -> float | None:
    if len(returns) < n_bins * 3:
        return None
    lo, hi = returns.min(), returns.max()
    if hi <= lo:
        return None
    counts, _ = np.histogram(returns, bins=n_bins, range=(lo, hi))
    probs = counts[counts > 0] / counts.sum()
    h = -np.sum(probs * np.log2(probs))
    return float(h / np.log2(n_bins))  # 0-1 normalize


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, opened_at, open_price
        FROM signals
        WHERE interval IN ('5m','15m','1h') AND open_price IS NOT NULL AND open_price > 0
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "open_price"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_closes_before(cur, symbol: str, interval: str, before, n: int) -> np.ndarray:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket < %s ORDER BY bucket DESC LIMIT %s",
        (symbol, before, n),
    )
    rows = cur.fetchall()
    if len(rows) < n:
        return np.array([])
    return np.array([float(r[0]) for r in reversed(rows)])


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


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} sinyal örneklendi (status/close_reason önemsiz)")

    max_n = max(_WINDOWS)
    cache: dict[int, tuple] = {}
    for i, row in signals.iterrows():
        closes = _fetch_closes_before(cur, row["symbol"], row["interval"], row["opened_at"], max_n)
        if len(closes) < max_n:
            continue
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        returns = np.diff(np.log(closes))
        cache[row["id"]] = (returns, fwd_ret, row["opened_at"])
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")
    conn.close()
    print(f"[collect] {len(cache)} sinyal için veri hazır\n")

    best_combo, best_abs_rho = None, 0.0
    for w in _WINDOWS:
        for n_bins in _BIN_COUNTS:
            records = []
            for _id, (returns, fwd_ret, opened_at) in cache.items():
                sub = returns[-w:]
                e = _entropy(sub, n_bins)
                if e is None:
                    continue
                records.append({"entropy": e, "fwd_ret": fwd_ret, "opened_at": opened_at})
            df = pd.DataFrame(records)
            if df.empty or df["entropy"].std() == 0:
                print(f"pencere={w:>4} kutu={n_bins}: veri yok/sabit")
                continue
            rho, p = spearmanr(df["entropy"], df["fwd_ret"])
            print(f"pencere={w:>4} kutu={n_bins:>2} | n={len(df):>5} | Entropy: "
                  f"medyan={df['entropy'].median():.3f} std={df['entropy'].std():.3f} "
                  f"| rho={rho:+.4f} (p={p:.4f})")
            if abs(rho) > best_abs_rho:
                best_abs_rho, best_combo = abs(rho), (w, n_bins, df)

    if best_combo is None:
        print("\nHiçbir kombinasyonda kullanılabilir veri yok.")
        return

    w, n_bins, df = best_combo
    print(f"\n{'='*70}\nEN UMUT VERİCİ KOMBİNASYON: pencere={w}, kutu={n_bins} — DERİN DOĞRULAMA\n{'='*70}")
    rho, p = spearmanr(df["entropy"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    tercile = pd.qcut(df["entropy"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["fwd_ret"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(g.to_string())

    rng = np.random.default_rng(42)
    vals = df["entropy"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
    )
    print(f"  Placebo (korelasyon karıştırma): %{count_ge/_PLACEBO_ITER*100:.1f}")

    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            r, p2 = spearmanr(sub["entropy"], sub["fwd_ret"])
            print(f"  Split — {label} (n={len(sub)}): rho={r:+.4f} (p={p2:.4f})")
        else:
            print(f"  Split — {label}: yetersiz örnek (n={len(sub)})")

    print("\n  Filtre simülasyonu + kronolojik yarı-yarı (her dörtte-bir dilim için):")
    for q_lo, q_hi, label in ((0.0, 0.25, "en düşük %25"), (0.75, 1.0, "en yüksek %25")):
        lo_val, hi_val = df["entropy"].quantile(q_lo), df["entropy"].quantile(q_hi)
        sub = df[(df["entropy"] >= lo_val) & (df["entropy"] <= hi_val)].sort_values("opened_at")
        s = _stats(sub["fwd_ret"].to_numpy())
        mid = sub["opened_at"].iloc[len(sub)//2] if len(sub) > 10 else None
        if mid is not None:
            fh = _stats(sub[sub["opened_at"] < mid]["fwd_ret"].to_numpy())
            sh = _stats(sub[sub["opened_at"] >= mid]["fwd_ret"].to_numpy())
            print(f"    {label}: {s} | ilk_yarı_PF={fh.get('pf','-')} ikinci_yarı_PF={sh.get('pf','-')}")
        else:
            print(f"    {label}: {s}")


if __name__ == "__main__":
    main()
