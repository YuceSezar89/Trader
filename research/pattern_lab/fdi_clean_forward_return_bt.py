"""
Fractal Dimension Index (FDI, Ehlers/FRAMA formülü) — Hurst ile AYNI temiz
yöntemle test (21 Tem 2026): sinyal açılış anında hesaplanan bir FİLTRE
olarak, çıkış mekanizmasından bağımsız sabit 24-bar ileri getiri hedefiyle.

Formül (Ehlers): pencere ikiye bölünür (ilk yarı/ikinci yarı), her yarının
ve tüm pencerenin (high-low)/uzunluk oranından D = (log(N1+N2)-log(N3))/log(2)
hesaplanır. D≈1 = düz/temiz trend, D≈2 = çırpıntı/gürültü. Filtre hipotezi:
DÜŞÜK FDI (temiz trend) sinyalleri DAHA İYİ forward-return vermeli.

Kullanım: python -m research.pattern_lab.fdi_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_FDI_WINDOWS = (16, 32, 64)  # Ehlers orijinali N=16, komşu boyutlar da denenir
_FORWARD_BARS = 24
_MAX_SIGNALS = 6000
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _fdi(highs: np.ndarray, lows: np.ndarray) -> float | None:
    n = len(highs)
    if n < 4 or n % 2 != 0:
        return None
    half = n // 2
    n1 = (highs[:half].max() - lows[:half].min()) / half
    n2 = (highs[half:].max() - lows[half:].min()) / half
    n3 = (highs.max() - lows.min()) / n
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        return None
    d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    return float(d)


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


def _fetch_hl_before(cur, symbol: str, interval: str, before, n: int) -> tuple[np.ndarray, np.ndarray]:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT high, low FROM {table} WHERE symbol=%s AND bucket < %s ORDER BY bucket DESC LIMIT %s",
        (symbol, before, n),
    )
    rows = cur.fetchall()
    if len(rows) < n:
        return np.array([]), np.array([])
    rows = list(reversed(rows))
    return np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows])


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

    max_n = max(_FDI_WINDOWS)
    cache: dict[int, tuple] = {}
    for i, row in signals.iterrows():
        highs, lows = _fetch_hl_before(cur, row["symbol"], row["interval"], row["opened_at"], max_n)
        if len(highs) < max_n:
            continue
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        cache[row["id"]] = (highs, lows, fwd_ret, row["opened_at"])
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")
    conn.close()
    print(f"[collect] {len(cache)} sinyal için veri hazır\n")

    for w in _FDI_WINDOWS:
        records = []
        for _id, (highs, lows, fwd_ret, opened_at) in cache.items():
            sub_h, sub_l = highs[-w:], lows[-w:]
            d = _fdi(sub_h, sub_l)
            if d is None:
                continue
            records.append({"fdi": d, "fwd_ret": fwd_ret, "opened_at": opened_at})
        df = pd.DataFrame(records)
        if df.empty:
            print(f"pencere={w}: veri yok")
            continue

        print(f"{'='*70}\npencere={w} bar (n={len(df)})\n{'='*70}")
        print(f"  FDI dağılımı: min={df['fdi'].min():.3f} medyan={df['fdi'].median():.3f} "
              f"max={df['fdi'].max():.3f} std={df['fdi'].std():.3f}")

        rho, p = spearmanr(df["fdi"], df["fwd_ret"])
        print(f"  Korelasyon (FDI vs {_FORWARD_BARS}-bar ileri getiri): rho={rho:+.4f} (p={p:.4f})")

        tercile = pd.qcut(df["fdi"], 4, labels=["1.düz", "2", "3", "4.çırpıntı"], duplicates="drop")
        g = df.groupby(tercile, observed=True)["fwd_ret"].agg(
            ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
        )
        print(g.to_string())

        rng = np.random.default_rng(42)
        vals = df["fdi"].to_numpy()
        target = df["fwd_ret"].to_numpy()
        count_ge = sum(
            1 for _ in range(_PLACEBO_ITER)
            if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
        )
        print(f"  Placebo: %{count_ge/_PLACEBO_ITER*100:.1f}")

        print("  Filtre simülasyonu (düşük FDI = temiz trend, eşik altını tut):")
        for thr in df["fdi"].quantile([0.25, 0.5, 0.75]).values:
            sub = df[df["fdi"] <= thr]
            s = _stats(sub["fwd_ret"].to_numpy())
            print(f"    FDI<={thr:.3f}: n={s.get('n',0):>5} | WR={s.get('wr',0):>5.1f}% "
                  f"ort%={s.get('ort_%',0):>7.3f} PF={s.get('pf',0):>6.3f}")

        crash = df[df["opened_at"] < _REGIME_SPLIT]
        recovery = df[df["opened_at"] >= _REGIME_SPLIT]
        for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
            if len(sub) >= 100:
                r, p2 = spearmanr(sub["fdi"], sub["fwd_ret"])
                print(f"  Split — {label} (n={len(sub)}): rho={r:+.4f} (p={p2:.4f})")
            else:
                print(f"  Split — {label}: yetersiz örnek (n={len(sub)})")
        print()


if __name__ == "__main__":
    main()
