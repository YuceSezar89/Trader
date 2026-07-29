"""
Hurst Exponent — ÇIKIŞ MEKANİZMASINDAN BAĞIMSIZ temiz hedefle test (21 Tem 2026).

Önceki testler (hurst_reversal_noise_bt.py) `realized_pnl`'i hedef aldı —
ama bu, sinyalin SL/TP/trailing/reversal çıkış mekanizmasına bağımlı, ve
bugün o mekanizmada (özellikle hızlı-reversal gürültüsü) sorun bulduk.
"Hurst işe yaramıyor" mu yoksa "gürültülü etiket Hurst'ü gizliyor" mu
ayırt edilemiyordu.

Bu script Madde-4/RSI Cross Combined Score'un kullandığı YÖNTEMLE aynı:
çıkış mekanizmasına HİÇ bakmadan, sabit N-bar SONRAKİ fiyatı ölçer
(yön-ayarlı). close_reason/status önemsiz — TÜM sinyaller kullanılabilir.

Kullanım: python -m research.pattern_lab.hurst_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_HURST_BARS = 100  # pencere taramasında en iyi (en az kötü) sonucu veren
_FORWARD_BARS = 24
_MAX_SIGNALS = 6000
_MIN_WINDOW = 8
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _hurst(returns: np.ndarray) -> float | None:
    n = len(returns)
    if n < _MIN_WINDOW * 4:
        return None
    max_window = n // 2
    window_sizes = np.unique(np.logspace(np.log10(_MIN_WINDOW), np.log10(max_window), num=12).astype(int))
    pairs = []
    for w in window_sizes:
        if w < 2:
            continue
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * w : (i + 1) * w]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            r = dev.max() - dev.min()
            s = chunk.std()
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            pairs.append((w, float(np.mean(rs_chunk))))
    if len(pairs) < 4:
        return None
    log_w = np.log([p[0] for p in pairs])
    log_rs = np.log([p[1] for p in pairs])
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


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


def _fetch_bars_before(cur, symbol: str, interval: str, before, n: int) -> np.ndarray:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket < %s ORDER BY bucket DESC LIMIT %s",
        (symbol, before, n),
    )
    rows = cur.fetchall()
    if len(rows) < n // 2:
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


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} sinyal örneklendi (status/close_reason önemsiz)")

    records = []
    for i, row in signals.iterrows():
        prices = _fetch_bars_before(cur, row["symbol"], row["interval"], row["opened_at"], _HURST_BARS)
        if len(prices) < 50:
            continue
        h = _hurst(np.diff(np.log(prices)))
        if h is None:
            continue
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        records.append({"hurst": h, "fwd_ret": fwd_ret, "opened_at": row["opened_at"]})
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için Hurst + {_FORWARD_BARS}-bar ileri getiri hesaplandı\n")
    if df.empty:
        return

    print(f"Hurst dağılımı: min={df['hurst'].min():.3f} medyan={df['hurst'].median():.3f} max={df['hurst'].max():.3f} std={df['hurst'].std():.3f}")

    print("\n=== [1] Korelasyon (Hurst vs sabit-{}-bar ileri getiri) ===".format(_FORWARD_BARS))
    rho, p = spearmanr(df["hurst"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    tercile = pd.qcut(df["hurst"], 4, labels=["1.en düşük", "2", "3", "4.en yüksek"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["fwd_ret"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print("\n  Hurst dörtte-biri:")
    print(g.to_string())

    print("\n=== [2] Placebo (Hurst karıştırma) ===")
    rng = np.random.default_rng(42)
    vals = df["hurst"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(vals)
        r, _ = spearmanr(shuffled, target)
        if abs(r) >= abs(rho):
            count_ge += 1
    print(f"  gerçek rho={rho:+.4f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    print("\n=== [3] Split-period (çöküş vs toparlanma) ===")
    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            r, p2 = spearmanr(sub["hurst"], sub["fwd_ret"])
            print(f"  {label} (n={len(sub)}): rho={r:+.4f} (p={p2:.4f})")
        else:
            print(f"  {label}: yetersiz örnek (n={len(sub)})")


if __name__ == "__main__":
    main()
