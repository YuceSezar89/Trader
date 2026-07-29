"""
Hurst Exponent — pencere boyutu duyarlılık taraması (21 Tem 2026).

hurst_reversal_noise_bt.py'nin (200 bar) rho=-0.005, placebo p=%77.7 ile
tamamen başarısız çıkmasının ardından: "pencere boyutu değişirse ne olur"
sorusu için — aynı SABİT örneklem üzerinde 4 farklı pencere boyutuyla
(100/200/400/800 bar) Hurst yeniden hesaplanır, dağılım genişliği +
korelasyon karşılaştırılır. Hız için örneklem küçültüldü (2000).

Kullanım: python -m research.pattern_lab.hurst_window_sweep_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_WINDOWS = (100, 200, 400, 800)
_MAX_SIGNALS = 2000
_MIN_WINDOW = 8


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
        SELECT id, symbol, interval, opened_at, closed_at, realized_pnl
        FROM signals
        WHERE status='closed' AND close_reason='reversal' AND realized_pnl IS NOT NULL
          AND interval IN ('5m','15m','1h')
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    cols = ["id", "symbol", "interval", "opened_at", "closed_at", "realized_pnl"]
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


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} reversal-kapanışı örneklendi (sabit örneklem, tüm pencerelerde AYNI)")

    max_bars_needed = max(_WINDOWS)
    # Her sinyal için EN BÜYÜK pencerelik fiyatı bir kez çek, küçük pencereler için son N'ini kullan
    signal_prices: dict[int, np.ndarray] = {}
    for i, row in signals.iterrows():
        prices = _fetch_bars_before(cur, row["symbol"], row["interval"], row["opened_at"], max_bars_needed)
        if len(prices) >= 50:
            signal_prices[row["id"]] = prices
        if (i + 1) % 500 == 0:
            print(f"  ... fiyat çekme {i+1}/{len(signals)}")
    conn.close()
    print(f"[collect] {len(signal_prices)} sinyal için yeterli fiyat verisi bulundu\n")

    for w in _WINDOWS:
        records = []
        for _, row in signals.iterrows():
            prices = signal_prices.get(row["id"])
            if prices is None or len(prices) < w // 2:
                continue
            sub_prices = prices[-w:] if len(prices) >= w else prices
            returns = np.diff(np.log(sub_prices))
            h = _hurst(returns)
            if h is None:
                continue
            records.append({"hurst": h, "realized_pnl": row["realized_pnl"]})

        df = pd.DataFrame(records)
        if df.empty:
            print(f"pencere={w:>4} bar: veri yok")
            continue
        rho, p = spearmanr(df["hurst"], df["realized_pnl"])
        print(f"pencere={w:>4} bar | n={len(df):>5} | Hurst: min={df['hurst'].min():.3f} "
              f"medyan={df['hurst'].median():.3f} max={df['hurst'].max():.3f} std={df['hurst'].std():.3f} "
              f"| rho={rho:+.4f} (p={p:.4f})")


if __name__ == "__main__":
    main()
