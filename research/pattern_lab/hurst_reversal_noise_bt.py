"""
Hurst Exponent — hızlı-reversal gürültüsünü öngörüyor mu? (21 Tem 2026)

Hipotez: sinyal açılış anında piyasa "mean-reverting" (H<0.45) durumdaysa,
bu sinyal daha yüksek ihtimalle hızlı-reversal (gürültü/whipsaw) ile kapanır.
R/S (Rescaled Range) analizi, SADECE opened_at'a kadarki geçmiş veriden
hesaplanıyor — look-ahead yok.

Veri kısıtı notu (kullanıcı ile netleşti, 21 Tem): elimizdeki dönem çöküş
(Nis-Haz 2026) + toparlanma (Haz-Tem 2026) rejimlerini kapsıyor ama GERÇEK
sürdürülebilir boğa piyasası verisi yok — split-period testi bu iki rejim
arasında tutarlılığı kontrol eder, ama boğa rejiminde nasıl davranacağı
bilinmiyor (bilerek kabul edilen kısıt).

4 kapı: korelasyon+tercile (realized_pnl VE duration_min), placebo,
split-period (çöküş vs toparlanma dönemi).

Kullanım: python -m research.pattern_lab.hurst_reversal_noise_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_HURST_BARS = 200
_MAX_SIGNALS = 6000
_MIN_WINDOW = 8
_PLACEBO_ITER = 300
# Rejim sınırı: 1 Haziran 2026 çöküş haftasının başlangıcı — öncesi "çöküş
# öncesi+içi", sonrası "toparlanma" olarak kabaca ayrılıyor.
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _hurst(prices: np.ndarray) -> float | None:
    returns = np.diff(np.log(prices))
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
        SELECT id, symbol, interval, signal_type, opened_at, closed_at, realized_pnl
        FROM signals
        WHERE status='closed' AND close_reason='reversal' AND realized_pnl IS NOT NULL
          AND interval IN ('5m','15m','1h')
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "closed_at", "realized_pnl"]
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
    print(f"[fetch] {len(signals)} reversal-kapanışı örneklendi (rastgele, maks {_MAX_SIGNALS})")

    records = []
    for i, row in signals.iterrows():
        prices = _fetch_bars_before(cur, row["symbol"], row["interval"], row["opened_at"], _HURST_BARS)
        if len(prices) < 50:
            continue
        h = _hurst(prices)
        if h is None:
            continue
        duration_min = (row["closed_at"] - row["opened_at"]).total_seconds() / 60.0
        records.append({
            "hurst": h, "realized_pnl": row["realized_pnl"], "duration_min": duration_min,
            "opened_at": row["opened_at"],
        })
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için Hurst hesaplandı\n")
    if df.empty:
        return

    print(f"Hurst dağılımı: min={df['hurst'].min():.3f} medyan={df['hurst'].median():.3f} max={df['hurst'].max():.3f}")

    print("\n=== [1] Korelasyon ===")
    rho_pnl, p_pnl = spearmanr(df["hurst"], df["realized_pnl"])
    rho_dur, p_dur = spearmanr(df["hurst"], df["duration_min"])
    print(f"  Hurst vs realized_pnl: rho={rho_pnl:+.3f} (p={p_pnl:.4f})")
    print(f"  Hurst vs duration_min: rho={rho_dur:+.3f} (p={p_dur:.4f})  (pozitifse: yüksek Hurst -> daha uzun süre reversal'sız kalıyor)")

    tercile = pd.qcut(df["hurst"], 4, labels=["1.en düşük(H<)", "2", "3", "4.en yüksek(H>)"], duplicates="drop")
    g = df.groupby(tercile, observed=True).agg(
        n=("realized_pnl", "count"),
        ort_pnl=("realized_pnl", "mean"),
        wr=("realized_pnl", lambda s: (s > 0).mean() * 100),
        ort_sure_dk=("duration_min", "mean"),
    )
    print("\n  Hurst dörtte-biri:")
    print(g.to_string())

    print("\n=== [2] Placebo (Hurst karıştırma) ===")
    rng = np.random.default_rng(42)
    vals = df["hurst"].to_numpy()
    target = df["realized_pnl"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(vals)
        rho, _ = spearmanr(shuffled, target)
        if abs(rho) >= abs(rho_pnl):
            count_ge += 1
    print(f"  gerçek rho={rho_pnl:+.3f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    print("\n=== [3] Split-period (çöküş vs toparlanma rejimi) ===")
    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    if len(crash) >= 100:
        r1, p1 = spearmanr(crash["hurst"], crash["realized_pnl"])
        print(f"  Çöküş dönemi (n={len(crash)}): rho={r1:+.3f} (p={p1:.4f})")
    else:
        print(f"  Çöküş dönemi: yetersiz örnek (n={len(crash)})")
    if len(recovery) >= 100:
        r2, p2 = spearmanr(recovery["hurst"], recovery["realized_pnl"])
        print(f"  Toparlanma dönemi (n={len(recovery)}): rho={r2:+.3f} (p={p2:.4f})")
    else:
        print(f"  Toparlanma dönemi: yetersiz örnek (n={len(recovery)})")

    print("\n=== [4] Filtre simülasyonu — düşük Hurst'ü eleyip kalanın WR/PF'i ===")
    for thr in (0.40, 0.45, 0.50, 0.55):
        sub = df[df["hurst"] >= thr]
        s = _stats(sub["realized_pnl"].to_numpy())
        elenen = len(df) - len(sub)
        print(f"  H>={thr:.2f}: kalan n={s.get('n',0):>5} (elenen {elenen:>5}) | "
              f"WR={s.get('wr',0):>5.1f}% ort%={s.get('ort_%',0):>7.3f} PF={s.get('pf',0):>6.3f}")


if __name__ == "__main__":
    main()
