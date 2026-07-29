"""
"12H MA with Live Angle" testi (20 Tem 2026, kullanıcının paylaştığı Pine
indikatöründen — /Users/yusuf/Downloads/12H MA with Live Angle.txt).

Mantık: Heikin Ashi kapanışından 12 saatlik EMA(20) hesaplanıyor. Bu MA'nın
bar-bar momentum'u (%) sıfırı KESTİĞİ an LONG/SHORT sinyali. Açı =
atan(momentum/10)×180/π — sinyal anındaki bu açının büyüklüğü, ileriki
fiyat hareketiyle ilişkili mi test ediliyor.

BASİTLEŞTİRME (şeffaf): orijinal Pine'daki "bir periyot GECİKMELİ alarm"
(önceki periyodun kesinleşmiş açısı eşiği geçti mi) yerine, sinyal anındaki
açının kendisini sürekli bir öngörücü olarak test ediyoruz — daha basit,
daha doğrudan test edilebilir, ama tam olarak aynı mekanizma değil.

Kullanım: python -m research.pattern_lab.ma12h_angle_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_MA_LENGTH = 20
_MIN_1H_BARS = 1800  # ~75 gün, 12H resample sonrası yeterli EMA warmup icin
_MAX_SYMBOLS = 250
_HORIZON_PERIODS = 4  # sinyalden sonraki 4x12H = 2 gün


def _fetch_symbols(cur) -> list[str]:
    cur.execute(
        """
        SELECT symbol FROM cagg_1h
        GROUP BY symbol HAVING count(*) >= %s
        ORDER BY count(*) DESC LIMIT %s
        """,
        (_MIN_1H_BARS, _MAX_SYMBOLS),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_1h(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT bucket, open, high, low, close FROM cagg_1h
        WHERE symbol = %s ORDER BY bucket ASC
        """,
        (symbol,),
    )
    rows = cur.fetchall()
    if len(rows) < _MIN_1H_BARS:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    df = df.set_index("bucket")
    return df


def _resample_12h(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.resample("12h", origin="2000-01-01").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna()


def _heikin_ashi_close(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    return pd.Series((o + h + l + c) / 4.0, index=df.index)


def _collect(cur, symbol: str) -> pd.DataFrame:
    df1h = _fetch_1h(cur, symbol)
    if df1h.empty:
        return pd.DataFrame()
    df12 = _resample_12h(df1h)
    if len(df12) < _MA_LENGTH + _HORIZON_PERIODS + 10:
        return pd.DataFrame()

    ha_close = _heikin_ashi_close(df12)
    ma = ha_close.ewm(span=_MA_LENGTH, adjust=False).mean()
    momentum = ma.pct_change() * 100.0
    angle = np.degrees(np.arctan(momentum / 10.0))

    long_sig = (momentum.shift(1) <= 0) & (momentum > 0)
    short_sig = (momentum.shift(1) >= 0) & (momentum < 0)

    fwd_close = df12["close"].shift(-_HORIZON_PERIODS)
    fwd_ret_long = (fwd_close - df12["close"]) / df12["close"] * 100.0
    fwd_ret_short = -fwd_ret_long

    records = []
    n = len(df12)
    for i in range(n - _HORIZON_PERIODS):
        if long_sig.iloc[i]:
            records.append(
                {
                    "symbol": symbol,
                    "bucket": df12.index[i],
                    "direction": "long",
                    "angle": float(angle.iloc[i]),
                    "fwd_ret": float(fwd_ret_long.iloc[i]),
                }
            )
        elif short_sig.iloc[i]:
            records.append(
                {
                    "symbol": symbol,
                    "bucket": df12.index[i],
                    "direction": "short",
                    "angle": float(-angle.iloc[i]),  # yön-ayarlı: pozitif=lehine
                    "fwd_ret": float(fwd_ret_short.iloc[i]),
                }
            )
    return pd.DataFrame(records)


def _stats(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    g, b = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets),
        "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / b), 3) if b > 0 else float("inf"),
    }


def _placebo(df: pd.DataFrame, n_iter: int = 300) -> float:
    """Sinyal ANLARI sabit, yön etiketi (long/short) rastgele karıştırılır."""
    raw = np.where(df["direction"].to_numpy() == "long", df["fwd_ret"], -df["fwd_ret"])
    real_mean = df["fwd_ret"].mean()
    n = len(df)
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_iter):
        fake_long = rng.random(n) < 0.5
        fake_signed = np.where(fake_long, raw, -raw)
        if abs(fake_signed.mean()) >= abs(real_mean):
            count_ge += 1
    return count_ge / n_iter


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    symbols = _fetch_symbols(cur)
    print(f"[fetch] {len(symbols)} sembol (>= {_MIN_1H_BARS} 1h bar)")

    pieces = []
    for i, sym in enumerate(symbols, 1):
        out = _collect(cur, sym)
        if not out.empty:
            pieces.append(out)
        if i % 50 == 0:
            print(f"  ... {i}/{len(symbols)}")
    conn.close()

    if not pieces:
        print("Hiç sinyal bulunamadı.")
        return
    df = pd.concat(pieces, ignore_index=True)
    print(f"\n[collect] {len(df)} sinyal ({df['symbol'].nunique()} sembolde)\n")

    print("=== 1) Genel sonuç (yön-ayarlı getiri) ===")
    print(f"  {_stats(df['fwd_ret'].to_numpy())}")

    print("\n=== 2) Yöne göre ayrı ===")
    for d in ("long", "short"):
        sub = df[df["direction"] == d]
        print(f"  {d}: {_stats(sub['fwd_ret'].to_numpy())}")

    print("\n=== 3) Açı büyüklüğü ile getiri korelasyonu ===")
    rho, p = spearmanr(df["angle"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f}), n={len(df)}")
    tercile = pd.qcut(df["angle"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["fwd_ret"].agg(ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    print("\n=== 4) Split-period (kronolojik ikiye böl) ===")
    df_sorted = df.sort_values("bucket")
    mid = df_sorted["bucket"].iloc[len(df_sorted) // 2]
    first = df_sorted[df_sorted["bucket"] < mid]
    second = df_sorted[df_sorted["bucket"] >= mid]
    print(f"  ilk yarı:    {_stats(first['fwd_ret'].to_numpy())}")
    print(f"  ikinci yarı: {_stats(second['fwd_ret'].to_numpy())}")

    print("\n=== 5) Placebo (yön karıştırma, 300 iterasyon) ===")
    p_val = _placebo(df)
    print(f"  gerçek ort. getiri ({df['fwd_ret'].mean():+.3f}%) rastgelede %{p_val*100:.1f} sıklıkta çıktı")


if __name__ == "__main__":
    main()
