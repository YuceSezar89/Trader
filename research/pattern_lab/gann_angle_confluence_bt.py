"""
Gann Açısı Üçlü Teyit testi (20 Tem 2026, kullanıcının paylaştığı
"GANN TEORY" Pine indikatöründen — /Users/yusuf/Downloads/Açılar.txt).

Fikir: fiyatın ATR'ye göre normalize edilmiş eğimi (derece cinsinden,
atan(Δfiyat/ATR)×180/π) üç farklı periyotta (28/35/50 bar) hesaplanıyor.
Üçü de aynı anda eşik (±20°) üstü/altına geçtiği AN ("üçlü teyit tetiklendi")
bir sinyal sayılabilir — Madde 4/13-state konsensüs ile aynı "bağımsız
ölçümler hemfikir mi" felsefesi.

Bar-bar TÜM sembollerde tarama (mevcut sinyal sistemine bağlı değil, kendi
başına bir tetikleyici olarak test ediliyor — bkz. feedback_test_designed_role,
bu indikatör de kendi başına tetikleyici olarak tasarlanmış, arka plan
boyaması bunu gösteriyor).

Kullanım: python -m research.pattern_lab.gann_angle_confluence_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_PERIODS = (28, 35, 50)
_THRESHOLD = 20.0
_HORIZON_BARS = 24  # 1h barda ~1 gün
_MIN_BARS = 400
_MAX_SYMBOLS = 250


def _fetch_symbols(cur) -> list[str]:
    cur.execute(
        """
        SELECT symbol FROM cagg_1h
        GROUP BY symbol HAVING count(*) >= %s
        ORDER BY count(*) DESC LIMIT %s
        """,
        (_MIN_BARS, _MAX_SYMBOLS),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_klines(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT bucket, high, low, close FROM cagg_1h
        WHERE symbol = %s ORDER BY bucket ASC
        """,
        (symbol,),
    )
    rows = cur.fetchall()
    if len(rows) < _MIN_BARS:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["bucket", "high", "low", "close"])


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _angle(df: pd.DataFrame, period: int) -> pd.Series:
    atr = _atr(df, period).clip(lower=1e-12)
    delta = df["close"] - df["close"].shift(period)
    return np.degrees(np.arctan(delta / atr))


def _collect(cur, symbol: str) -> pd.DataFrame:
    df = _fetch_klines(cur, symbol)
    if df.empty:
        return df

    angles = {p: _angle(df, p) for p in _PERIODS}
    bull = pd.concat([angles[p] > _THRESHOLD for p in _PERIODS], axis=1).all(axis=1)
    bear = pd.concat([angles[p] < -_THRESHOLD for p in _PERIODS], axis=1).all(axis=1)

    # Tetik: önceki bar'da tam teyit YOKTU, bu bar'da VAR (sürekli aynı
    # state'i tekrar tekrar saymamak icin)
    bull_trigger = bull & ~bull.shift(1, fill_value=False)
    bear_trigger = bear & ~bear.shift(1, fill_value=False)

    fwd_close = df["close"].shift(-_HORIZON_BARS)
    fwd_ret_long = (fwd_close - df["close"]) / df["close"] * 100.0
    fwd_ret_short = -fwd_ret_long

    # "Uyum sayısı" — korelasyon kapısı için: kaç periyot eşik üstü/altı
    align_count = pd.concat([angles[p] > _THRESHOLD for p in _PERIODS], axis=1).sum(axis=1) - pd.concat(
        [angles[p] < -_THRESHOLD for p in _PERIODS], axis=1
    ).sum(axis=1)

    records = []
    n = len(df)
    for i in range(n - _HORIZON_BARS):
        if bull_trigger.iloc[i]:
            records.append(
                {
                    "symbol": symbol,
                    "bucket": df["bucket"].iloc[i],
                    "direction": "bull",
                    "fwd_ret": fwd_ret_long.iloc[i],
                    "align_count": int(align_count.iloc[i]),
                }
            )
        elif bear_trigger.iloc[i]:
            records.append(
                {
                    "symbol": symbol,
                    "bucket": df["bucket"].iloc[i],
                    "direction": "bear",
                    "fwd_ret": fwd_ret_short.iloc[i],
                    "align_count": int(-align_count.iloc[i]),
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
    """Tetik ANLARI (zaman noktaları) SABİT tutulur, ama bull/bear YÖN
    etiketi rastgele karıştırılır — gerçek yön ataması, rastgele yön
    atamasından daha mı iyi getiri üretiyor sorusunu test eder. Ham
    (yönsüz) getiri her satır için |fwd_ret_raw| olarak saklanmış olmalı;
    burada elimizde sadece yön-ayarlı fwd_ret var, o yüzden raw'ı
    fwd_ret * sign(1 if bull else -1) ile geri türetiyoruz."""
    raw = np.where(df["direction"].to_numpy() == "bull", df["fwd_ret"], -df["fwd_ret"])
    real_mean = df["fwd_ret"].mean()
    n = len(df)
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_iter):
        fake_bull = rng.random(n) < 0.5
        fake_signed = np.where(fake_bull, raw, -raw)
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
    print(f"[fetch] {len(symbols)} sembol (>= {_MIN_BARS} bar, 1h)")

    pieces = []
    for i, sym in enumerate(symbols, 1):
        out = _collect(cur, sym)
        if not out.empty:
            pieces.append(out)
        if i % 50 == 0:
            print(f"  ... {i}/{len(symbols)}")
    conn.close()

    if not pieces:
        print("Hiç tetik bulunamadı.")
        return
    df = pd.concat(pieces, ignore_index=True)
    print(f"\n[collect] {len(df)} tetik olayı ({df['symbol'].nunique()} sembolde)\n")

    print("=== 1) Genel sonuç (yön-ayarlı getiri, tüm tetikler) ===")
    print(f"  {_stats(df['fwd_ret'].to_numpy())}")

    print("\n=== 2) Yöne göre ayrı ===")
    for d in ("bull", "bear"):
        sub = df[df["direction"] == d]
        print(f"  {d}: {_stats(sub['fwd_ret'].to_numpy())}")

    print("\n=== 3) Uyum-sayısı ile getiri korelasyonu (align_count: 1/2/3 periyot hemfikir) ===")
    rho, p = spearmanr(df["align_count"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f}), n={len(df)}")
    for cnt, g in df.groupby("align_count"):
        print(f"    align_count={cnt}: n={len(g)}, ort_%={g['fwd_ret'].mean():.3f}, wr={(g['fwd_ret']>0).mean()*100:.1f}")

    print("\n=== 4) Split-period (kronolojik ikiye böl) ===")
    df_sorted = df.sort_values("bucket")
    mid = df_sorted["bucket"].iloc[len(df_sorted) // 2]
    first = df_sorted[df_sorted["bucket"] < mid]
    second = df_sorted[df_sorted["bucket"] >= mid]
    print(f"  ilk yarı:    {_stats(first['fwd_ret'].to_numpy())}")
    print(f"  ikinci yarı: {_stats(second['fwd_ret'].to_numpy())}")

    print("\n=== 5) Placebo (rastgele örnekleme, 300 iterasyon) ===")
    p_val = _placebo(df)
    print(f"  gerçek ort. getiri ({df['fwd_ret'].mean():+.3f}%) rastgelede %{p_val*100:.1f} sıklıkta çıktı")


if __name__ == "__main__":
    main()
