"""
ME (Kombine Verimlilik) — Pattern Lab doğrulaması.

ME = ΔPrice% / (ΔRSI × RVOL), RVOL = Volume / SMA(Volume, 20) — [[project_devisso_ersi]]'de
"beklemede" bırakılan üçüncü Devisso formülü, ERSI (ΔPrice%/ΔRSI) ve EVOL
(ΔPrice%/RVOL)'un birleşimi. ERSI mutlak değeri PnL'i öngörmüyordu (|r|<0.05,
28.878 sinyal), EVOL(Short) ise 3 kapıyı geçen sağlam bir bulguydu — ME bu
ikisini TEK metrikte birleştirdiği için, bu oturumdaki meta-ders (kompozit
metrikler placebo'ya yapışıyor, tek-başına temiz metrikler sağlam çıkıyor)
burada özellikle geçerli bir uyarı.

evol_bt.py ile BİREBİR AYNI metodoloji (EMA(7) smoothing, son 100 bar
percentile-rank, aynı sorgu/fetch deseni) — sadece payda ΔRSI×RVOL. RSI,
üretimdeki (_compute_devisso_score) ile AYNI fonksiyon: calculate_rsi(period=14).

ME hiçbir yerde hesaplanmıyor/saklanmıyor — bu script bellek içinde hesaplayıp
test ediyor. Kapsam 5m/15m ile sınırlı (_SIGNAL_GENERATION_TFS).

Kullanım: python -m research.pattern_lab.me_bt
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m"}
_BARS_NEEDED = 130
_RVOL_WINDOW = 20
_RSI_PERIOD = 14


def _compute_me(df: pd.DataFrame) -> "float | None":
    """ERSI/EVOL ile birebir aynı normalize/smoothing/rank metodolojisi —
    payda ΔRSI(14) × RVOL(20)."""
    try:
        if len(df) < 30:
            return None
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        rsi = calculate_rsi(df, period=_RSI_PERIOD)
        rvol = volume / volume.rolling(_RVOL_WINDOW).mean()
        price_pct = close.pct_change() * 100.0

        denom = rsi.diff().replace(0.0, np.nan) * rvol.replace(0.0, np.nan)
        raw = price_pct / denom
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 20:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        rank = float((recent < current).sum()) / len(recent)
        return round(rank * 100.0, 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> "pd.DataFrame | None":
    cagg = _CAGG.get(interval)
    if not cagg:
        return None
    cur.execute(f"""
        SELECT bucket AS open_time, open, high, low, close, volume
        FROM {cagg}
        WHERE symbol = %s AND bucket <= %s
        ORDER BY bucket DESC
        LIMIT %s
    """, (symbol, opened_at, _BARS_NEEDED))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    return df.iloc[::-1].reset_index(drop=True)


def _fetch_signals(cur) -> list:
    cur.execute("""
        SELECT id, symbol, interval, opened_at, signal_type, realized_pnl
        FROM signals
        WHERE status='closed' AND realized_pnl IS NOT NULL
          AND interval IN ('5m', '15m')
        ORDER BY symbol, interval, opened_at
    """)
    return cur.fetchall()


def _report_correlation(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["me"], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    signals = _fetch_signals(cur)
    print(f"Toplam kapanmış sinyal (5m/15m): {len(signals)}")

    rows = []
    for i, sig in enumerate(signals, 1):
        bars = _fetch_bars(cur, sig["symbol"], sig["interval"], sig["opened_at"])
        if bars is None or len(bars) < 30:
            continue
        me = _compute_me(bars)
        if me is None:
            continue
        rows.append({
            "symbol": sig["symbol"], "signal_type": sig["signal_type"],
            "interval": sig["interval"], "opened_at": sig["opened_at"],
            "me": me, "realized_pnl": sig["realized_pnl"],
        })
        if i % 5000 == 0:
            print(f"  [{i}/{len(signals)}] işlendi, {len(rows)} geçerli ME")

    conn.close()
    df = pd.DataFrame(rows)
    print(f"\nGeçerli ME hesaplanan sinyal: {len(df)}\n")

    print("=== ANA TEST (ME mutlak değeri vs realized_pnl) ===")
    _report_correlation(df, "gerçek")

    print("\n=== BAND ANALİZİ (<35 / 35-65 / >=65) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].copy()
        sub["band"] = pd.cut(sub["me"], bins=[-1, 35, 65, 101], labels=["dusuk", "orta", "yuksek"])
        g = sub.groupby("band", observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  {sig_type}:\n{g.to_string()}")

    print("\n=== PLACEBO (ME rastgele karıştırıldı) ===")
    rng = np.random.default_rng(42)
    placebo_df = df.copy()
    placebo_df["me"] = rng.permutation(placebo_df["me"].values)
    _report_correlation(placebo_df, "placebo")

    print("\n=== SPLIT-PERIOD ===")
    mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", df[df["opened_at"] < mid]),
        ("ikinci_yari", df[df["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)}) --")
        _report_correlation(half_df, half_name)


if __name__ == "__main__":
    main()
