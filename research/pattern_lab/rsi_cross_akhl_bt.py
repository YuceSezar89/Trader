"""
A%/K%/H%/L% — C99 BTHN Pine script'inin (Devis'So Traders HTFS) "crossPrice"
tekniği, RSI_Cross'un KENDİ fast/slow RSI farkına uygulanıyor. Kullanıcının
11 Tem 2026'daki gözlemi doğrulandı: kesişim ANI, sinyal barının (i) kendi
Open/Close'u + BİR ÖNCEKİ barın (i-1) gösterge değeri kullanılarak (gelecek
YOK, look-ahead YOK) interpolasyonla tahmin edilebiliyor:

    diff[i] = fast_rsi[i] - slow_rsi[i]           (eşik=0'ı kesen fark)
    t = (0 - diff[i-1]) / (diff[i] - diff[i-1])    (bar içinde kesişim ORANI)
    crossPrice = open[i] + (close[i]-open[i]) * t  (aynı oranın FİYAT karşılığı)

Sonra mum 4 parçaya ayrılıyor (Pine'daki A%/K%/H%/L%, f_pct(base,value)=
100*(value-base)/base ile):
    A% = crossPrice, Open'a göre (kesişimden ÖNCEki hareket)
    K% = Close, crossPrice'a göre (kesişimden SONRA, AYNI mumda)
    H% = High, crossPrice'a göre (kesişim üstü aşım)
    L% = Low,  crossPrice'a göre (kesişim altı aşım)

Veri: `signals` tablosundaki GERÇEK kapanmış RSI_Cross sinyalleri (st_confirmed
dahil, v2-23 ile birlikte gösteriliyor).
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD,
    ROUND_TRIP_FEE,
)
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    INTERVALS,
    _fetch_symbol_history,
    _signal_bar_ts,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

MIN_DIFF = 1e-9


def _pct(base: float, value: float) -> float:
    if not np.isfinite(base) or abs(base) <= MIN_DIFF:
        return np.nan
    return 100.0 * (value - base) / base


def _akhl(
    direction: str, o: float, h: float, l: float, c: float, diff_prev: float, diff_now: float
) -> tuple:
    denom = diff_now - diff_prev
    if abs(denom) <= MIN_DIFF:
        return (np.nan,) * 4
    t = (0.0 - diff_prev) / denom
    if not (0.0 <= t <= 1.0):
        return (np.nan,) * 4
    cross_price = o + (c - o) * t

    a_pct = _pct(o, cross_price)
    k_pct = _pct(cross_price, c)
    if direction == "Long":
        h_pct = _pct(cross_price, h)
        l_pct = _pct(l, cross_price)
    else:
        h_pct = _pct(h, cross_price)
        l_pct = _pct(cross_price, l)
    return a_pct, k_pct, h_pct, l_pct


def _fetch_signals(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, st_confirmed, realized_pnl, opened_at
        FROM signals
        WHERE indicators LIKE '%%RSI_Cross%%'
          AND interval = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(interval,))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


def _dollar_stats(df: pd.DataFrame, days_span: float) -> dict:
    if len(df) == 0 or days_span <= 0:
        return {"n": 0}
    pnl = df["realized_pnl"].to_numpy() / 100 * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    return {
        "n": len(pnl),
        "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(total / days_span * 30, 1),
    }


def _tercile_report(df: pd.DataFrame, col: str) -> None:
    valid = df.dropna(subset=[col])
    if len(valid) < 60:
        print(f"  {col}: örneklem çok küçük")
        return
    q1, q2 = valid[col].quantile([0.333, 0.667])
    print(f"  {col} (q1={q1:.2f}, q2={q2:.2f})")
    for name, mask in (
        ("düşük", valid[col] < q1),
        ("orta", (valid[col] >= q1) & (valid[col] < q2)),
        ("yüksek", valid[col] >= q2),
    ):
        sub = valid[mask]
        s = _pf(sub)
        print(
            f"    {name:8} n={s.get('n',0):>6} WR%={s.get('wr',0):>6} "
            f"ort%={s.get('ort_%',0):>7} PF={s.get('pf',0):>7}"
        )


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if hist.empty:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            fast = calculate_rsi(hist, period=Config.RSI_FAST_WINDOW).to_numpy()
            slow = calculate_rsi(hist, period=Config.RSI_SLOW_WINDOW).to_numpy()
            diff = fast - slow
            o = hist["open"].to_numpy(float)
            h = hist["high"].to_numpy(float)
            l = hist["low"].to_numpy(float)
            c = hist["close"].to_numpy(float)

            for _, row in sub.iterrows():
                i = ts_to_idx.get(_signal_bar_ts(row["opened_at"], interval))
                if (
                    i is None
                    or i - 1 < 0
                    or not (np.isfinite(diff[i - 1]) and np.isfinite(diff[i]))
                ):
                    continue
                direction = row["signal_type"]
                a_pct, k_pct, h_pct, l_pct = _akhl(
                    direction, o[i], h[i], l[i], c[i], diff[i - 1], diff[i]
                )
                if not all(np.isfinite(v) for v in (a_pct, k_pct, h_pct, l_pct)):
                    continue
                rows.append(
                    {
                        "a_pct": a_pct,
                        "k_pct": k_pct,
                        "h_pct": h_pct,
                        "l_pct": l_pct,
                        "st_confirmed": row["st_confirmed"],
                        "realized_pnl": row["realized_pnl"],
                        "opened_at": row["opened_at"],
                    }
                )

    df = pd.DataFrame(rows)
    print(f"\ntoplam eşleşen (A/K/H/L hesaplanabilen) sinyal: {len(df):,}\n")
    if len(df) < 200:
        print("Örneklem çok küçük.")
        return

    baseline = _pf(df)
    print(
        f"baseline (tümü): n={baseline.get('n',0)} WR%={baseline.get('wr',0)} "
        f"ort%={baseline.get('ort_%',0)} PF={baseline.get('pf',0)}\n"
    )

    print("── Terciller (tüm sinyaller) ──")
    for col in ("a_pct", "k_pct", "h_pct", "l_pct"):
        _tercile_report(df, col)
        print()

    only_true = df[df["st_confirmed"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    print(f"── SADECE st_confirmed=True içinde (n={len(only_true)}) ──")
    for col in ("a_pct", "k_pct", "h_pct", "l_pct"):
        _tercile_report(only_true, col)
        print()

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"── Ekonomik etki (${POSITION_USD:.0f} pozisyon, {days_span:.1f} gün) ──")
    s = _dollar_stats(df, days_span)
    print(f"baseline: n={s.get('n',0)} $/ay={s.get('usd_per_month',0)}")
    s = _dollar_stats(only_true, days_span)
    print(f"st_confirmed=True: n={s.get('n',0)} $/ay={s.get('usd_per_month',0)}")


if __name__ == "__main__":
    run()
