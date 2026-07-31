"""
ersi_trough_bt.py — evol_trough_bt.py ile AYNI metodoloji, ERSI için.

Motivasyon: ERSI'nin statik seviyesi zaten test edilmiş ve BAŞARISIZ çıkmıştı
(bkz. memory: project_devisso_ersi, 6 Tem 2026 — 28.878 sinyal, |r|<0.05,
bandlar tutarsız, "PnL filtresi olarak kullanılmamalı" sonucu). EVOL'de ise
statik seviye Short'ta sağlam çıkmıştı AMA kullanıcının canlı yarış
grafiklerinde gördüğü asıl örüntü ("dip yapıp dönüş") FARKLI bir test
gerektiriyordu (evol_trough_bt.py) — ve o da gerçek çıktı (1-4h ufukta).

Bu script aynı soruyu ERSI için soruyor: statik seviye başarısız olsa da,
ERSI'nin YEREL MİNİMUM/MAKSİMUM'u (dönüş noktası) ileri getiriyi öngörüyor mu?

ERSI = ΔPrice% / ΔRSI(14) — indicators/core.py::calculate_rsi (canlı kod) ile
hesaplanan RSI serisi üzerinden, signals/signal_processor.py::_compute_devisso_score
ile AYNI EMA(7)+percentile-rank disiplini, tam seri için rolling uygulanıyor.

Kullanım: python -m research.pattern_lab.ersi_trough_bt
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # noqa: E402  pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # noqa: E402  pylint: disable=wrong-import-position

DAYS = 40
MIN_BARS = 700
RANK_WINDOW = 100
CONFIRM_BEFORE = 6
CONFIRM_AFTER = 3
FORWARD_BARS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}
_GAP_HOURS_THRESHOLD = 200
_PLACEBO_SEED = 42


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _bad_symbols(cur) -> set[str]:
    cur.execute(
        """
        WITH gaps AS (
            SELECT symbol, EXTRACT(EPOCH FROM (curr_ts-prev_ts))/3600 AS saat
            FROM (
                SELECT symbol, timestamp AS curr_ts,
                       LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                FROM price_data WHERE interval='1m'
            ) t
            WHERE prev_ts IS NOT NULL
        )
        SELECT DISTINCT symbol FROM gaps WHERE saat > %s
        """,
        (_GAP_HOURS_THRESHOLD,),
    )
    return {r[0] for r in cur.fetchall()}


def _fetch(exclude: set[str]) -> pd.DataFrame:
    conn = _conn()
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if exclude:
        df = df[~df["symbol"].isin(exclude)].reset_index(drop=True)
    return df


def _ersi_series(g: pd.DataFrame) -> np.ndarray:
    """_compute_devisso_score ile AYNI formül — her bar için rolling
    percentile-rank skoru (0-100), tam seri boyunca. RSI canlı kodla
    (indicators/core.py::calculate_rsi) hesaplanıyor."""
    close = g["close"].to_numpy(float)
    rsi = calculate_rsi(g, period=14).to_numpy(float)

    price_pct = np.diff(close, prepend=np.nan) / np.roll(close, 1) * 100.0
    price_pct[0] = np.nan
    rsi_diff = np.diff(rsi, prepend=np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = price_pct / np.where(rsi_diff == 0, np.nan, rsi_diff)

    smoothed = pd.Series(raw).ewm(span=7, adjust=False).mean().to_numpy()

    n = len(smoothed)
    score = np.full(n, np.nan)
    valid_mask = np.isfinite(smoothed)
    valid_idx = np.where(valid_mask)[0]
    valid_vals = smoothed[valid_mask]
    for pos in range(len(valid_vals)):
        if pos < 19:
            continue
        start = max(0, pos - RANK_WINDOW + 1)
        window = valid_vals[start : pos + 1]
        current = valid_vals[pos]
        rank = (window < current).sum() / len(window)
        score[valid_idx[pos]] = round(rank * 100.0, 2)
    return score


def _find_troughs_and_peaks(score: np.ndarray) -> tuple[list[int], list[int]]:
    n = len(score)
    troughs, peaks = [], []
    for i in range(CONFIRM_BEFORE, n - CONFIRM_AFTER):
        window = score[i - CONFIRM_BEFORE : i + 1]
        after = score[i + 1 : i + 1 + CONFIRM_AFTER]
        if np.isnan(window).any() or np.isnan(after).any():
            continue
        if score[i] == window.min() and after[-1] > score[i] and np.all(np.diff(after) >= -1e-9):
            troughs.append(i)
        if score[i] == window.max() and after[-1] < score[i] and np.all(np.diff(after) <= 1e-9):
            peaks.append(i)
    return troughs, peaks


def _forward_returns(c: np.ndarray, idx: int) -> dict[str, float]:
    entry = c[idx]
    out = {}
    for name, bars in FORWARD_BARS.items():
        j = min(idx + bars, len(c) - 1)
        out[name] = (c[j] - entry) / entry * 100.0
    return out


def _stats(vals: np.ndarray) -> dict:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": len(vals),
        "ort_%": round(float(vals.mean()), 3),
        "medyan_%": round(float(np.median(vals)), 3),
        "wr": round(float((vals > 0).mean() * 100), 1),
    }


def run() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    trough_records: list[dict] = []
    peak_records: list[dict] = []
    placebo_records: list[dict] = []
    rng = np.random.default_rng(_PLACEBO_SEED)

    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        ts = g["ts"]
        score = _ersi_series(g)
        troughs, peaks = _find_troughs_and_peaks(score)

        for idx in troughs:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "ersi_at": score[idx]}
            rec.update(_forward_returns(c, idx))
            trough_records.append(rec)

        for idx in peaks:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "ersi_at": score[idx]}
            rec.update(_forward_returns(c, idx))
            peak_records.append(rec)

        valid_range = np.arange(CONFIRM_BEFORE, len(c) - max(FORWARD_BARS.values()))
        if len(valid_range) > 0 and troughs:
            sample = rng.choice(valid_range, size=min(len(troughs), len(valid_range)), replace=False)
            for idx in sample:
                rec = {"symbol": sym, "ts": ts.iloc[idx]}
                rec.update(_forward_returns(c, idx))
                placebo_records.append(rec)

    print(f"[tarama] {n_syms} sembol")
    print(f"Toplam TROUGH (dip) noktası: {len(trough_records)}")
    print(f"Toplam PEAK (tepe) noktası: {len(peak_records)}")
    print(f"Toplam PLACEBO (rastgele) noktası: {len(placebo_records)}\n")

    tdf = pd.DataFrame(trough_records)
    pdf = pd.DataFrame(peak_records)
    plc = pd.DataFrame(placebo_records)

    print("=== TROUGH (ERSI dip yaptı, dönüşe geçti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(tdf[name].to_numpy())}")

    print("\n=== PEAK (ERSI tepe yaptı, düşüşe geçti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(pdf[name].to_numpy())}")

    print("\n=== PLACEBO (rastgele noktalar) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(plc[name].to_numpy())}")

    print("\n=== SPLIT-PERIOD (TROUGH, 24h) ===")
    mid = tdf["ts"].median()
    for label, sub in [("IS", tdf[tdf["ts"] < mid]), ("OOS", tdf[tdf["ts"] >= mid])]:
        print(f"  {label}: {_stats(sub['24h'].to_numpy())}")

    print("\n=== SPLIT-PERIOD (PEAK, 24h) ===")
    mid_p = pdf["ts"].median()
    for label, sub in [("IS", pdf[pdf["ts"] < mid_p]), ("OOS", pdf[pdf["ts"] >= mid_p])]:
        print(f"  {label}: {_stats(sub['24h'].to_numpy())}")

    print("\n=== ERSI değeri troughlarda gerçekten düşük mü? (sağlık kontrolü) ===")
    print(f"  trough'larda ERSI ortalaması: {tdf['ersi_at'].mean():.1f} (düşük olmalı)")
    print(f"  peak'lerde ERSI ortalaması:   {pdf['ersi_at'].mean():.1f} (yüksek olmalı)")


if __name__ == "__main__":
    run()
