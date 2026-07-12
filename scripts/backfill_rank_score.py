"""signals tablosundaki rank_score/vs_btc NULL olan kapalı sinyalleri
cagg_5m/15m/1h/4h verisinden geriye dönük doldurur.

Formüller desktop/workers/ranking_worker.py::_compute ve ::_vpmv ile birebir
aynıdır; sinyaller 15 dakikalık kovalara gruplanır, her kova için tüm sembol
evreninin combined skoru bir kez hesaplanıp aynı cross-sectional ranking o
kovadaki tüm sinyallere uygulanır."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from indicators.core import calculate_atr, calculate_rsi
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)

_TF_WEIGHTS = {"5m": 0.35, "15m": 0.30, "1h": 0.20, "4h": 0.15}
_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
_TF_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_MIN_BARS = 50
_WARMUP_BARS = 210
_REF_SYMBOL = "BTCUSDT"
_WORKERS = 6


def _connect():
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )


def _vpmv_series(df: pd.DataFrame) -> pd.Series:
    rsi_series = calculate_rsi(df, period=14)
    rsi_centered = rsi_series - 50
    atr_series = calculate_atr(df, period=Config.ATR_PERIOD)
    price_pct = df["close"].pct_change().fillna(0.0) * 100.0
    return (
        normalize_volume_0_100(df["volume"]) * 0.35
        + normalize_momentum_0_100(rsi_centered) * 0.35
        + normalize_volatility_0_100(atr_series) * 0.20
        + normalize_price_0_100(price_pct) * 0.10
    )


def _sample_tf(
    bar_epochs: np.ndarray, values: np.ndarray, bucket_epochs: np.ndarray, interval_s: int
) -> np.ndarray:
    out = np.full(len(bucket_epochs), np.nan)
    if len(bar_epochs) == 0:
        return out
    idx = np.searchsorted(bar_epochs, bucket_epochs - interval_s, side="right") - 1
    valid = idx >= (_MIN_BARS - 1)
    safe_idx = np.clip(idx, 0, len(bar_epochs) - 1)
    valid &= bar_epochs[safe_idx] >= bucket_epochs - 3 * interval_s
    valid &= ~np.isnan(values[safe_idx])
    out[valid] = values[safe_idx[valid]]
    return out


def _compute_symbol(
    symbol: str, bucket_epochs: np.ndarray, first_bucket, last_bucket
) -> np.ndarray:
    conn = _connect()
    try:
        tf_rows = []
        for tf, weight in _TF_WEIGHTS.items():
            start = first_bucket - pd.Timedelta(seconds=_WARMUP_BARS * _TF_SECONDS[tf])
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT bucket, open, high, low, close, volume FROM {_TF_TABLE[tf]} "
                    "WHERE symbol=%s AND bucket >= %s AND bucket <= %s ORDER BY bucket",
                    (symbol, start.to_pydatetime(), last_bucket.to_pydatetime()),
                )
                rows = cur.fetchall()
            if len(rows) < _MIN_BARS:
                tf_rows.append((weight, np.full(len(bucket_epochs), np.nan)))
                continue
            df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close", "volume"])
            try:
                vpmv = np.round(_vpmv_series(df).to_numpy(dtype=float), 1)
            except Exception:
                tf_rows.append((weight, np.full(len(bucket_epochs), np.nan)))
                continue
            bar_epochs = df["bucket"].astype("datetime64[s]").astype(np.int64).to_numpy()
            tf_rows.append((weight, _sample_tf(bar_epochs, vpmv, bucket_epochs, _TF_SECONDS[tf])))
    finally:
        conn.close()

    weights = np.array([w for w, _ in tf_rows])[:, None]
    matrix = np.vstack([v for _, v in tf_rows])
    mask = ~np.isnan(matrix)
    total_w = (weights * mask).sum(axis=0)
    with np.errstate(invalid="ignore"):
        combined = np.where(total_w > 0, np.nansum(matrix * weights, axis=0) / total_w, np.nan)
    return np.round(combined, 1)


def main() -> None:
    t0 = time.time()
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, symbol, opened_at FROM signals "
            "WHERE rank_score IS NULL AND status='closed' ORDER BY opened_at"
        )
        signals = cur.fetchall()
    print(f"{len(signals)} sinyal backfill edilecek")
    if not signals:
        conn.close()
        return

    sig_df = pd.DataFrame(signals, columns=["id", "symbol", "opened_at"])
    sig_df["bucket"] = pd.to_datetime(sig_df["opened_at"]).dt.floor("15min")
    buckets = np.sort(sig_df["bucket"].unique())
    bucket_index = {b: i for i, b in enumerate(buckets)}
    bucket_epochs = buckets.astype("datetime64[s]").astype(np.int64)
    first_bucket = pd.Timestamp(buckets[0])
    last_bucket = pd.Timestamp(buckets[-1])
    print(f"{len(buckets)} kova: {first_bucket} → {last_bucket}")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT symbol FROM cagg_1h WHERE bucket >= %s AND bucket <= %s",
            ((first_bucket - pd.Timedelta(days=2)).to_pydatetime(), last_bucket.to_pydatetime()),
        )
        universe = sorted(r[0] for r in cur.fetchall())
    print(f"{len(universe)} sembol evreni")

    scores: dict[str, np.ndarray] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_compute_symbol, sym, bucket_epochs, first_bucket, last_bucket): sym
            for sym in universe
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                scores[sym] = fut.result()
            except Exception as exc:
                print(f"  {sym} hesaplanamadı: {exc}")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(universe)} sembol ({time.time() - t0:.0f}s)")

    score_matrix = np.vstack([scores[s] for s in sorted(scores)])
    sym_index = {s: i for i, s in enumerate(sorted(scores))}
    btc_row = scores.get(_REF_SYMBOL)

    updates = []
    skipped = 0
    for bucket, grp in sig_df.groupby("bucket"):
        bi = bucket_index[bucket]
        col = score_matrix[:, bi]
        valid = col[~np.isnan(col)]
        n = len(valid)
        if n == 0:
            skipped += len(grp)
            continue
        sorted_vals = np.sort(valid)
        btc_val = btc_row[bi] if btc_row is not None else np.nan
        for sig_id, sym in zip(grp["id"], grp["symbol"]):
            si = sym_index.get(sym)
            own = score_matrix[si, bi] if si is not None else np.nan
            if np.isnan(own):
                skipped += 1
                continue
            rank_score = round(float(np.searchsorted(sorted_vals, own, side="left")) / n * 100, 1)
            vs_btc = round(float(own - btc_val), 1) if not np.isnan(btc_val) else None
            updates.append((int(sig_id), rank_score, vs_btc))

    print(f"{len(updates)} sinyal güncellenecek, {skipped} atlandı (skor hesaplanamadı)")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "UPDATE signals AS s SET rank_score = v.rank_score, vs_btc = v.vs_btc "
            "FROM (VALUES %s) AS v(id, rank_score, vs_btc) WHERE s.id = v.id",
            updates,
            page_size=2000,
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM signals WHERE rank_score IS NOT NULL")
        total_filled = cur.fetchone()[0]
    conn.close()
    print(f"Bitti ({time.time() - t0:.0f}s). rank_score dolu toplam satır: {total_filled}")


if __name__ == "__main__":
    main()
