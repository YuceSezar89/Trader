"""
v2-1c — Hacim sönmesi × Destek/Direnç birleşik testi.

İzole hacim sönmesi (vol_exhaustion_bt.py) hiçbir edge göstermedi. Trader
ekranlarındaki her örnekte yatay destek/direnç çizgileri vardı — hipotez
muhtemelen BİRLEŞİK: "hacim söndü VE fiyat önceki bir desteği test ediyor".

Destek/direnç tanımı (ÖNCEDEN BEYAN, sonradan ayarlanmadı):
  support(t)    = min(low)  [t-192, t-12] bar aralığında (mevcut dibin
                  kendini referans almasını önlemek için 12 bar/3h tampon)
  resistance(t) = max(high) aynı pencerede
  near_support(t)    = |close-support|/support   <= 2.0%
  near_resistance(t) = |resistance-close|/close  <= 2.0%

Gruplar: baseline, COLD, near_support, COLD∧near_support (asıl hipotez) —
ve ayna: HOT, near_resistance, HOT∧near_resistance.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from config import Config
from research.pattern_lab.vol_exhaustion_bt import (
    COLD_LEVEL, DAYS, HORIZONS_BARS, HOT_LEVEL, MIN_BARS, _events, _fwd_returns, _stats, _vol_rank,
)

SR_WINDOW = 180
SR_BUFFER = 12
SR_THRESHOLD_PCT = 2.0


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, close, high, low, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _sr_flags(low: np.ndarray, high: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = pd.Series(low).rolling(SR_WINDOW).min().shift(SR_BUFFER).to_numpy()
    resistance = pd.Series(high).rolling(SR_WINDOW).max().shift(SR_BUFFER).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_s = np.abs(close - support) / support * 100
        dist_r = np.abs(resistance - close) / close * 100
    return dist_s <= SR_THRESHOLD_PCT, dist_r <= SR_THRESHOLD_PCT


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    groups = ["baseline", "cold", "near_support", "cold_x_support",
              "hot", "near_resistance", "hot_x_resistance"]
    fwd = {h: {g: [] for g in groups} for h in HORIZONS_BARS}
    n_syms = 0
    n_events = {g: 0 for g in groups}

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        close = g["close"].to_numpy(float)
        high = g["high"].to_numpy(float)
        low = g["low"].to_numpy(float)
        rank = _vol_rank(g["volume"].to_numpy(float))
        near_support, near_resistance = _sr_flags(low, high, close)
        cold_idx, hot_idx = _events(rank)

        max_h = max(HORIZONS_BARS.values())
        all_idx = list(range(SR_WINDOW + SR_BUFFER, len(close) - max_h, 4))
        support_idx = [i for i in all_idx if near_support[i]]
        resistance_idx = [i for i in all_idx if near_resistance[i]]
        cold_support_idx = [i for i in cold_idx if i < len(near_support) and near_support[i]]
        hot_resistance_idx = [i for i in hot_idx if i < len(near_resistance) and near_resistance[i]]

        idx_map = {
            "baseline": all_idx, "cold": cold_idx, "near_support": support_idx,
            "cold_x_support": cold_support_idx, "hot": hot_idx,
            "near_resistance": resistance_idx, "hot_x_resistance": hot_resistance_idx,
        }
        for gname, idxs in idx_map.items():
            n_events[gname] += len(idxs)
            for h, bars in HORIZONS_BARS.items():
                fwd[h][gname].append(_fwd_returns(close, idxs, bars))

    print(f"analize giren sembol: {n_syms}")
    print("olay sayıları:", n_events, "\n")

    labels = {
        "baseline": "baseline (tüm barlar)", "cold": "COLD (tam sönme)",
        "near_support": "near_support (yalnız)", "cold_x_support": "COLD ∧ near_support (HİPOTEZ)",
        "hot": "HOT (yalnız)", "near_resistance": "near_resistance (yalnız)",
        "hot_x_resistance": "HOT ∧ near_resistance (ayna)",
    }
    for h in HORIZONS_BARS:
        print(f"── ufuk: {h} ──")
        print(f"{'grup':32} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for gname in groups:
            rets = np.concatenate(fwd[h][gname]) if fwd[h][gname] else np.array([])
            s = _stats(rets)
            print(f"{labels[gname]:32} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
