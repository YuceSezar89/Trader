"""
do_open_streak'in GAUSS_THRESHOLD'unu (mevcut: 4.5) düşürmenin etkisini test
eder. Motivasyon (12 Tem 2026, TRADOORUSDT): 07:00 barında gate açıkken
long_perc=%2.52, gauss=4.44 — eşiğin (4.5) hemen altında kaldığı için ıskalandı.
Gate mantığına DOKUNULMUYOR (do_break_or_lift_bt.py'de test edilip reddedildi),
SADECE gauss eşiği değişiyor.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from signals.do_open_streak import (  # pylint: disable=wrong-import-position
    MAX_HOLD_HOURS, SL_ATR_MULT, STREAK_THRESHOLD, _gauss_sum,
)

DAYS = 60
MIN_BARS = 250
WARMUP = 200
HORIZON_BARS = int(MAX_HOLD_HOURS * 4)
N_PLACEBO = 200
GAUSS_THRESHOLDS_TO_TEST = [4.5, 4.44, 4.0, 3.0]


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _signal_series(o, h, l, c, do, gauss_threshold: float) -> np.ndarray:
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break = (c > do) & (prev_c <= do) & np.isfinite(do)
    is_long = c > o

    gate_active = False
    count_long = 0
    start_low = np.nan
    signal = np.zeros(n, dtype=bool)
    for i in range(n):
        if do_break[i]:
            gate_active = True
        elif not is_long[i]:
            gate_active = False

        if is_long[i]:
            count_long += 1
            if count_long == 1 or np.isnan(start_low):
                start_low = l[i]
        else:
            count_long = 0
            start_low = np.nan

        if gate_active and count_long == STREAK_THRESHOLD:
            long_perc = (h[i] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val >= gauss_threshold:
                signal[i] = True
    return signal


def _simulate(low, close, entry_idx, entry_price, sl, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
    return close[last_i] / entry_price - 1


def _collect_events(df: pd.DataFrame, gauss_threshold: float) -> tuple:
    rets, opened_ats = [], []
    n_syms, n_events = 0, 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        ts_np = ts.to_numpy()

        do, _ = _daily_open(ts, o)
        signal = _signal_series(o, h, l, c, do, gauss_threshold)
        atr_series = calculate_atr(g, period=14).to_numpy()

        idxs = np.where(signal)[0]
        idxs = idxs[(idxs >= WARMUP) & (idxs < len(g) - HORIZON_BARS)]

        for i in idxs:
            atr = atr_series[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = c[i]
            sl = entry - SL_ATR_MULT * atr
            n_events += 1
            rets.append(_simulate(l, c, i, entry, sl, HORIZON_BARS))
            opened_ats.append(pd.Timestamp(ts_np[i]))

    return rets, opened_ats, n_syms, n_events


def _placebo_for_variant(df: pd.DataFrame, gauss_threshold: float, real_pf: float) -> None:
    rng = np.random.default_rng(42)
    per_symbol_meta = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        do, _ = _daily_open(ts, o)
        signal = _signal_series(o, h, l, c, do, gauss_threshold)
        atr_series = calculate_atr(g, period=14).to_numpy()
        idxs = np.where(signal)[0]
        idxs = idxs[(idxs >= WARMUP) & (idxs < len(g) - HORIZON_BARS)]
        n_ev = len(idxs)
        if n_ev == 0:
            continue
        valid_range = np.arange(WARMUP, len(g) - HORIZON_BARS)
        per_symbol_meta.append((l, c, atr_series, valid_range, n_ev))

    if not per_symbol_meta:
        print("placebo: örnek yok")
        return

    placebo_pfs = []
    for _ in range(N_PLACEBO):
        rets = []
        for l, c, atr_series, valid_range, n_ev in per_symbol_meta:
            picks = rng.choice(valid_range, size=n_ev, replace=False)
            for i in picks:
                atr = atr_series[i]
                if not np.isfinite(atr) or atr <= 0:
                    continue
                entry = c[i]
                sl = entry - SL_ATR_MULT * atr
                rets.append(_simulate(l, c, i, entry, sl, HORIZON_BARS))
        if len(rets) >= 20:
            s = _stats(np.array(rets))
            placebo_pfs.append(s.get("pf", 0.0))

    if not placebo_pfs:
        print("placebo: yetersiz örnek")
        return
    arr = np.array(placebo_pfs)
    rank = float((arr < real_pf).mean() * 100)
    print(f"placebo (n={len(arr)}) PF ort={arr.mean():.3f} p90={np.percentile(arr,90):.3f} "
          f"max={arr.max():.3f} | gerçek PF={real_pf:.3f} (placebo'nun %{rank:.0f}'ini geçiyor)")


def run() -> None:
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    print(f"{'gauss_esik':12} {'dönem':12} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")

    results = {}
    for th in GAUSS_THRESHOLDS_TO_TEST:
        rets, opened_ats, n_syms, n_events = _collect_events(df, th)
        results[th] = rets
        if n_events < 30:
            print(f"{th:<12} {'---':12} {n_events:>6}  örnek yetersiz")
            continue

        ts_arr = pd.Series(opened_ats)
        mid = ts_arr.min() + (ts_arr.max() - ts_arr.min()) / 2
        first_mask = (ts_arr < mid).to_numpy()
        arr = np.array(rets)

        for label, mask in (("tum", np.ones(len(arr), dtype=bool)), ("ilk_yari", first_mask), ("ikinci_yari", ~first_mask)):
            s = _stats(arr[mask])
            print(f"{th:<12} {label:12} {s.get('n',0):>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()

    print("── Placebo (eşik=4.5 dışındaki her varyant için) ──")
    for th in GAUSS_THRESHOLDS_TO_TEST:
        if th == 4.5:
            continue
        rets = results.get(th, [])
        if len(rets) < 30:
            continue
        real_pf = _stats(np.array(rets)).get("pf", 0.0)
        print(f"\nesik={th}:")
        _placebo_for_variant(df, th, real_pf)


if __name__ == "__main__":
    run()
