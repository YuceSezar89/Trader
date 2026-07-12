"""
ha_cross_evol_exit_bt.py'nin (EVOL-disiplinli erken çıkış, [[project_devisso_ersi]])
eşik×min_hold ızgara taraması — veri/EVOL serisi BİR KEZ çekilip önbelleğe
alınıyor, sadece simülasyon tekrarlanıyor (40 kombinasyon için tekrar DB
sorgusu yok). Her hücre için split-period (ilk/ikinci yarı PF) + "floor%"
(EVOL-çıkışlarının kaçının tam min_hold barında tetiklendiği — yüksekse hâlâ
artefakt şüphesi) raporlanıyor.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.ha_cross_evol_exit_bt import (  # pylint: disable=wrong-import-position
    HORIZON_BARS, RANK_WINDOW, RVOL_WINDOW, SL_MULT, TP_MULT,
    _fetch_execution_bars, _simulate_baseline,
)
from research.pattern_lab.ha_cross_pivot_tp_bt import _fetch_signals  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

THRESHOLDS = [20, 25, 30, 35, 40, 45, 50, 55]
MIN_HOLDS = [3, 5, 8, 10, 15]


def _simulate_evol_exit_detailed(high, low, close, evol, entry_idx, entry_price, sl, tp, horizon, threshold, min_hold):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        held = i - entry_idx
        if low[i] <= sl:
            return sl / entry_price - 1, "sl", held
        if held >= min_hold and not np.isnan(evol[i]) and evol[i] < threshold:
            return close[i] / entry_price - 1, "evol", held
        if high[i] >= tp:
            return tp / entry_price - 1, "tp", held
    return close[last_i] / entry_price - 1, "timeout", last_i - entry_idx


def _load_entries() -> list:
    sig_df = _fetch_signals("Long")
    entries = []
    for interval, sub in sig_df.groupby("interval"):
        bars = _fetch_execution_bars(sub["symbol"].unique().tolist(), interval)
        horizon = HORIZON_BARS.get(interval, 96)
        for row in sub.itertuples():
            g = bars.get(row.symbol)
            if g is None or len(g) < RANK_WINDOW + RVOL_WINDOW + 5:
                continue
            opened_at = pd.Timestamp(row.opened_at)
            idx = g["ts"].searchsorted(opened_at, side="right") - 1
            if idx < RANK_WINDOW + RVOL_WINDOW or idx >= len(g) - 1:
                continue
            entry = float(row.open_price)
            atr = float(row.atr)
            sl = entry - SL_MULT * atr
            tp = entry + TP_MULT * atr
            if not (sl < entry < tp):
                continue
            entries.append({
                "idx": idx, "entry": entry, "sl": sl, "tp": tp, "horizon": horizon,
                "high": g["high"].to_numpy(float), "low": g["low"].to_numpy(float),
                "close": g["close"].to_numpy(float), "evol": g["evol"].to_numpy(float),
                "opened_at": opened_at,
            })
    return entries


def run() -> None:
    print("Sinyaller + yürütme barları + EVOL serisi yükleniyor (bir kez)...")
    entries = _load_entries()
    print(f"Geçerli giriş: {len(entries)}\n")

    ts_arr = pd.Series([e["opened_at"] for e in entries])
    mid = ts_arr.min() + (ts_arr.max() - ts_arr.min()) / 2
    first_mask = (ts_arr < mid).to_numpy()

    baseline_rets = np.array([
        _simulate_baseline(e["high"], e["low"], e["close"], e["idx"], e["entry"], e["sl"], e["tp"], e["horizon"])
        for e in entries
    ])
    b_tum, b_ilk, b_ik = _stats(baseline_rets), _stats(baseline_rets[first_mask]), _stats(baseline_rets[~first_mask])
    print(f"BASELINE (ATR-tek-TP): PF_tum={b_tum.get('pf',0)} PF_ilk={b_ilk.get('pf',0)} PF_ikinci={b_ik.get('pf',0)}\n")

    print(f"{'eşik':>5} {'min_hold':>9} {'PF_tum':>8} {'PF_ilk':>8} {'PF_ikinci':>10} {'evol%':>7} {'floor%':>7}")
    for th in THRESHOLDS:
        for mh in MIN_HOLDS:
            rets, held_list, reasons = [], [], []
            for e in entries:
                r, reason, held = _simulate_evol_exit_detailed(
                    e["high"], e["low"], e["close"], e["evol"],
                    e["idx"], e["entry"], e["sl"], e["tp"], e["horizon"], th, mh,
                )
                rets.append(r)
                held_list.append(held)
                reasons.append(reason)
            arr = np.array(rets)
            reasons_arr = np.array(reasons)
            held_arr = np.array(held_list)
            s_tum = _stats(arr)
            s_ilk = _stats(arr[first_mask])
            s_ik = _stats(arr[~first_mask])
            evol_mask = reasons_arr == "evol"
            evol_pct = evol_mask.mean()
            floor_pct = (held_arr[evol_mask] == mh).mean() if evol_mask.sum() > 0 else 0.0
            print(f"{th:>5} {mh:>9} {s_tum.get('pf',0):>8} {s_ilk.get('pf',0):>8} "
                  f"{s_ik.get('pf',0):>10} {evol_pct:>7.1%} {floor_pct:>7.1%}")


if __name__ == "__main__":
    run()
