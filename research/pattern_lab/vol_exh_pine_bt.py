"""
v2-1d — DÜZELTME: Pine'ın BİREBİR EXH tetikleyicisi (önceki testte YANLIŞ
nokta ölçülmüştü — "tam sönme/≤20" yerine gerçek formül: sıcakken (rank[i-1]
> 75) ANİ tek-bar düşüş (≥5 puan), 3 bar cooldown. Bu, indikatörün gerçekten
işaretlediği (üçgen etiket) nokta — trader ekranlarındaki ok da bu türden
bir geçiş anına işaret ediyordu, tam mavi dibe değil.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import HORIZONS_BARS, MIN_BARS, _fwd_returns, _stats, _vol_rank
from research.pattern_lab.vol_exhaustion_sr_bt import _fetch, _sr_flags

EXH_LEVEL = 75.0
EXH_DROP = 5.0
EXH_COOLDOWN = 3


def _pine_exh_events(rank: np.ndarray) -> list[int]:
    """Pine'ın BİREBİR formülü: exhaustion_signal = rank[i-1]>75 AND
    (rank[i-1]-rank[i])>=5 AND (i-last)>3. İşaretlenen gerçek nokta budur."""
    idx, last = [], -10**9
    for i in range(1, len(rank)):
        r0, r1 = rank[i - 1], rank[i]
        if np.isnan(r0) or np.isnan(r1):
            continue
        if r0 > EXH_LEVEL and (r0 - r1) >= EXH_DROP and (i - last) > EXH_COOLDOWN:
            idx.append(i)
            last = i
    return idx


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar\n")

    fwd_exh = {h: [] for h in HORIZONS_BARS}
    fwd_exh_support = {h: [] for h in HORIZONS_BARS}
    fwd_base = {h: [] for h in HORIZONS_BARS}
    n_syms, n_exh, n_exh_support = 0, 0, 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        close = g["close"].to_numpy(float)
        high = g["high"].to_numpy(float)
        low = g["low"].to_numpy(float)
        rank = _vol_rank(g["volume"].to_numpy(float))
        near_support, _ = _sr_flags(low, high, close)

        exh_idx = _pine_exh_events(rank)
        exh_support_idx = [i for i in exh_idx if i < len(near_support) and near_support[i]]
        n_exh += len(exh_idx)
        n_exh_support += len(exh_support_idx)

        max_h = max(HORIZONS_BARS.values())
        all_idx = list(range(200, len(close) - max_h, 4))
        for h, bars in HORIZONS_BARS.items():
            fwd_exh[h].append(_fwd_returns(close, exh_idx, bars))
            fwd_exh_support[h].append(_fwd_returns(close, exh_support_idx, bars))
            fwd_base[h].append(_fwd_returns(close, all_idx, bars))

    print(f"analize giren sembol: {n_syms} | Pine-EXH olay: {n_exh} | EXH∧support: {n_exh_support}\n")
    print(f"{'ufuk':6} {'grup':28} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for h in HORIZONS_BARS:
        for name, store in (("baseline", fwd_base),
                            ("Pine-EXH (gerçek işaret)", fwd_exh),
                            ("Pine-EXH ∧ near_support", fwd_exh_support)):
            rets = np.concatenate(store[h]) if store[h] else np.array([])
            s = _stats(rets)
            print(f"{h:6} {name:28} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
