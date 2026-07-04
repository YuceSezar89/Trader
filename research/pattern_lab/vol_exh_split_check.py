"""
v2-1e — Sağlamlık: 45 günü ikiye böl, Pine-EXH deseni (kısa vadede
baseline < EXH < EXH∧support) her iki yarıda BAĞIMSIZ olarak tekrarlanıyor
mu? Rolling hesaplar tam seri üzerinde yapılır (warmup bozulmasın), olaylar
sonradan zaman damgasına göre iki kovaya ayrılır.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import HORIZONS_BARS, MIN_BARS, _fwd_returns, _stats, _vol_rank
from research.pattern_lab.vol_exhaustion_sr_bt import _fetch, _sr_flags
from research.pattern_lab.vol_exh_pine_bt import _pine_exh_events


def run():
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"dönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    halves = ["1_ilk_yari", "2_ikinci_yari"]
    groups = ["baseline", "exh", "exh_support"]
    res = {half: {h: {g: [] for g in groups} for h in HORIZONS_BARS} for half in halves}
    n_syms = 0
    max_h = max(HORIZONS_BARS.values())

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        close = g["close"].to_numpy(float)
        high = g["high"].to_numpy(float)
        low = g["low"].to_numpy(float)
        ts = g["ts"].to_numpy()
        rank = _vol_rank(g["volume"].to_numpy(float))
        near_support, _ = _sr_flags(low, high, close)
        exh_idx = _pine_exh_events(rank)
        exh_support_idx = [i for i in exh_idx if i < len(near_support) and near_support[i]]
        all_idx = list(range(200, len(close) - max_h, 4))

        def bucket(idx_list):
            first, second = [], []
            for i in idx_list:
                if i >= len(close) - max_h:
                    continue
                (first if pd.Timestamp(ts[i]) < mid else second).append(i)
            return first, second

        idx_sets = {"baseline": bucket(all_idx), "exh": bucket(exh_idx), "exh_support": bucket(exh_support_idx)}
        for h, bars in HORIZONS_BARS.items():
            for gi, half in enumerate(halves):
                for gname in groups:
                    res[half][h][gname].append(_fwd_returns(close, idx_sets[gname][gi], bars))

    print(f"analiz edilen sembol: {n_syms}\n")
    for half in halves:
        print(f"══ {half} ══")
        print(f"{'ufuk':6} {'grup':14} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for h in HORIZONS_BARS:
            for gname in groups:
                rets = np.concatenate(res[half][h][gname]) if res[half][h][gname] else np.array([])
                s = _stats(rets)
                print(f"{h:6} {gname:14} {s.get('n',0):>7} {s.get('wr',0):>6} "
                      f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
