"""
Sağlamlık: do_break_gauss_short_bt.py'nin temel bulgusu (DO aşağı kırılımı +
3 ardışık kırmızı mum, PF 1.082 > baseline 1.029 — Gauss refinement Short'ta
işe yaramadığı için sadece streak ekseni test ediliyor) 45 günlük dönemi
ikiye bölünce HER İKİ yarıda BAĞIMSIZ tekrarlanıyor mu? do_open_streak_split_check.py
ile aynı desen (Long tarafının split-period testi), Short'a aynalanmış.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, STREAK_THRESHOLDS, _fetch,
)
from research.pattern_lab.do_open_touch_gauss_bt import _threshold_events  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_short_bt import (  # pylint: disable=wrong-import-position
    _do_break_gate_down, _fwd_returns_neg, _streak_state_down,
)


def run():
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"dönem: {t_min} .. {t_max} ({DAYS} gün)\norta nokta: {mid}\n")

    halves = ["1_ilk_yari", "2_ikinci_yari"]
    groups = ["baseline"] + [f"down_streak_{th}" for th in STREAK_THRESHOLDS]
    res = {half: {g: [] for g in groups} for half in halves}
    n_syms = 0

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        ts_np = ts.to_numpy()
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate_down(o, c, daily_open)
        count_short, _short_perc = _streak_state_down(o, h, l, c)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))

        def bucket(idx_list):
            first, second = [], []
            for i in idx_list:
                if i >= len(c) - HORIZON_BARS:
                    continue
                (first if pd.Timestamp(ts_np[i]) < mid else second).append(i)
            return first, second

        idx_sets = {"baseline": bucket(all_idx)}
        for th in STREAK_THRESHOLDS:
            ev = _threshold_events(count_short, gate, th)
            idx_sets[f"down_streak_{th}"] = bucket(ev)

        for gi, half in enumerate(halves):
            for gname in groups:
                res[half][gname].append(_fwd_returns_neg(c, idx_sets[gname][gi], HORIZON_BARS))

    print(f"analiz edilen sembol: {n_syms}\n")
    for half in halves:
        print(f"══ {half} ══")
        print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for gname in groups:
            rets = np.concatenate(res[half][gname]) if res[half][gname] else np.array([])
            s = _stats(rets)
            print(f"{gname:20} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
