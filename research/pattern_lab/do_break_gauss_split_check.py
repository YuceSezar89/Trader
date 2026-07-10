"""
Sağlamlık: do_break_gauss_bt.py bulgusu (do_break + 3 ardışık yeşil + yüksek
Gauss tercili → PF 1.17, ort% +0.77) 45 günlük dönemi ikiye bölünce HER İKİ
yarıda BAĞIMSIZ tekrarlanıyor mu? Gauss tercil eşikleri (q1/q2) TAM SERİDEN
(45 günün tamamından) hesaplanıp sabit tutuluyor — sadece olaylar sonradan
zaman damgasına göre iki kovaya ayrılıyor (vol_exh_split_check.py deseni).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, _fetch, _do_break_gate,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _gauss_sum, _streak_state, _threshold_events,
)


def run():
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"dönem: {t_min} .. {t_max} ({DAYS} gün)\norta nokta: {mid}\n")

    # 1. geçiş: her sembol için streak==3 olaylarını ve Gauss değerlerini topla
    # (tam seri üzerinde — warmup/tercil eşiği bozulmasın)
    per_symbol = []  # (ts_np, close, all_idx, ev3_idx, gauss_vals)
    all_gauss_vals = []
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
        gate = _do_break_gate(o, c, daily_open)
        count_long, long_perc = _streak_state(o, h, l, c)

        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(long_perc[ev3], 2))
        valid = [(i, gv) for i, gv in zip(ev3, gauss_perc)
                 if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        all_gauss_vals.extend(gv for _, gv in valid)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        per_symbol.append((ts_np, c, all_idx, valid))

    q1, q2 = np.percentile(all_gauss_vals, [33.3, 66.7])
    print(f"analiz edilen sembol: {n_syms} | tam seriden Gauss eşiği: q1={q1:.2f}, q2={q2:.2f}\n")

    halves = ["1_ilk_yari", "2_ikinci_yari"]
    groups = ["baseline", "streak3_tumu", "gauss_düşük", "gauss_orta", "gauss_yüksek"]
    res = {half: {g: [] for g in groups} for half in halves}

    for ts_np, c, all_idx, valid in per_symbol:
        def half_of(i):
            return 0 if pd.Timestamp(ts_np[i]) < mid else 1

        buckets = {g: ([], []) for g in groups}
        for i in all_idx:
            buckets["baseline"][half_of(i)].append(i)
        for i, gv in valid:
            buckets["streak3_tumu"][half_of(i)].append(i)
            name = "gauss_düşük" if gv < q1 else ("gauss_orta" if gv < q2 else "gauss_yüksek")
            buckets[name][half_of(i)].append(i)

        for gi in (0, 1):
            half = halves[gi]
            for gname in groups:
                res[half][gname].append(_fwd_returns(c, buckets[gname][gi], HORIZON_BARS))

    for half in halves:
        print(f"══ {half} ══")
        print(f"{'grup':16} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for gname in groups:
            rets = np.concatenate(res[half][gname]) if res[half][gname] else np.array([])
            s = _stats(rets)
            print(f"{gname:16} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
