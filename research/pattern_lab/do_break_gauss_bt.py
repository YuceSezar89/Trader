"""
do_open_touch_gauss_bt.py ile aynı Gauss tercil analizi, ama gate olarak
do_lift (temas+sekme) yerine ORİJİNAL do_break (DO'nun kapanışla yukarı
kesilmesi) kullanılıyor — do_break zaten do_lift'ten daha güçlü çıkmıştı
(streak==3: PF 1.034 vs 0.989), şimdi ona Gauss büyüklük terciliyle ek
ayrım gücü eklenip eklenmediğine bakılıyor. Yardımcı fonksiyonlar (streak
state, threshold events, gauss sum) tekrar yazılmadı, doğrudan import edildi.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, STREAK_THRESHOLDS, _fetch, _do_break_gate,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _gauss_sum, _streak_state, _threshold_events,
)


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = []
    break_streak_fwd = {th: [] for th in STREAK_THRESHOLDS}
    n_break_events = {th: 0 for th in STREAK_THRESHOLDS}

    gauss_tercile_fwd = {"düşük": [], "orta": [], "yüksek": []}
    all_gauss_vals = []
    n_syms = 0
    per_symbol_cache = []

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        count_long, long_perc = _streak_state(o, h, l, c)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        baseline_fwd.append(_fwd_returns(c, all_idx, HORIZON_BARS))

        for th in STREAK_THRESHOLDS:
            ev = _threshold_events(count_long, gate, th)
            n_break_events[th] += len(ev)
            break_streak_fwd[th].append(_fwd_returns(c, ev, HORIZON_BARS))

        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(long_perc[ev3], 2))
        valid_idx = [i for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        valid_gauss = [gv for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        per_symbol_cache.append((c, valid_idx, valid_gauss))
        all_gauss_vals.extend(valid_gauss)

    print(f"analize giren sembol: {n_syms}\n")
    print("── do_break (DO kırılımı) gate ile ardışık yeşil eşikleri ──")
    print(f"{'grup':38} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    s = _stats(np.concatenate(baseline_fwd) if baseline_fwd else np.array([]))
    print(f"{'baseline (tüm barlar)':38} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
    for th in STREAK_THRESHOLDS:
        rets = np.concatenate(break_streak_fwd[th]) if break_streak_fwd[th] else np.array([])
        s = _stats(rets)
        label = f"do_break + {th} ardışık yeşil"
        print(f"{label:38} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}  (olay={n_break_events[th]})")

    if not all_gauss_vals:
        print("\nGauss analizi için yeterli olay yok.")
        return
    q1, q2 = np.percentile(all_gauss_vals, [33.3, 66.7])
    print(f"\n── streak=={GAUSS_STREAK_THRESHOLD} grubu, gauss_long_perc terciline göre ── "
          f"(q1={q1:.2f}, q2={q2:.2f})")

    for c, idx_list, gauss_list in per_symbol_cache:
        for i, gv in zip(idx_list, gauss_list):
            bucket = "düşük" if gv < q1 else ("orta" if gv < q2 else "yüksek")
            r = c[i + HORIZON_BARS] / c[i] - 1
            gauss_tercile_fwd[bucket].append(r)

    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = np.array(gauss_tercile_fwd[name])
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


if __name__ == "__main__":
    run()
