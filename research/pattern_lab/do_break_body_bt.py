"""
"Body%" testi (kullanıcının paylaştığı C99 BTHN Pine script'inden — kullanıcı
"body kısmı ne anlatıyor" diye sordu, do_kirilimi.py'deki B kapısı/Marubozu
şartıyla (body/rng>=0.70, ikili) aynı formülün SÜREKLİ/1-100 skor hâli).

do_break_gauss_bt.py ile AYNI yöntem: do_break + 3 ardışık yeşil grubunun
GİRİŞ barının (3. yeşil mum) body_pct = |close-open|/(high-low)*100 değerine
göre tercillere ayrılıp 24h forward getiri karşılaştırılıyor — Gauss tercilini
test ettiğimiz yöntemin birebir aynısı, farklı özellik.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, _fetch, _do_break_gate,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _streak_state, _threshold_events,
)


def _body_pct(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int) -> float:
    rng = h[i] - l[i]
    if rng <= 0:
        return 0.0
    return abs(c[i] - o[i]) / rng * 100.0


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = []
    body_tercile_fwd = {"düşük": [], "orta": [], "yüksek": []}
    all_body_vals = []
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
        count_long, _long_perc = _streak_state(o, h, l, c)
        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        baseline_fwd.append(_fwd_returns(c, all_idx, HORIZON_BARS))

        valid = [(i, _body_pct(o, h, l, c, i)) for i in ev3 if i < len(c) - HORIZON_BARS]
        per_symbol_cache.append((c, valid))
        all_body_vals.extend(bv for _, bv in valid)

    print(f"analize giren sembol: {n_syms}\n")

    s = _stats(np.concatenate(baseline_fwd) if baseline_fwd else np.array([]))
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tüm barlar)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    if not all_body_vals:
        print("\nBody analizi için yeterli olay yok.")
        return
    q1, q2 = np.percentile(all_body_vals, [33.3, 66.7])
    print(f"\n── streak==3 grubu, GİRİŞ barının body% terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")

    for c, valid in per_symbol_cache:
        for i, bv in valid:
            bucket = "düşük" if bv < q1 else ("orta" if bv < q2 else "yüksek")
            r = c[i + HORIZON_BARS] / c[i] - 1
            body_tercile_fwd[bucket].append(r)

    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = np.array(body_tercile_fwd[name])
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


if __name__ == "__main__":
    run()
