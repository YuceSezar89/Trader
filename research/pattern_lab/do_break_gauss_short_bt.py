"""
Simetri testi (kullanıcı isteği, 10 Tem 2026): do_break_gauss_bt.py'nin Long
bulgusu (DO kırılımı + 3 ardışık yeşil + Gauss büyüklük üst tercili → PF 1.17,
OOS +$367/ay) Short tarafta AYNA olarak da çalışıyor mu? DO'yu aşağı kıran +
3 ardışık kırmızı mum + o mumlarda kat edilen düşüşün Gauss-ağırlıklı büyüklüğü.

Eğer simetrik çalışırsa: bu tesadüf/overfitting değil, gerçek bir piyasa
fenomeni olduğuna dair güçlü ek kanıt (Long-only bir regülarite genelde daha
şüpheli, iki yönde de tutarlı bir etki daha az şüpheli). Çalışmazsa: bulgunun
Long'a özgü bir asimetri taşıdığını öğreniriz (ör. panik-satış farklı dinamik
olabilir) — ikisi de bilgi değeri taşıyor.

_stats() pozitif getiriyi "iyi" sayıyor — short için ham getiri NEGATİF
edilerek geçiriliyor (fiyat düştükçe short kâr eder).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, STREAK_THRESHOLDS, _fetch,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _gauss_sum, _threshold_events,
)


def _do_break_gate_down(o: np.ndarray, c: np.ndarray, daily_open: np.ndarray) -> np.ndarray:
    """_do_break_gate'in aynası: DO'nun kapanışla AŞAĞI kesilmesinden itibaren,
    ardışık kırmızı mum bozulana (yeşil mum) kadar True kalan maske."""
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break_down = (c < daily_open) & (prev_c >= daily_open) & np.isfinite(daily_open)
    is_short = c < o
    gate = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if do_break_down[i]:
            active = True
        elif not is_short[i]:
            active = False
        gate[i] = active
    return gate


def _streak_state_down(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """_streak_state'in aynası: count_short / short_start_high / short_perc."""
    n = len(c)
    is_short = c < o
    count_short = np.zeros(n)
    short_perc = np.full(n, np.nan)
    cnt = 0
    start_high = np.nan
    for i in range(n):
        if is_short[i]:
            cnt += 1
            if cnt == 1 or np.isnan(start_high):
                start_high = h[i]
            short_perc[i] = (start_high - l[i]) / start_high * 100.0
        else:
            cnt = 0
            start_high = np.nan
        count_short[i] = cnt
    return count_short, short_perc


def _fwd_returns_neg(close: np.ndarray, idx: list[int], bars: int) -> np.ndarray:
    """Short için: fiyat düşerse pozitif getiri sayılsın diye ham getiri negatif edilir."""
    out = []
    n = len(close)
    for i in idx:
        if i + bars < n:
            out.append(-(close[i + bars] / close[i] - 1))
    return np.array(out)


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = []
    down_streak_fwd = {th: [] for th in STREAK_THRESHOLDS}
    n_down_events = {th: 0 for th in STREAK_THRESHOLDS}

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
        gate = _do_break_gate_down(o, c, daily_open)
        count_short, short_perc = _streak_state_down(o, h, l, c)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        baseline_fwd.append(_fwd_returns_neg(c, all_idx, HORIZON_BARS))

        for th in STREAK_THRESHOLDS:
            ev = _threshold_events(count_short, gate, th)
            n_down_events[th] += len(ev)
            down_streak_fwd[th].append(_fwd_returns_neg(c, ev, HORIZON_BARS))

        ev3 = _threshold_events(count_short, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(short_perc[ev3], 2))
        valid_idx = [i for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        valid_gauss = [gv for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        per_symbol_cache.append((c, valid_idx, valid_gauss))
        all_gauss_vals.extend(valid_gauss)

    print(f"analize giren sembol: {n_syms}\n")
    print("── do_break (DO aşağı kırılımı) gate ile ardışık kırmızı eşikleri (Short) ──")
    print(f"{'grup':38} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    s = _stats(np.concatenate(baseline_fwd) if baseline_fwd else np.array([]))
    print(f"{'baseline (tüm barlar, short)':38} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
    for th in STREAK_THRESHOLDS:
        rets = np.concatenate(down_streak_fwd[th]) if down_streak_fwd[th] else np.array([])
        s = _stats(rets)
        label = f"do_break_down + {th} ardışık kırmızı"
        print(f"{label:38} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}  (olay={n_down_events[th]})")

    if not all_gauss_vals:
        print("\nGauss analizi için yeterli olay yok.")
        return
    q1, q2 = np.percentile(all_gauss_vals, [33.3, 66.7])
    print(f"\n── streak=={GAUSS_STREAK_THRESHOLD} grubu (Short), gauss_short_perc terciline göre ── "
          f"(q1={q1:.2f}, q2={q2:.2f})")

    for c, idx_list, gauss_list in per_symbol_cache:
        for i, gv in zip(idx_list, gauss_list):
            bucket = "düşük" if gv < q1 else ("orta" if gv < q2 else "yüksek")
            r = -(c[i + HORIZON_BARS] / c[i] - 1)
            gauss_tercile_fwd[bucket].append(r)

    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = np.array(gauss_tercile_fwd[name])
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


if __name__ == "__main__":
    run()
