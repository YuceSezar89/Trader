"""
do_open_streak_bt.py'nin iki değişiklikli varyantı (kullanıcı isteği, 9 Tem 2026):

1. Gate: `do_break` (kapanışın DO'yu yukarı KESMESİ) yerine `do_lift` — DO'ya
   dokunup (low<=DO) ÜSTÜNDE kapanma ("temas + sekme", do_kirilimi.py'deki
   mevcut tanım, memory'deki "DO temas ANI" hipoteziyle örtüşüyor).
2. Gauss toplamı: Pine'ın f_gauss_sum(n)=n(n+1)/2 formülü hem ardışık mum
   SAYISINA hem de o ana kadarki % yükselişe uygulanıyor. Sayıya uygulanan
   monoton bir dönüşüm olduğu için (eşik seçimini değiştirmez) tek başına
   test edilmiyor; asıl yeni bilgi gauss_long_perc'in (% hareketin
   ağırlıklandırılmış büyüklüğü) streak==3 grubunu terciller halinde ayırıp
   ayırmadığında — büyüklük ek ayrım gücü katıyor mu sorusu.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, STREAK_THRESHOLDS, _fetch,
)

GAUSS_STREAK_THRESHOLD = 3  # ana bulgu — bu eşikteki olaylar Gauss'a göre tercillere ayrılır


def _gauss_sum(x: np.ndarray) -> np.ndarray:
    return x * (x + 1) / 2.0


def _do_lift_gate(o: np.ndarray, l: np.ndarray, c: np.ndarray,
                   daily_open: np.ndarray) -> np.ndarray:
    """do_break yerine do_lift: DO'ya dokunup (low<=DO) ÜSTÜNDE kapanma anından
    itibaren, ardışık yeşil mum bozulana kadar True kalan maske."""
    n = len(c)
    do_lift = (l <= daily_open) & (c > daily_open) & np.isfinite(daily_open)
    is_long = c > o
    gate = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if do_lift[i]:
            active = True
        elif not is_long[i]:
            active = False
        gate[i] = active
    return gate


def _streak_state(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """Pine'ın count_long / long_start_low / long_perc mantığını BİREBİR,
    her bar için (sadece eşik anında değil) hesaplar."""
    n = len(c)
    is_long = c > o
    count_long = np.zeros(n)
    long_perc = np.full(n, np.nan)
    cnt = 0
    start_low = np.nan
    for i in range(n):
        if is_long[i]:
            cnt += 1
            if cnt == 1 or np.isnan(start_low):
                start_low = l[i]
            long_perc[i] = (h[i] - start_low) / start_low * 100.0
        else:
            cnt = 0
            start_low = np.nan
        count_long[i] = cnt
    return count_long, long_perc


def _threshold_events(count_long: np.ndarray, gate: np.ndarray, th: int) -> list[int]:
    """count_long eşiğe İLK ULAŞTIĞI, gate=True olan bar indeksleri."""
    idx = []
    prev = 0
    for i, cnt in enumerate(count_long):
        if gate[i] and cnt == th and prev != th:
            idx.append(i)
        prev = cnt
    return idx


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = []
    lift_streak_fwd = {th: [] for th in STREAK_THRESHOLDS}
    n_lift_events = {th: 0 for th in STREAK_THRESHOLDS}

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
        gate = _do_lift_gate(o, l, c, daily_open)
        count_long, long_perc = _streak_state(o, h, l, c)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        baseline_fwd.append(_fwd_returns(c, all_idx, HORIZON_BARS))

        for th in STREAK_THRESHOLDS:
            ev = _threshold_events(count_long, gate, th)
            n_lift_events[th] += len(ev)
            lift_streak_fwd[th].append(_fwd_returns(c, ev, HORIZON_BARS))

        # Ana bulgu grubu (streak==3, do_lift gate) için Gauss değerlerini topla
        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(long_perc[ev3], 2))
        valid = [i for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        valid_gauss = [gv for i, gv in zip(ev3, gauss_perc) if np.isfinite(gv) and i < len(c) - HORIZON_BARS]
        per_symbol_cache.append((c, valid, valid_gauss))
        all_gauss_vals.extend(valid_gauss)

    print(f"analize giren sembol: {n_syms}\n")
    print("── do_lift (DO temas + sekme) gate ile ardışık yeşil eşikleri ──")
    print(f"{'grup':38} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    s = _stats(np.concatenate(baseline_fwd) if baseline_fwd else np.array([]))
    print(f"{'baseline (tüm barlar)':38} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
    for th in STREAK_THRESHOLDS:
        rets = np.concatenate(lift_streak_fwd[th]) if lift_streak_fwd[th] else np.array([])
        s = _stats(rets)
        label = f"do_lift + {th} ardışık yeşil"
        print(f"{label:38} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}  (olay={n_lift_events[th]})")

    # Gauss terciline göre ayırma (streak==3 grubu)
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
