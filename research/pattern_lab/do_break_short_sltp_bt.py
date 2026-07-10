"""
do_break_gauss_sltp_bt.py'nin Short aynası. Gauss refinement Short'ta işe
yaramadığı için (do_break_gauss_short_bt.py) burada sadece temel sinyal
(DO aşağı kırılımı + 3 ardışık kırmızı mum) kullanılıyor — OOS eşik
kalibrasyonuna gerek yok (filtrelenecek sürekli bir Gauss değişkeni yok).

Split-period testi (do_break_short_split_check.py) bu sinyalin ilk yarıda
baseline'ın ALTINDA kaldığını göstermişti — yani Short tarafı Long kadar
sağlam değil, bu ekonomik test o zayıflığı miras alıyor (tüm 45 gün
birleştirilmiş, iki rejim karışık).
"""
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, _fetch,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _threshold_events,
)
from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD, ROUND_TRIP_FEE,
)
from research.pattern_lab.do_break_gauss_short_bt import (  # pylint: disable=wrong-import-position
    _do_break_gate_down, _streak_state_down,
)
from research.pattern_lab.do_break_gauss_sltp_bt import DAR_ONCEKI, GENIS_CONFIGS  # pylint: disable=wrong-import-position


def _simulate_exit_short(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          entry_idx: int, entry_price: float, atr_val: float,
                          sl_mult: float, tp_mult: Optional[float],
                          breakeven_mult: Optional[float]) -> tuple[float, str]:
    """Short pozisyon için bar-bar SL/TP/breakeven simülasyonu (_simulate_exit'in aynası).
    SL girişin ÜSTÜNDE, TP ALTINDA, breakeven fiyat lehte (aşağı) gidince tetiklenir."""
    sl = entry_price + sl_mult * atr_val
    tp = entry_price - tp_mult * atr_val if tp_mult is not None else None
    breakeven_level = entry_price - breakeven_mult * atr_val if breakeven_mult is not None else None
    stop = sl
    breakeven_armed = False

    n = len(close)
    last_i = min(entry_idx + HORIZON_BARS, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if breakeven_level is not None and not breakeven_armed and low[i] <= breakeven_level:
            stop = min(stop, entry_price)  # breakeven'e çek, asla geriye gitme
            breakeven_armed = True
        if high[i] >= stop:
            reason = "breakeven" if breakeven_armed and stop <= entry_price else "stop_loss"
            return stop, reason
        if tp is not None and low[i] <= tp:
            return tp, "take_profit"

    return close[last_i], "timeout"


def _dollar_stats(pnls: np.ndarray, days_span: float) -> dict:
    if len(pnls) == 0:
        return {"n": 0}
    total = float(pnls.sum())
    return {
        "n": len(pnls),
        "wr": round(float((pnls > 0).mean() * 100), 1),
        "avg_usd": round(float(pnls.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(total / days_span * 30, 1) if days_span > 0 else 0.0,
    }


def run():
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    days_span = (t_max - t_min).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max} ({DAYS} gün, tüm dönem — OOS ayrımı yok)\n")

    per_symbol = []
    n_syms = 0

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
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate_down(o, c, daily_open)
        count_short, _short_perc = _streak_state_down(o, h, l, c)
        ev3 = _threshold_events(count_short, gate, GAUSS_STREAK_THRESHOLD)
        valid = [i for i in ev3 if np.isfinite(atr[i]) and atr[i] > 0 and i < len(c) - HORIZON_BARS]

        per_symbol.append((h, l, c, atr, valid))

    print(f"analiz edilen sembol: {n_syms}\n")

    blind24h_pnls = []
    all_configs = {"[önceki] " + "SL=1.5 TP=3 BE=1": DAR_ONCEKI, **GENIS_CONFIGS}
    config_pnls: dict[str, list] = {name: [] for name in all_configs}
    config_reasons: dict[str, dict] = {name: {} for name in all_configs}

    for h, l, c, atr, valid in per_symbol:
        for i in valid:
            entry_price = c[i]

            blind_ret = -(c[min(i + HORIZON_BARS, len(c) - 1)] / entry_price - 1)
            blind24h_pnls.append(blind_ret * POSITION_USD - ROUND_TRIP_FEE)

            for name, cfg in all_configs.items():
                exit_price, reason = _simulate_exit_short(h, l, c, i, entry_price, atr[i], **cfg)
                config_reasons[name][reason] = config_reasons[name].get(reason, 0) + 1
                ret = -(exit_price / entry_price - 1)
                config_pnls[name].append(ret * POSITION_USD - ROUND_TRIP_FEE)

    blind24h_pnls = np.array(blind24h_pnls)

    print(f"── Ekonomik karşılaştırma (Short, ${POSITION_USD:.0f} pozisyon, "
          f"round-trip fee ${ROUND_TRIP_FEE:.2f}) ──")
    print(f"{'yöntem':28} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    s = _dollar_stats(blind24h_pnls, days_span)
    print(f"{'kör 24h bekleme':28} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
          f"{s['total_usd']:>10} {s['usd_per_month']:>10}")
    for name, pnls in config_pnls.items():
        s = _dollar_stats(np.array(pnls), days_span)
        if s.get("n", 0) == 0:
            continue
        print(f"{name:28} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")

    print("\n── Çıkış nedeni dağılımı ──")
    for name, reasons in config_reasons.items():
        total = sum(reasons.values())
        breakdown = ", ".join(f"{r}=%{c/total*100:.0f}" for r, c in
                               sorted(reasons.items(), key=lambda x: -x[1]))
        print(f"  {name:28} {breakdown}")


if __name__ == "__main__":
    run()
