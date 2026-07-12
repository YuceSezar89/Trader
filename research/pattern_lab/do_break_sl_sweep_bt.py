"""
MAE/MFE analizinin (do_break_mae_mfe_bt.py) devamı — birden çok SL çarpanını
(TP YOK, breakeven YOK — v2-4'ün kazanan formülü) do_break+streak==3
olaylarında GERÇEK $ ekonomik etkiyle karşılaştırır. Hem BASELINE (tüm
olaylar) hem ULTRA (2/2 MTF onaylı alt küme, v2-15) ayrı ayrı ölçülüyor.

Simülasyon mantığı (`_simulate_exit`) do_break_gauss_sltp_bt.py'den doğrudan
import edildi — kod tekrarı yok, aynı konservatif varsayım (bar-içi sıra
bilinmiyor, SL önce değdi kabul edilir) korunuyor. Split-period sağlamlık
kontrolü var.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD,
    ROUND_TRIP_FEE,
)
from research.pattern_lab.do_break_gauss_sltp_bt import (
    _simulate_exit,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
    MTF_CONFIRM_TFS,
    MTF_STREAK_TARGET,
    _do_break_gate,
    _fetch,
    _streak_events,
)
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.mtf_helpers import (  # pylint: disable=wrong-import-position
    _confirm_count,
    _fetch_dir_data,
)
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

SL_MULTIPLES = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


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


def _report(label: str, events: list, days_span: float) -> None:
    print(f"\n=== {label} (n={len(events)}) ===")
    if not events:
        return

    blind_pnls = []
    sl_pnls: dict[float, list] = {sl: [] for sl in SL_MULTIPLES}
    sl_reasons: dict[float, dict] = {sl: {} for sl in SL_MULTIPLES}

    for h, l, c, atr_val, i, entry_price in events:
        blind_ret = c[min(i + HORIZON_BARS, len(c) - 1)] / entry_price - 1
        blind_pnls.append(blind_ret * POSITION_USD - ROUND_TRIP_FEE)

        for sl in SL_MULTIPLES:
            exit_price, reason = _simulate_exit(
                h,
                l,
                c,
                i,
                entry_price,
                atr_val,
                sl_mult=sl,
                tp_mult=None,
                breakeven_mult=None,
            )
            sl_reasons[sl][reason] = sl_reasons[sl].get(reason, 0) + 1
            ret = exit_price / entry_price - 1
            sl_pnls[sl].append(ret * POSITION_USD - ROUND_TRIP_FEE)

    print(f"{'yöntem':24} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    s = _dollar_stats(np.array(blind_pnls), days_span)
    print(
        f"{'kör 24h bekleme':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
        f"{s['total_usd']:>10} {s['usd_per_month']:>10}"
    )
    for sl in SL_MULTIPLES:
        s = _dollar_stats(np.array(sl_pnls[sl]), days_span)
        print(
            f"{'SL='+str(sl)+'×ATR (TP yok)':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
            f"{s['total_usd']:>10} {s['usd_per_month']:>10}"
        )

    print("\n-- çıkış nedeni dağılımı --")
    for sl in SL_MULTIPLES:
        total = sum(sl_reasons[sl].values())
        breakdown = ", ".join(
            f"{r}=%{c/total*100:.0f}"
            for r, c in sorted(sl_reasons[sl].items(), key=lambda x: -x[1])
        )
        print(f"  SL={sl}×ATR: {breakdown}")


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_events = []  # (h, l, c, atr_val, i, entry_price, ts)
    ultra_events = []

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _streak_events(o, c, gate=gate)
        target_events = [
            i
            for i in events[MTF_STREAK_TARGET]
            if i + HORIZON_BARS < len(c) and np.isfinite(atr[i]) and atr[i] > 0
        ]
        if not target_events:
            continue

        dir_data = _fetch_dir_data(sym, MTF_CONFIRM_TFS)

        for i in target_events:
            entry_price = c[i]
            rec = (h, l, c, atr[i], i, entry_price)
            baseline_events.append((rec, ts.iloc[i]))

            if dir_data is not None:
                confirm_count = _confirm_count(
                    dir_data, MTF_CONFIRM_TFS, ts.iloc[i], want_bullish=True
                )
                if confirm_count == len(MTF_CONFIRM_TFS):
                    ultra_events.append((rec, ts.iloc[i]))

    all_ts = [t for _, t in baseline_events]
    t_min, t_max = min(all_ts), max(all_ts)
    days_span = (t_max - t_min).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max} ({days_span:.1f} gün)")

    _report("BASELINE (tüm do_break+streak3)", [r for r, _ in baseline_events], days_span)
    _report("ULTRA (2/2 MTF onaylı alt küme, v2-15)", [r for r, _ in ultra_events], days_span)

    mid = t_min + (t_max - t_min) / 2
    half_days = days_span / 2
    print("\n\n──────── SPLIT-PERIOD SAĞLAMLIK (ULTRA alt kümesi) ────────")
    ultra_first = [r for r, t in ultra_events if t < mid]
    ultra_second = [r for r, t in ultra_events if t >= mid]
    _report("ULTRA — ilk yarı", ultra_first, half_days)
    _report("ULTRA — ikinci yarı", ultra_second, half_days)


if __name__ == "__main__":
    run()
