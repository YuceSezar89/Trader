"""
do_break_gauss_economic_bt.py'nin devamı — kör 24h bekleme yerine GERÇEK SL/TP
simülasyonu ile ekonomik etkiyi ölçer. SL/TP mantığı iki kaynaktan geliyor:

1. TRader'ın kendi projesindeki mevcut standart (config.py): SL=1.5×ATR,
   TP=3.0×ATR (do_kirilimi'nin kullandığı temel çarpanlar, bonus'suz).
2. Hocanın DevisSoTrader kodundaki (`/Users/yusuf/Documents/Sezar/risk_management.py`
   ::adaptive_stop_loss) breakeven fikri: fiyat 1×ATR lehte gidince stop girişe
   çekilir — STOP_TRAILING_ANALIZI.md'de tespit edilen "TP'ye ulaşmayan işlem
   korumasız SL'e gider" açığını simülasyonda baştan kapatıyor (Plan A'nın
   canlıya alınmadan önceki backtest doğrulaması).

Aynı OOS (out-of-sample) kalibrasyon deseni korundu: Gauss eşiği SADECE
ilk yarıdan (in-sample) türetilip ikinci yarıya (out-of-sample) sabit
uygulanıyor — do_break_gauss_economic_bt.py ile birebir karşılaştırılabilir.

Sınır: Bar-ici (intra-bar) SL/TP sırası bilinmiyor (15m OHLC'de hangisi önce
değdi belli değil) — SL önce değdi varsayımıyla KONSERVATİF ölçülüyor (aynı
bar'da hem SL hem TP'ye değinen barlarda SL kazanır).
"""
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, _fetch, _do_break_gate,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _gauss_sum, _streak_state, _threshold_events,
)
from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD, ROUND_TRIP_FEE,
)

DAR_ONCEKI = dict(sl_mult=1.5, tp_mult=3.0, breakeven_mult=1.0)  # önceki test (kötü çıktı)

GENIS_CONFIGS = {
    "SL=3 TP=8 BE=yok":     dict(sl_mult=3.0, tp_mult=8.0, breakeven_mult=None),
    "SL=3 TP=8 BE=2":       dict(sl_mult=3.0, tp_mult=8.0, breakeven_mult=2.0),
    "SL=4 TP=10 BE=yok":    dict(sl_mult=4.0, tp_mult=10.0, breakeven_mult=None),
    "SL=3 TP=yok(sadece SL)": dict(sl_mult=3.0, tp_mult=None, breakeven_mult=None),
}


def _simulate_exit(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    entry_idx: int, entry_price: float, atr_val: float,
                    sl_mult: float, tp_mult: Optional[float],
                    breakeven_mult: Optional[float]) -> tuple[float, str]:
    """Long pozisyon için bar-bar SL/TP/breakeven simülasyonu.
    tp_mult=None → TP yok (sadece SL + horizon timeout).
    breakeven_mult=None → breakeven kilidi yok.
    Döner: (exit_price, reason) — reason: stop_loss | breakeven | take_profit | timeout."""
    sl = entry_price - sl_mult * atr_val
    tp = entry_price + tp_mult * atr_val if tp_mult is not None else None
    breakeven_level = entry_price + breakeven_mult * atr_val if breakeven_mult is not None else None
    stop = sl
    breakeven_armed = False

    n = len(close)
    last_i = min(entry_idx + HORIZON_BARS, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if breakeven_level is not None and not breakeven_armed and high[i] >= breakeven_level:
            stop = max(stop, entry_price)  # breakeven'e çek, asla geriye gitme
            breakeven_armed = True
        if low[i] <= stop:
            reason = "breakeven" if breakeven_armed and stop >= entry_price else "stop_loss"
            return stop, reason
        if tp is not None and high[i] >= tp:
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
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max} ({DAYS} gün)")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    is_gauss_vals = []
    per_symbol = []
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
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        count_long, long_perc = _streak_state(o, h, l, c)
        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(long_perc[ev3], 2))
        valid = [(i, gv) for i, gv in zip(ev3, gauss_perc)
                 if np.isfinite(gv) and np.isfinite(atr[i]) and atr[i] > 0
                 and i < len(c) - HORIZON_BARS]

        is_gauss_vals.extend(gv for i, gv in valid if pd.Timestamp(ts_np[i]) < mid)
        per_symbol.append((ts_np, h, l, c, atr, valid))

    oos_threshold = float(np.percentile(is_gauss_vals, 66.7))
    print(f"analiz edilen sembol: {n_syms} | SABİT Gauss eşiği: {oos_threshold:.2f}\n")

    blind24h_pnls = []
    all_configs = {"[önceki] " + "SL=1.5 TP=3 BE=1": DAR_ONCEKI, **GENIS_CONFIGS}
    config_pnls: dict[str, list] = {name: [] for name in all_configs}
    config_reasons: dict[str, dict] = {name: {} for name in all_configs}

    for ts_np, h, l, c, atr, valid in per_symbol:
        oos_high = [(i, gv) for i, gv in valid
                    if pd.Timestamp(ts_np[i]) >= mid and gv >= oos_threshold]
        for i, _gv in oos_high:
            entry_price = c[i]

            blind_ret = c[min(i + HORIZON_BARS, len(c) - 1)] / entry_price - 1
            blind24h_pnls.append(blind_ret * POSITION_USD - ROUND_TRIP_FEE)

            for name, cfg in all_configs.items():
                exit_price, reason = _simulate_exit(h, l, c, i, entry_price, atr[i], **cfg)
                config_reasons[name][reason] = config_reasons[name].get(reason, 0) + 1
                ret = exit_price / entry_price - 1
                config_pnls[name].append(ret * POSITION_USD - ROUND_TRIP_FEE)

    blind24h_pnls = np.array(blind24h_pnls)

    print(f"\n── Ekonomik karşılaştırma (${POSITION_USD:.0f} pozisyon, "
          f"round-trip fee ${ROUND_TRIP_FEE:.2f}) ──")
    print(f"{'yöntem':28} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    s = _dollar_stats(blind24h_pnls, oos_days)
    print(f"{'kör 24h bekleme':28} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
          f"{s['total_usd']:>10} {s['usd_per_month']:>10}")
    for name, pnls in config_pnls.items():
        s = _dollar_stats(np.array(pnls), oos_days)
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
