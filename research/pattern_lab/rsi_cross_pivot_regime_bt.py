"""
Bugün panele eklenen Fibonacci Pivot Point'i (indicators/core.py::calculate_fib_pivots
— panelle BİREBİR AYNI fonksiyon, [[project_chart_panel_pivots]]) RSI_Cross'a
rejim/filtre olarak uygular. rsi_cross_volbreakout_regime_bt.py ile aynı desen.

pivot_bias: +1 = kapanış günlük PP'nin ÜSTÜNDE (boğa yanlılığı), -1 = ALTINDA
(ayı yanlılığı). PP önceki takvim gününün H/L/C'sinden hesaplanır, o günün TÜM
barlarına uygulanır — klasik "pivot bias" kullanımı: PP üstü Long'u mu, PP altı
Short'u mu destekliyor?

Look-ahead yok: PP DAİMA bir önceki günün H/L/C'sinden (pp.shift(1)) — bugünün
verisi bugünün PP'sini hiç etkilemiyor. Bar-kapanış güvenliği merge_regime'in
mevcut (opened_at - 15dk) kesme mantığından geliyor.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_fib_pivots  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    INDICATOR, _fetch_regime, _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position

BAR_DURATION = pd.Timedelta(minutes=15)


def _pivot_bias_series(g: pd.DataFrame) -> pd.Series:
    g = g.copy()
    g["date"] = g["ts"].dt.date
    daily = g.groupby("date").agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).sort_index()
    pp = daily.apply(lambda r: calculate_fib_pivots(r["high"], r["low"], r["close"])["pp"], axis=1)
    pp_prev = pp.shift(1)  # bugünün PP'si DÜNÜN H/L/C'sinden
    g["pp"] = g["date"].map(pp_prev.to_dict())
    bias = np.sign(g["close"] - g["pp"])
    return bias.fillna(0)


def run() -> None:
    for direction in ("Long", "Short"):
        sig_df = _fetch_signals(INDICATOR, direction)
        if len(sig_df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
            continue

        regime_df = _fetch_regime(sig_df["symbol"].unique().tolist(), "15m", _pivot_bias_series, "pivot_bias")
        merged = _merge_regime(sig_df, regime_df, "pivot_bias", BAR_DURATION)
        print(f"{INDICATOR} — {direction}: {len(sig_df):,} sinyal, {len(merged):,} pivot_bias ile eşleşti "
              f"(ort={merged['pivot_bias'].mean():.2f}, "
              f"PP-üstü={len(merged[merged['pivot_bias']==1])}, "
              f"PP-altı={len(merged[merged['pivot_bias']==-1])})")

        label = f"{INDICATOR} — {direction} — pivot_bias (Fib PP üstü/altı)"
        _run_single_var_on_df(label, merged, "pivot_bias")


if __name__ == "__main__":
    run()
