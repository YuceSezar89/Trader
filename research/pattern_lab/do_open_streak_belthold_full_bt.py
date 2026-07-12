"""
do_open_streak_belthold_bt.py'nin (baseline karşılaştırma) devamı — aynı
simüle edilmiş olayları threshold_optimizer'ın TAM 3-kapılı disiplinine
(IS/OOS + split-period + placebo) sokuyor, RSI_Cross testleriyle birebir
karşılaştırılabilir olması için.

belt_confirm: +BELTHOLD (boğa, do_open_streak SADECE LONG olduğu için teyit
eden tek yön) o giriş barında var mı (0/1).
"""
import os
import sys

import numpy as np
import pandas as pd
import pandas_ta_classic as pta  # pylint: disable=unused-import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_evol_exit_bt import (  # pylint: disable=wrong-import-position
    HORIZON_BARS, MIN_BARS, WARMUP, _fetch, _signal_series,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position
from signals.do_open_streak import SL_ATR_MULT  # pylint: disable=wrong-import-position


def _belthold_confirm_series(g: pd.DataFrame) -> np.ndarray:
    df_cdl = g[["open", "high", "low", "close"]].copy().astype(float)
    df_cdl.ta.cores = 0
    result = df_cdl.ta.cdl_pattern(name="all")
    if result is None or "CDL_BELTHOLD" not in result.columns:
        return np.zeros(len(g), dtype=bool)
    return (result["CDL_BELTHOLD"] > 0).to_numpy()


def _simulate(low, close, entry_idx, entry_price, sl, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
    return close[last_i] / entry_price - 1


def run() -> None:
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar\n")

    rows = []
    n_syms = 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        signal = _signal_series(g)
        atr_series = calculate_atr(g, period=14).to_numpy()
        belt_confirm = _belthold_confirm_series(g)
        close = g["close"].to_numpy(float)
        low = g["low"].to_numpy(float)
        ts_np = g["ts"].to_numpy()

        idxs = np.where(signal)[0]
        idxs = idxs[(idxs >= WARMUP) & (idxs < len(g) - HORIZON_BARS)]

        for i in idxs:
            atr = atr_series[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = close[i]
            sl = entry - SL_ATR_MULT * atr
            ret = _simulate(low, close, i, entry, sl, HORIZON_BARS)
            rows.append({
                "opened_at": pd.Timestamp(ts_np[i]),
                "realized_pnl": ret * 100.0,  # threshold_optimizer yüzde-puan bekliyor
                "belt_confirm": 1.0 if belt_confirm[i] else 0.0,
            })

    merged = pd.DataFrame(rows)
    print(f"analiz edilen sembol: {n_syms} | toplam olay: {len(merged)} | "
          f"+BELTHOLD teyitli oranı={merged['belt_confirm'].mean():.2%}\n")

    if len(merged) < 200:
        print("Örneklem çok küçük.")
        return

    label = "do_open_streak — Long — belt_confirm (+BELTHOLD teyitli)"
    _run_single_var_on_df(label, merged, "belt_confirm")


if __name__ == "__main__":
    run()
