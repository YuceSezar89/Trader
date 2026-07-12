"""
do_open_streak sinyallerinde BELTHOLD formasyonunu (pandas_ta_classic::cdl_pattern,
signal_processor.py::_compute_candle_pattern ile AYNI fonksiyon/kütüphane) test
eder. do_open_streak `signals` tablosuna hiç yazmıyor (detector-bazlı) — hazır
candle_pattern verisi yok, bu yüzden hem sinyaller hem BELTHOLD do_open_streak_evol_exit_bt.py'nin
altyapısıyla (üretim dedektörü + gerçek SL=3xATR/TP-yok/24h-timeout simülasyonu)
GEÇMİŞ veriden yeniden üretiliyor.

do_open_streak SADECE LONG olduğu için sadece +BELTHOLD (boğa, teyit eden)
aranıyor — RSI_Cross testindeki "yön-eşleşen BELTHOLD" mantığının aynısı.

Look-ahead yok: cdl_pattern her bar için SADECE o ana kadarki OHLC ile
hesaplanıyor (mum formasyonları zaten geriye-dönük tanımlar, ileri bakmıyor).
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
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
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

    no_belt_rets, belt_rets, opened_ats_no, opened_ats_belt = [], [], [], []
    n_syms, n_events, n_belt = 0, 0, 0

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
            n_events += 1
            ret = _simulate(low, close, i, entry, sl, HORIZON_BARS)
            if belt_confirm[i]:
                n_belt += 1
                belt_rets.append(ret)
                opened_ats_belt.append(pd.Timestamp(ts_np[i]))
            else:
                no_belt_rets.append(ret)
                opened_ats_no.append(pd.Timestamp(ts_np[i]))

    print(f"analiz edilen sembol: {n_syms} | toplam olay: {n_events} | +BELTHOLD teyitli: {n_belt}\n")
    if n_belt < 30:
        print("BELTHOLD'lu olay sayısı çok az, güvenilir yorum yapılamaz.")
        return

    all_ts = pd.Series(opened_ats_no + opened_ats_belt)
    mid = all_ts.min() + (all_ts.max() - all_ts.min()) / 2

    def _report(name, rets, opened_ats):
        arr = np.array(rets)
        ts_arr = pd.Series(opened_ats)
        first_mask = (ts_arr < mid).to_numpy()
        for label, mask in (("tum", np.ones(len(arr), dtype=bool)), ("ilk_yari", first_mask), ("ikinci_yari", ~first_mask)):
            s = _stats(arr[mask])
            print(f"{name:20} {label:12} {s.get('n',0):>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print(f"{'grup':20} {'dönem':12} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    _report("BELTHOLD yok", no_belt_rets, opened_ats_no)
    _report("+BELTHOLD teyitli", belt_rets, opened_ats_belt)


if __name__ == "__main__":
    run()
