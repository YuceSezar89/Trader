"""
HA_Cross Long'da RVOL bileşenini (EVOL'ün payda bileşeni, [[project_devisso_ersi]])
SL DEĞİL, EK bir "hacim-sönmesi" çıkışı olarak test eder: pozisyon açıkken RVOL
belirli bir eşiğin altına düşerse (hareketi süren hacim kayboldu), fiyat SL'e
hiç değmeden ERKEN çıkılır. Mevcut ATR-bazlı SL/TP (ha_cross_pivot_tp_bt.py'deki
gibi, risk_policy.py tabanı: SL=1.5×ATR, TP=3.0×ATR) DEĞİŞMİYOR, sadece ek bir
çıkış kapısı ekleniyor.

RVOL = Volume / SMA(Volume, 20) — evol_bt.py ile aynı tanım. Birkaç eşik adayı
(0.5-0.9) taranıyor; SL > hacim-sönmesi > TP önceliğiyle (aynı bar'da birden
fazla koşul tetiklenirse en kötümser sırayla, do_break_gauss_sltp_bt.py'deki
disiplinle aynı).

Look-ahead yok: RVOL her barda SADECE o ana kadarki geçmiş 20 barla hesaplanıyor.
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_pivot_tp_bt import (
    _fetch_signals,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

DAYS = 60
SL_MULT = Config.RISK_SL_MULTIPLIER  # 1.5
TP_MULT = Config.RISK_TP_MULTIPLIER  # 3.0
HORIZON_HOURS = 24.0
HORIZON_BARS = {"5m": int(HORIZON_HOURS * 12), "15m": int(HORIZON_HOURS * 4)}
RVOL_WINDOW = 20
RVOL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _fetch_execution_bars(symbols: list, interval: str) -> dict:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, high, low, close, volume
        FROM cagg_{interval}
        WHERE bucket > NOW() - INTERVAL '{DAYS} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        g["rvol"] = g["volume"] / g["volume"].rolling(RVOL_WINDOW).mean()
        out[sym] = g
    return out


def _simulate_baseline(high, low, close, entry_idx, entry_price, sl, tp, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
        if high[i] >= tp:
            return tp / entry_price - 1
    return close[last_i] / entry_price - 1


def _simulate_volume_decay(
    high, low, close, rvol, entry_idx, entry_price, sl, tp, horizon, threshold
):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
        if not np.isnan(rvol[i]) and rvol[i] < threshold:
            return close[i] / entry_price - 1
        if high[i] >= tp:
            return tp / entry_price - 1
    return close[last_i] / entry_price - 1


def run() -> None:
    sig_df = _fetch_signals("Long")
    if len(sig_df) < 50:
        print("HA_Cross — Long: yetersiz sinyal, atlanıyor")
        return

    baseline_rets = []
    decay_rets = {th: [] for th in RVOL_THRESHOLDS}
    opened_ats = []

    for interval, sub in sig_df.groupby("interval"):
        bars = _fetch_execution_bars(sub["symbol"].unique().tolist(), interval)
        horizon = HORIZON_BARS.get(interval, 96)

        for row in sub.itertuples():
            g = bars.get(row.symbol)
            if g is None or len(g) < RVOL_WINDOW + 5:
                continue

            opened_at = pd.Timestamp(row.opened_at)
            idx = g["ts"].searchsorted(opened_at, side="right") - 1
            if idx < RVOL_WINDOW or idx >= len(g) - 1:
                continue

            entry = float(row.open_price)
            atr = float(row.atr)
            sl = entry - SL_MULT * atr
            tp = entry + TP_MULT * atr
            if not (sl < entry < tp):
                continue

            high = g["high"].to_numpy(float)
            low = g["low"].to_numpy(float)
            close = g["close"].to_numpy(float)
            rvol = g["rvol"].to_numpy(float)

            baseline_rets.append(_simulate_baseline(high, low, close, idx, entry, sl, tp, horizon))
            for th in RVOL_THRESHOLDS:
                decay_rets[th].append(
                    _simulate_volume_decay(high, low, close, rvol, idx, entry, sl, tp, horizon, th)
                )
            opened_ats.append(opened_at)

    print(
        f"\n{'='*70}\nHA_Cross — Long — hacim-sönmesi erken çıkışı  (n={len(baseline_rets)})\n{'='*70}"
    )

    ts_arr = pd.Series(opened_ats)
    mid = ts_arr.min() + (ts_arr.max() - ts_arr.min()) / 2
    first_mask = (ts_arr < mid).to_numpy()
    print(f"dönem: {ts_arr.min()} .. {ts_arr.max()} | orta nokta: {mid}\n")

    print(f"{'strateji':28} {'dönem':12} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")

    def _print_row(name, rets):
        arr = np.array(rets)
        for label, mask in (
            ("tum", np.ones(len(arr), dtype=bool)),
            ("ilk_yari", first_mask),
            ("ikinci_yari", ~first_mask),
        ):
            s = _stats(arr[mask])
            print(
                f"{name:28} {label:12} {s.get('n',0):>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}"
            )

    _print_row("ATR-baz-SL/TP (mevcut)", baseline_rets)
    for th in RVOL_THRESHOLDS:
        _print_row(f"+ hacim-sönmesi <{th}", decay_rets[th])


if __name__ == "__main__":
    run()
