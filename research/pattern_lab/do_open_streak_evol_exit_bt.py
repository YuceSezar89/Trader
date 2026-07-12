"""
do_open_streak (canlıda ENABLED_STRATEGIES'te aktif, HA_Cross'un aksine) için
EVOL-disiplinli erken çıkış testi — ha_cross_evol_exit_bt.py'nin AYNI metodolojisi
(EMA(7)+son-100-bar percentile-rank, min_hold bar şartı), ama farklı bir taban
strateji üzerinde: signals/do_open_streak.py::DoOpenStreakDetector.check() ile
BİREBİR AYNI giriş mantığı (D-open kırılım + 3 ardışık yeşil + Gauss>=4.5),
BİREBİR AYNI çıkış mekaniği (SL=3×ATR TEK BAŞINA, TP YOK, 24h timeout —
[[project_paper_trading]]).

do_open_streak sinyalleri `signals` tablosunda YOK (detector-bazlı, canlıda
henüz 0 kapalı paper_trade var, ~birkaç saatlik veri) — bu yüzden üretim
dedektörünün mantığı GEÇMİŞ cagg_15m verisine vektörel olarak (bar-bar check()
çağırmak yerine, ama BİREBİR aynı state machine) uygulanıyor.

Look-ahead yok: sinyal SADECE geçmiş barlarla üretiliyor (production check()
ile aynı), EVOL causal rolling, min_hold ile artefakt önleniyor
([[project_devisso_ersi]] — ha_cross_evol_exit_bt.py'de bulunan ders).
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_evol_exit_bt import (  # pylint: disable=wrong-import-position
    RANK_WINDOW,
    RVOL_WINDOW,
    _evol_series,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from signals.do_open_streak import (  # pylint: disable=wrong-import-position
    GAUSS_THRESHOLD,
    MAX_HOLD_HOURS,
    SL_ATR_MULT,
    STREAK_THRESHOLD,
    _gauss_sum,
)

DAYS = 60
MIN_BARS = 250
WARMUP = 200
HORIZON_BARS = int(MAX_HOLD_HOURS * 4)  # 24h @ 15m = 96 bar
EVOL_THRESHOLD = 25
EVOL_MIN_HOLD_BARS = 8


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _signal_series(g: pd.DataFrame) -> np.ndarray:
    """DoOpenStreakDetector.check() ile BİREBİR AYNI mantık, tüm seri üzerinde."""
    o = g["open"].to_numpy(float)
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    c = g["close"].to_numpy(float)
    n = len(c)
    ts = g["ts"]

    daily_open, _ = _daily_open(ts, o)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break = (c > daily_open) & (prev_c <= daily_open) & np.isfinite(daily_open)
    is_long = c > o

    gate_active = False
    count_long = 0
    start_low = np.nan
    signal = np.zeros(n, dtype=bool)
    for i in range(n):
        if do_break[i]:
            gate_active = True
        elif not is_long[i]:
            gate_active = False

        if is_long[i]:
            count_long += 1
            if count_long == 1 or np.isnan(start_low):
                start_low = l[i]
        else:
            count_long = 0
            start_low = np.nan

        if gate_active and count_long == STREAK_THRESHOLD:
            long_perc = (h[i] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val >= GAUSS_THRESHOLD:
                signal[i] = True
    return signal


def _simulate_baseline(low, close, entry_idx, entry_price, sl, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
    return close[last_i] / entry_price - 1


def _simulate_evol_exit(low, close, evol, entry_idx, entry_price, sl, horizon, threshold, min_hold):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
        if (i - entry_idx) >= min_hold and not np.isnan(evol[i]) and evol[i] < threshold:
            return close[i] / entry_price - 1
    return close[last_i] / entry_price - 1


def run() -> None:
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_rets, evol_rets, opened_ats = [], [], []
    n_syms, n_events = 0, 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        signal = _signal_series(g)
        atr_series = calculate_atr(g, period=14).to_numpy()
        evol = _evol_series(g)
        close = g["close"].to_numpy(float)
        low = g["low"].to_numpy(float)
        ts_np = g["ts"].to_numpy()

        idxs = np.where(signal)[0]
        idxs = idxs[
            (idxs >= max(WARMUP, RANK_WINDOW + RVOL_WINDOW)) & (idxs < len(g) - HORIZON_BARS)
        ]

        for i in idxs:
            atr = atr_series[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = close[i]
            sl = entry - SL_ATR_MULT * atr
            n_events += 1
            baseline_rets.append(_simulate_baseline(low, close, i, entry, sl, HORIZON_BARS))
            evol_rets.append(
                _simulate_evol_exit(
                    low, close, evol, i, entry, sl, HORIZON_BARS, EVOL_THRESHOLD, EVOL_MIN_HOLD_BARS
                )
            )
            opened_ats.append(pd.Timestamp(ts_np[i]))

    print(f"analiz edilen sembol: {n_syms} | üretim-eşleşmeli olay: {n_events}\n")
    if n_events < 30:
        print("Örneklem çok küçük, güvenilir yorum yapılamaz.")
        return

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

    _print_row("SL=3xATR tek başına (mevcut)", baseline_rets)
    _print_row(f"+ EVOL-çıkış <{EVOL_THRESHOLD} (mh={EVOL_MIN_HOLD_BARS})", evol_rets)


if __name__ == "__main__":
    run()
