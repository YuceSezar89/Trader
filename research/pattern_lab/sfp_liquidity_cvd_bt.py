"""
Kullanıcının kendi yöntemi: likidite alımı (equal high/low süpürmesi) + SFP
(Swing Failure Pattern — fitille geçip geri kapanma) + CVD uyumsuzluğu — Tlosx'un
BSLSSL likidite motoru + footprint v3'ün SWEEP/Delta-Divergence bileşenlerinin
birleşimi. Önceki test (rsi_cross_cvd_divergence_bt.py) bunu RSI_Cross'a bağlıyordu
ve "son 20 barın dibi" gibi kaba bir referans kullanıyordu — burada GERÇEK likidite
seviyesi (≥3 kez test edilmiş eşit tepe/dip) + gerçek SFP (fitille aşıp geri
kapanma) + kendi başına bir sinyal (RSI_Cross'a bağımlı değil).

CVD: [[project_cvd_divergence]] — gerçek taker buy/sell hacminden (price_data,
15m'ye toplanmış), geometrik proxy değil.

Look-ahead yok: likidite seviyesi SADECE mevcut barın ÖNCESİNDEKİ WINDOW bardan
kuruluyor (shift(1) + rolling), SFP/CVD kontrolü mevcut barda, ileri getiri
ayrı ölçülüyor.
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

DAYS = 20  # 22 Haziran'dan itibaren tam sembol kapsamı (480-605) var; öncesi delikli
MIN_BARS = 150
WINDOW = 100  # likidite seviyesi arama penceresi (bar)
TOUCH_TOL = 0.002  # %0.2 — find_support_resistance ile aynı tolerans
MIN_TOUCHES = 3  # Tlosx BSLSSL: "3+ eşit tepe/dip kümesi = likidite havuzu"
MIN_DEPTH_ATR = 0.4  # Tlosx SWEEP: "Min Süpürme Derinliği 0.4xATR ~%60 isabet"
ATR_PERIOD = 14
HORIZONS = {"4s": 16, "12s": 48, "24s": 96}


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, time_bucket('15 minutes', timestamp) AS ts,
               first(open, timestamp) AS open, max(high) AS high, min(low) AS low,
               last(close, timestamp) AS close,
               sum(buy_volume) AS buy_volume, sum(sell_volume) AS sell_volume
        FROM price_data
        WHERE interval = '1m' AND timestamp > NOW() - INTERVAL '{DAYS} days'
          AND buy_volume IS NOT NULL
        GROUP BY symbol, time_bucket('15 minutes', timestamp)
        ORDER BY symbol, ts
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _sfp_cvd_signals(g: pd.DataFrame) -> tuple:
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    close = g["close"].to_numpy(float)
    cvd = (g["buy_volume"] - g["sell_volume"]).cumsum().to_numpy()
    n = len(g)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(ATR_PERIOD).mean().to_numpy()

    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    if n <= WINDOW:
        return buy_sig, sell_sig

    win_low = sliding_window_view(low, WINDOW)  # win_low[k] = low[k : k+WINDOW]
    win_high = sliding_window_view(high, WINDOW)
    win_min = win_low.min(axis=1)
    win_max = win_high.max(axis=1)
    win_argmin = win_low.argmin(axis=1)
    win_argmax = win_high.argmax(axis=1)
    touches_low = (win_low <= win_min[:, None] * (1 + TOUCH_TOL)).sum(axis=1)
    touches_high = (win_high >= win_max[:, None] * (1 - TOUCH_TOL)).sum(axis=1)

    # bar i için pencere k = i - WINDOW (low[i-WINDOW : i], mevcut bar HARİÇ)
    for i in range(WINDOW, n):
        k = i - WINDOW
        if np.isnan(atr[i]):
            continue

        if touches_low[k] >= MIN_TOUCHES:
            level = win_min[k]
            depth = level - low[i]
            if depth >= MIN_DEPTH_ATR * atr[i] and close[i] > level:
                level_idx = k + int(win_argmin[k])
                if cvd[i] > cvd[level_idx]:
                    buy_sig[i] = True

        if touches_high[k] >= MIN_TOUCHES:
            level = win_max[k]
            depth = high[i] - level
            if depth >= MIN_DEPTH_ATR * atr[i] and close[i] < level:
                level_idx = k + int(win_argmax[k])
                if cvd[i] < cvd[level_idx]:
                    sell_sig[i] = True

    return buy_sig, sell_sig


def run() -> None:
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(
        f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar\n"
        f"dönem: {t_min} .. {t_max} | orta nokta: {mid}\n"
    )

    labels = ["baseline", "buy", "sell"]
    halves = ["tum", "ilk_yari", "ikinci_yari"]
    res = {h: {lbl: {half: [] for half in halves} for lbl in labels} for h in HORIZONS}
    n_syms, n_buy_events, n_sell_events = 0, 0, 0

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        ts_np = g["ts"].to_numpy()
        close = g["close"].to_numpy(float)
        max_h = max(HORIZONS.values())
        buy_sig, sell_sig = _sfp_cvd_signals(g)
        n_buy_events += int(buy_sig.sum())
        n_sell_events += int(sell_sig.sum())

        all_idx = np.arange(WINDOW, len(g) - max_h)
        idx_by_label = {
            "baseline": all_idx,
            "buy": all_idx[buy_sig[all_idx]],
            "sell": all_idx[sell_sig[all_idx]],
        }
        for lbl, idxs in idx_by_label.items():
            if len(idxs) == 0:
                continue
            first_mask = ts_np[idxs] < np.datetime64(mid)
            for h_name, h_bars in HORIZONS.items():
                rets = close[idxs + h_bars] / close[idxs] - 1
                if lbl == "sell":
                    rets = -rets
                res[h_name][lbl]["tum"].append(rets)
                res[h_name][lbl]["ilk_yari"].append(rets[first_mask])
                res[h_name][lbl]["ikinci_yari"].append(rets[~first_mask])

    print(
        f"analiz edilen sembol: {n_syms} | SFP+CVD Long olay: {n_buy_events} | Short olay: {n_sell_events}\n"
    )
    for h_name in HORIZONS:
        print(f"── ufuk: {h_name} ──")
        print(f"{'grup':10} {'dönem':12} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for lbl in labels:
            for half in halves:
                arrs = res[h_name][lbl][half]
                rets = np.concatenate(arrs) if arrs else np.array([])
                s = _stats(rets)
                print(
                    f"{lbl:10} {half:12} {s.get('n',0):>7} {s.get('wr',0):>6} "
                    f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
                )
        print()


if __name__ == "__main__":
    run()
