"""
ha_cross_rvol_exit_bt.py'nin DÜZELTİLMİŞ hali: ham/tek-barlık RVOL eşiği yerine
EVOL'ün kendi disiplinini (evol_bt.py — EMA(7) yumuşatma + son-100-bar
percentile-rank, [[project_devisso_ersi]]) kullanıyor. İlk deneme (ham RVOL<eşik)
medyan 2 bar'da tetikleniyordu (gürültü, gerçek "hacim sönmesi" değil) — bu
sürüm EVOL skorunu HER barda dinamik hesaplayıp (o ana kadarki geçmişle,
look-ahead yok) + minimum bekleme süresi (min_hold) ekleyerek aynı artefaktı
önlüyor.

Çıkış kuralı: pozisyon en az `min_hold` bar açık kaldıktan sonra, EVOL skoru
eşiğin (varsayılan <35, evol_bt.py'nin "düşük" bandı) altına düşerse ERKEN çık.
SL/TP mevcut ATR-bazlı politika (risk_policy.py), değişmiyor.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_pivot_tp_bt import _fetch_signals  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

DAYS = 60
SL_MULT = Config.RISK_SL_MULTIPLIER   # 1.5
TP_MULT = Config.RISK_TP_MULTIPLIER   # 3.0
HORIZON_HOURS = 24.0
HORIZON_BARS = {"5m": int(HORIZON_HOURS * 12), "15m": int(HORIZON_HOURS * 4)}
RVOL_WINDOW = 20
RANK_WINDOW = 100
EVOL_THRESHOLDS = [25, 35, 45]
MIN_HOLD_BARS = 5


def _rolling_percentile_rank(values: np.ndarray, window: int) -> np.ndarray:
    """values[i]'nin, önceki `window` bar İÇİNDEKİ (kendisi dahil) dağılıma göre
    percentile-rank'i — causal, look-ahead yok."""
    n = len(values)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    windows = sliding_window_view(values, window)  # windows[k] = values[k:k+window]
    last_vals = windows[:, -1]
    with np.errstate(invalid="ignore"):
        ranks = (windows < last_vals[:, None]).mean(axis=1) * 100
    out[window - 1:] = ranks
    return out


def _evol_series(df: pd.DataFrame) -> np.ndarray:
    """evol_bt.py::_compute_evol ile AYNI formül/smoothing, tek bir noktada
    değil TÜM seri boyunca (causal rolling) hesaplanmış hali."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    price_pct = close.pct_change() * 100.0
    rvol = volume / volume.rolling(RVOL_WINDOW).mean()
    raw = price_pct / rvol.replace(0.0, np.nan)
    smoothed = raw.ewm(span=7, adjust=False).mean().to_numpy()
    return _rolling_percentile_rank(smoothed, RANK_WINDOW)


def _fetch_execution_bars(symbols: list, interval: str) -> dict:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
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
        g["evol"] = _evol_series(g)
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


def _simulate_evol_exit(high, low, close, evol, entry_idx, entry_price, sl, tp, horizon, threshold, min_hold):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if low[i] <= sl:
            return sl / entry_price - 1
        if (i - entry_idx) >= min_hold and not np.isnan(evol[i]) and evol[i] < threshold:
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
    evol_rets = {th: [] for th in EVOL_THRESHOLDS}
    bars_held = {th: [] for th in EVOL_THRESHOLDS}
    opened_ats = []

    for interval, sub in sig_df.groupby("interval"):
        bars = _fetch_execution_bars(sub["symbol"].unique().tolist(), interval)
        horizon = HORIZON_BARS.get(interval, 96)

        for row in sub.itertuples():
            g = bars.get(row.symbol)
            if g is None or len(g) < RANK_WINDOW + RVOL_WINDOW + 5:
                continue

            opened_at = pd.Timestamp(row.opened_at)
            idx = g["ts"].searchsorted(opened_at, side="right") - 1
            if idx < RANK_WINDOW + RVOL_WINDOW or idx >= len(g) - 1:
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
            evol = g["evol"].to_numpy(float)

            baseline_rets.append(_simulate_baseline(high, low, close, idx, entry, sl, tp, horizon))
            for th in EVOL_THRESHOLDS:
                n0 = len(close)
                last_i = min(idx + horizon, n0 - 1)
                held = last_i - idx
                for i in range(idx + 1, last_i + 1):
                    if low[i] <= sl or ((i - idx) >= MIN_HOLD_BARS and not np.isnan(evol[i]) and evol[i] < th) or high[i] >= tp:
                        held = i - idx
                        break
                bars_held[th].append(held)
                evol_rets[th].append(
                    _simulate_evol_exit(high, low, close, evol, idx, entry, sl, tp, horizon, th, MIN_HOLD_BARS)
                )
            opened_ats.append(opened_at)

    print(f"\n{'='*70}\nHA_Cross — Long — EVOL-disiplinli erken çıkış (min_hold={MIN_HOLD_BARS} bar)  "
          f"(n={len(baseline_rets)})\n{'='*70}")

    ts_arr = pd.Series(opened_ats)
    mid = ts_arr.min() + (ts_arr.max() - ts_arr.min()) / 2
    first_mask = (ts_arr < mid).to_numpy()
    print(f"dönem: {ts_arr.min()} .. {ts_arr.max()} | orta nokta: {mid}\n")

    print(f"{'strateji':24} {'dönem':12} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")

    def _print_row(name, rets):
        arr = np.array(rets)
        for label, mask in (("tum", np.ones(len(arr), dtype=bool)), ("ilk_yari", first_mask), ("ikinci_yari", ~first_mask)):
            s = _stats(arr[mask])
            print(f"{name:24} {label:12} {s.get('n',0):>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    _print_row("ATR-baz-SL/TP (mevcut)", baseline_rets)
    for th in EVOL_THRESHOLDS:
        _print_row(f"+ EVOL-çıkış <{th}", evol_rets[th])
        held_arr = np.array(bars_held[th])
        print(f"  (medyan tutma={np.median(held_arr):.1f} bar, 1-2 barda kapanan oran={(held_arr<=2).mean():.1%})")


if __name__ == "__main__":
    run()
