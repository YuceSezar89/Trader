"""
HA_Cross sinyallerinde TP mekanizmasını test eder: mevcut ATR-bazlı TEK TP
(signals/risk_policy.py::default_policy tabanı — SL=1.5×ATR, TP=3.0×ATR) vs
Fibonacci Pivot tabanlı KADEMELİ TP (TP1=R1/S1'de yarı pozisyon, TP2=R2/S2'de
kalan yarı) — calculate_fib_pivots (indicators/core.py, panelle [[project_chart_panel_pivots]]
AYNI fonksiyon).

Gerekçe: paper_trades'te 107 kapalı ha_cross işleminin HİÇBİRİ take_profit'e
ulaşmamış (63 stop_loss, 42 trailing_stop, 2 manual) — mevcut TP fiyat
yapısından bağımsız (salt ATR mesafesi), muhtemelen ya hiç isabet etmiyor ya
da trailing stop ondan önce devreye giriyor.

SL ikisinde de AYNI (1.5×ATR, mevcut politika) — sadece TP mekanizması
değişiyor, izole karşılaştırma.

Look-ahead yok: pivot seviyeleri DAİMA önceki takvim gününün H/L/C'sinden
(cagg_4h'den günlük'e resample + shift(1)). Bar-içi sıra bilinmiyor (OHLC'de
hangisi önce değdi belli değil) — SL/TP1/TP2 kontrolü SL-önce varsayımıyla
KONSERVATİF (do_break_gauss_sltp_bt.py ile aynı disiplin).
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_fib_pivots  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

DAYS = 60
SL_MULT = Config.RISK_SL_MULTIPLIER   # 1.5
TP_MULT = Config.RISK_TP_MULTIPLIER   # 3.0
HORIZON_HOURS = 24.0
HORIZON_BARS = {"5m": int(HORIZON_HOURS * 12), "15m": int(HORIZON_HOURS * 4)}


def _fetch_signals(direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, interval, open_price, atr, opened_at
        FROM signals
        WHERE indicators = 'HA_Cross' AND signal_type = %s AND status = 'closed'
          AND atr IS NOT NULL AND atr > 0 AND open_price IS NOT NULL
        ORDER BY symbol, opened_at
    """
    df = pd.read_sql(q, conn, params=(direction,))
    conn.close()
    return df


def _fetch_daily_pivots(symbols: list) -> dict:
    """symbol -> {date: {pp,r1,r2,r3,s1,s2,s3}} — 4h barlardan günlük H/L/C,
    ÖNCEKİ günün pivotu bugüne uygulanacak şekilde shift(1) edilmiş."""
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, high, low, close
        FROM cagg_4h
        WHERE bucket > NOW() - INTERVAL '{DAYS + 5} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()

    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        daily = g.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily) < 2:
            continue
        pivots = daily.apply(lambda r: calculate_fib_pivots(r["high"], r["low"], r["close"]), axis=1)
        pivots.index = pivots.index.date
        pivots = pivots.shift(1)  # bugünün pivotu DÜNÜN H/L/C'sinden
        out[sym] = pivots.dropna().to_dict()
    return out


def _fetch_execution_bars(symbols: list, interval: str) -> dict:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, high, low, close
        FROM cagg_{interval}
        WHERE bucket > NOW() - INTERVAL '{DAYS} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        out[sym] = g
    return out


def _simulate_single(high, low, close, entry_idx, entry_price, is_long, sl, tp, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    for i in range(entry_idx + 1, last_i + 1):
        if is_long:
            if low[i] <= sl:
                return sl / entry_price - 1
            if high[i] >= tp:
                return tp / entry_price - 1
        else:
            if high[i] >= sl:
                return 1 - sl / entry_price
            if low[i] <= tp:
                return 1 - tp / entry_price
    return (close[last_i] / entry_price - 1) if is_long else (1 - close[last_i] / entry_price)


def _simulate_ladder(high, low, close, entry_idx, entry_price, is_long, sl, tp1, tp2, horizon):
    n = len(close)
    last_i = min(entry_idx + horizon, n - 1)
    position = 1.0
    realized = 0.0
    tp1_taken = False
    for i in range(entry_idx + 1, last_i + 1):
        hit_sl = (low[i] <= sl) if is_long else (high[i] >= sl)
        if hit_sl:
            leg = (sl / entry_price - 1) if is_long else (1 - sl / entry_price)
            return realized + position * leg
        if not tp1_taken:
            hit_tp1 = (high[i] >= tp1) if is_long else (low[i] <= tp1)
            if hit_tp1:
                leg = (tp1 / entry_price - 1) if is_long else (1 - tp1 / entry_price)
                realized += 0.5 * leg
                position = 0.5
                tp1_taken = True
        if tp1_taken:
            hit_tp2 = (high[i] >= tp2) if is_long else (low[i] <= tp2)
            if hit_tp2:
                leg = (tp2 / entry_price - 1) if is_long else (1 - tp2 / entry_price)
                return realized + position * leg
    final_leg = (close[last_i] / entry_price - 1) if is_long else (1 - close[last_i] / entry_price)
    return realized + position * final_leg


def run() -> None:
    for direction in ("Long", "Short"):
        is_long = direction == "Long"
        sig_df = _fetch_signals(direction)
        if len(sig_df) < 50:
            print(f"HA_Cross — {direction}: yetersiz sinyal, atlanıyor")
            continue

        symbols = sig_df["symbol"].unique().tolist()
        pivots = _fetch_daily_pivots(symbols)

        baseline_rets, ladder_rets, opened_ats = [], [], []
        skipped_no_pivot, skipped_bad_geometry = 0, 0

        for interval, sub in sig_df.groupby("interval"):
            bars = _fetch_execution_bars(sub["symbol"].unique().tolist(), interval)
            horizon = HORIZON_BARS.get(interval, 96)

            for row in sub.itertuples():
                g = bars.get(row.symbol)
                if g is None or len(g) < 20:
                    continue
                sym_pivots = pivots.get(row.symbol)
                if not sym_pivots:
                    skipped_no_pivot += 1
                    continue

                opened_at = pd.Timestamp(row.opened_at)
                pv = sym_pivots.get(opened_at.date())
                if pv is None:
                    skipped_no_pivot += 1
                    continue

                idx = g["ts"].searchsorted(opened_at, side="right") - 1
                if idx < 0 or idx >= len(g) - 1:
                    continue

                entry = float(row.open_price)
                atr = float(row.atr)
                sl = entry - SL_MULT * atr if is_long else entry + SL_MULT * atr
                tp_base = entry + TP_MULT * atr if is_long else entry - TP_MULT * atr
                tp1 = pv["r1"] if is_long else pv["s1"]
                tp2 = pv["r2"] if is_long else pv["s2"]

                valid = (tp1 > entry > sl) if is_long else (tp1 < entry < sl)
                valid = valid and ((tp2 > tp1) if is_long else (tp2 < tp1))
                if not valid:
                    skipped_bad_geometry += 1
                    continue

                high = g["high"].to_numpy(float)
                low = g["low"].to_numpy(float)
                close = g["close"].to_numpy(float)

                baseline_rets.append(_simulate_single(high, low, close, idx, entry, is_long, sl, tp_base, horizon))
                ladder_rets.append(_simulate_ladder(high, low, close, idx, entry, is_long, sl, tp1, tp2, horizon))
                opened_ats.append(opened_at)

        print(f"\n{'='*70}\nHA_Cross — {direction}  "
              f"(n={len(baseline_rets)}, pivot-yok={skipped_no_pivot}, geometri-bozuk={skipped_bad_geometry})\n{'='*70}")

        ts_arr = pd.Series(opened_ats)
        mid = ts_arr.min() + (ts_arr.max() - ts_arr.min()) / 2
        first_mask = (ts_arr < mid).to_numpy()
        print(f"dönem: {ts_arr.min()} .. {ts_arr.max()} | orta nokta: {mid}\n")

        print(f"{'strateji':22} {'dönem':12} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for name, rets in (("ATR-tek-TP (mevcut)", baseline_rets), ("Pivot-ladder (TP1/TP2)", ladder_rets)):
            arr = np.array(rets)
            for label, mask in (("tum", np.ones(len(arr), dtype=bool)), ("ilk_yari", first_mask), ("ikinci_yari", ~first_mask)):
                s = _stats(arr[mask])
                print(f"{name:22} {label:12} {s.get('n',0):>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}")


if __name__ == "__main__":
    run()
