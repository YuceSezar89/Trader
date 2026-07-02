"""
DO Kırılımı backtest entry'leri üzerinde filtre analizi:

  1. Ayrışma: coin'in gün-içi getirisi (DO'ya göre) − BTC'nin gün-içi getirisi.
     Pozitif ayrışan = kendi başına hareket eden coin.
  2. BTC rejimi: BTC aynı TF'de EMA200 üstünde mi + BTC kendi DO'sunun üstünde mi.
  3. SL taraması: 1.5 / 2.0 / 2.5 / 3.0 × ATR (TP = 2×SL, canlı R:R korunur).

Girdi: logs/backtest_do_kirilimi_{tf}.csv (backtest_do_kirilimi.py çıktısı)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import pandas_ta_classic as pta

from config import Config

SL_SWEEP = [1.5, 2.0, 2.5, 3.0]
RR = 2.0            # TP = RR × SL
SIM_MAX_BARS = 1000
DO_HOUR = 3


def load_bars(tf: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket, open, high, low, close
        FROM cagg_{tf}
        WHERE symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.drop(columns="symbol").reset_index(drop=True)
        g[["open", "high", "low", "close"]] = g[["open", "high", "low", "close"]].astype(float)
        out[sym] = g
    return out


def daily_open_series(df: pd.DataFrame) -> np.ndarray:
    day_key = (df["bucket"] - pd.Timedelta(hours=DO_HOUR)).dt.date
    new_day = (day_key != day_key.shift(1)).to_numpy()
    new_day[0] = True
    do = np.where(new_day, df["open"].to_numpy(), np.nan)
    return pd.Series(do).ffill().to_numpy()


def simulate(h, l, c, i0, entry, atr, sl_mult):
    sl = entry - sl_mult * atr
    tp = entry + RR * sl_mult * atr
    dist = sl_mult * atr
    trail = np.nan
    n = len(c)
    for j in range(i0 + 1, min(i0 + SIM_MAX_BARS, n)):
        if np.isnan(trail):
            if l[j] <= sl:
                return (sl - entry) / entry * 100
            if h[j] >= tp:
                trail = tp - dist
        else:
            nt = h[j] - dist
            if nt > trail:
                trail = nt
            if l[j] <= trail:
                return (trail - entry) / entry * 100
    j = min(i0 + SIM_MAX_BARS, n) - 1
    return (c[j] - entry) / entry * 100


def stat_line(pnls: pd.Series) -> str:
    if len(pnls) == 0:
        return "n=0"
    wr = (pnls > 0).mean() * 100
    g = pnls[pnls > 0].sum()
    b = -pnls[pnls < 0].sum()
    pf = g / b if b > 0 else float("inf")
    return f"n={len(pnls):>4}  wr={wr:5.1f}%  ort={pnls.mean():+.3f}%  pf={pf:4.2f}"


def main() -> None:
    for tf in ("15m", "5m"):
        path = f"logs/backtest_do_kirilimi_{tf}.csv"
        if not os.path.exists(path):
            print(f"{path} yok, atlanıyor")
            continue
        tr = pd.read_csv(path, parse_dates=["ts"])
        symbols = sorted(tr["symbol"].unique().tolist())
        bars = load_bars(tf, symbols + ["BTCUSDT"])
        btc = bars["BTCUSDT"]
        btc_idx = {t: i for i, t in enumerate(btc["bucket"])}
        btc_do = daily_open_series(btc)
        btc_c = btc["close"].to_numpy()
        btc_ema200 = btc["close"].ewm(span=200, adjust=False).mean().to_numpy()

        # Sembol başına DO + ATR serileri (CSV'de yok, bar verisinden)
        sym_do = {s: daily_open_series(g) for s, g in bars.items()}
        sym_atr = {s: pta.atr(g["high"], g["low"], g["close"], length=14).to_numpy()
                   for s, g in bars.items()}
        sym_idx = {s: {t: i for i, t in enumerate(g["bucket"])} for s, g in bars.items()}

        # Her entry için ayrışma + rejim metrikleri
        rows = []
        for r in tr.itertuples():
            sym_df = bars.get(r.symbol)
            if sym_df is None:
                continue
            ts = pd.Timestamp(r.ts)
            bi = btc_idx.get(ts)
            si = sym_idx[r.symbol].get(ts)
            if bi is None or si is None:
                continue
            c_do = sym_do[r.symbol][si]
            coin_day_ret = (r.price / c_do - 1) * 100 if c_do and c_do > 0 else np.nan
            b_do = btc_do[bi]
            btc_day_ret = (btc_c[bi] / b_do - 1) * 100 if b_do and b_do > 0 else np.nan
            rows.append(dict(
                idx=r.Index,
                ayrisma=coin_day_ret - btc_day_ret,
                btc_trend_up=btc_c[bi] > btc_ema200[bi],
                btc_day_up=btc_day_ret > 0,
            ))
        met = pd.DataFrame(rows).set_index("idx")
        tr = tr.join(met, how="inner")

        # SL taraması: her entry'yi her SL ile yeniden simüle et
        sl_cols = {}
        for sl_mult in SL_SWEEP:
            pnls = []
            for r in tr.itertuples():
                g = bars[r.symbol]
                mask = g["bucket"] == pd.Timestamp(r.ts)
                if not mask.any():
                    pnls.append(np.nan)
                    continue
                i0 = int(g.index[mask][0])
                atr = sym_atr[r.symbol][i0]
                if not np.isfinite(atr) or atr <= 0:
                    pnls.append(np.nan)
                    continue
                pnls.append(simulate(
                    g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy(),
                    i0, r.price, atr, sl_mult,
                ))
            sl_cols[sl_mult] = pd.Series(pnls, index=tr.index)

        print(f"\n{'═' * 78}\n{tf} — {len(tr)} entry | filtre × SL matrisi (TP = 2×SL + trailing)\n{'═' * 78}")

        filters = {
            "hepsi":                 pd.Series(True, index=tr.index),
            "ayrışma > 0":           tr["ayrisma"] > 0,
            "ayrışma > +1%":         tr["ayrisma"] > 1.0,
            "BTC trend up":          tr["btc_trend_up"],
            "BTC gün+ & ayrışma>0":  tr["btc_day_up"] & (tr["ayrisma"] > 0),
            "trend+ & ayrışma>0":    tr["btc_trend_up"] & (tr["ayrisma"] > 0),
            "trend+ & ayrışma>+1%":  tr["btc_trend_up"] & (tr["ayrisma"] > 1.0),
        }
        for fname, fmask in filters.items():
            print(f"\n  ▸ {fname}")
            for sl_mult in SL_SWEEP:
                pnls = sl_cols[sl_mult][fmask].dropna()
                print(f"      SL {sl_mult:.1f}×ATR  {stat_line(pnls)}")

        # Ayrışma dağılımı bilgisi
        print(f"\n  ayrışma dağılımı: medyan={tr['ayrisma'].median():+.2f}%  "
              f">0 olan={((tr['ayrisma'] > 0).mean() * 100):.0f}%  "
              f"BTC trend up olan={(tr['btc_trend_up'].mean() * 100):.0f}%")


if __name__ == "__main__":
    main()
