"""
v2-1 — "Para kazandıran bölge" (ST-equity güven filtresi) replay backtest.

Doktrin (Turtle Traders): varsayılan ST ayarlarıyla her sinyalde işlem
yaparsan yatayda kaybedersin; strateji equity'si kâra geçtiği bölgede
sinyallere güvenilir.

Test: ST(10,3) long-only flip stratejisi, 1h barlar, tüm semboller.
  A kolu: tüm işlemler
  B kolu: girişte equity > MA(son 10 kapanmış işlem) ise gir
  C kolu: tersi (equity MA altında) — kontrast

Look-ahead koruması: filtre kararı yalnızca GİRİŞ anından önce kapanmış
işlemlerin equity'sini görür. Ücret/kayma yok (göreli kıyas), not düşülür.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from config import Config

DAYS = 90
ST_LEN, ST_MULT = 10, 3.0
MA_TRADES = 10
MIN_BARS = 500
MIN_TRADES = 8


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, high, low, close
        FROM cagg_1h
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _st_trades(g: pd.DataFrame) -> list[tuple]:
    """ST(10,3) long-only flip işlemleri: (entry_ts, ret)."""
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    c = g["close"].to_numpy(float)
    n = len(g)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / ST_LEN, adjust=False).mean().to_numpy()
    mid = (h + l) / 2
    ub, lb = mid + ST_MULT * atr, mid - ST_MULT * atr
    f_ub, f_lb = ub.copy(), lb.copy()
    d = np.ones(n, dtype=int)
    for i in range(1, n):
        f_ub[i] = min(ub[i], f_ub[i - 1]) if c[i - 1] <= f_ub[i - 1] else ub[i]
        f_lb[i] = max(lb[i], f_lb[i - 1]) if c[i - 1] >= f_lb[i - 1] else lb[i]
        if c[i] > f_ub[i - 1]:
            d[i] = 1
        elif c[i] < f_lb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    trades, entry = [], None
    ts = g["ts"].to_numpy()
    for i in range(1, n):
        if d[i] == 1 and d[i - 1] == -1:
            entry = (ts[i], c[i])
        elif d[i] == -1 and d[i - 1] == 1 and entry:
            trades.append((entry[0], c[i] / entry[1] - 1))
            entry = None
    return trades


def _stats(rets: list[float]) -> dict:
    r = pd.Series(rets, dtype=float)
    if r.empty:
        return {"n": 0}
    g, b = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n": len(r),
        "wr": round(float((r > 0).mean() * 100), 1),
        "ort_%": round(float(r.mean() * 100), 3),
        "pf": round(float(g / b), 3) if b > 0 else float("inf"),
        "toplam_%": round(float(r.sum() * 100), 1),
    }


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 1h bar ({DAYS} gün)")

    # Önceden beyan edilen 3 filtre tanımı (başka tanım denenmeyecek):
    #   MA10 : equity > son 10 işlem equity ortalaması
    #   HWM  : equity yeni zirvede (>= tüm geçmiş max) — "kâra geçti" okuması
    #   SON5 : son 5 işlemin ortalaması pozitif
    arms = {"A_tum": [], "B_ma10": [], "B_hwm": [], "B_son5": []}
    n_syms = 0
    for _, g in df.groupby("symbol"):
        if len(g) < MIN_BARS:
            continue
        trades = _st_trades(g.reset_index(drop=True))
        if len(trades) < MIN_TRADES:
            continue
        n_syms += 1
        equity: list[float] = []
        rets_hist: list[float] = []
        for _, ret in trades:
            if len(equity) >= 5:
                cur = equity[-1]
                if len(equity) >= MA_TRADES and cur > np.mean(equity[-MA_TRADES:]):
                    arms["B_ma10"].append(ret)
                if cur >= max(equity):
                    arms["B_hwm"].append(ret)
                if np.mean(rets_hist[-5:]) > 0:
                    arms["B_son5"].append(ret)
            arms["A_tum"].append(ret)
            equity.append((equity[-1] if equity else 1.0) * (1 + ret))
            rets_hist.append(ret)

    print(f"analize giren sembol: {n_syms}")
    print(f"\n{'kol':30} {'n':>6} {'WR%':>6} {'ort%':>8} {'PF':>7} {'toplam%':>9}")
    labels = {"A_tum": "A — tüm işlemler",
              "B_ma10": "B1 — equity > MA10",
              "B_hwm": "B2 — equity yeni zirvede",
              "B_son5": "B3 — son 5 işlem ort. > 0"}
    for key, arm in arms.items():
        s = _stats(arm)
        print(f"{labels[key]:30} {s['n']:>6} {s.get('wr',0):>6} {s.get('ort_%',0):>8} "
              f"{s.get('pf',0):>7} {s.get('toplam_%',0):>9}")


if __name__ == "__main__":
    run()
