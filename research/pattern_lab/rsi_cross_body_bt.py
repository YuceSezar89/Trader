"""
Body% testi — RSI_Cross sinyalleri üzerinde, GERÇEK kapanmış sinyal verisiyle
(signals tablosu, gerçek realized_pnl — sentetik backtest değil).

do_break_body_bt.py'de bulunan U-şekilli ilişkinin (düşük VE yüksek body% iyi,
orta kötü) genel bir piyasa fenomeni mi yoksa sadece o setup'a mı özgü olduğunu
test ediyor. body_pct sinyal barının kendisinden (cagg_5m/15m ile symbol+opened_at
join'i) hesaplanıyor — signals tablosunda OHLC yok, sadece open_price var.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

INTERVALS = ["5m", "15m"]


def _fetch(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT s.symbol, s.signal_type, s.realized_pnl,
               c.open, c.high, c.low, c.close
        FROM signals s
        JOIN cagg_{interval} c
          ON c.symbol = s.symbol AND c.bucket = s.opened_at
        WHERE s.indicators LIKE '%%RSI_Cross%%'
          AND s.status = 'closed'
          AND s.interval = %s
          AND s.realized_pnl IS NOT NULL
    """
    df = pd.read_sql(q, conn, params=(interval,))
    conn.close()
    return df


def run():
    frames = []
    for interval in INTERVALS:
        df = _fetch(interval)
        df["interval"] = interval
        frames.append(df)
        print(f"{interval}: {len(df):,} kapanmış RSI_Cross sinyali (OHLC join'li)")
    df = pd.concat(frames, ignore_index=True)
    print(f"\ntoplam: {len(df):,}\n")

    rng = df["high"] - df["low"]
    df["body_pct"] = np.where(rng > 0, (df["close"] - df["open"]).abs() / rng * 100, 0.0)

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    q1, q2 = df["body_pct"].quantile([0.333, 0.667])
    print(f"\n── RSI_Cross (5m+15m), sinyal barının body% terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")

    def bucket(bp):
        return "düşük" if bp < q1 else ("orta" if bp < q2 else "yüksek")

    df["tercil"] = df["body_pct"].apply(bucket)
    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = df.loc[df["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print("\n── Long/Short ayrı ──")
    for sig_type in ("Long", "Short"):
        sub = df[df["signal_type"] == sig_type]
        print(f"\n{sig_type}:")
        print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for name in ("düşük", "orta", "yüksek"):
            rets = sub.loc[sub["tercil"] == name, "realized_pnl"].to_numpy() / 100
            s = _stats(rets)
            print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


if __name__ == "__main__":
    run()
