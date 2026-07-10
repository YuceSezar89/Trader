"""
Sağlamlık: rsi_cross_body_bt.py bulgusu (sinyal barının body% terciline göre
monoton PF artışı, hem Long hem Short'ta simetrik) — İKİ ek şartla test ediliyor:

1. SADECE 3 Temmuz 2026 19:22:16 SONRASI kapanan sinyaller (commit e81aa34 —
   signals tablosunun fiyat-bazlı SL/TP kapanışı kaldırılıp ters-sinyal/timeout'a
   geçtiği an). Eski rejimde kapanan sinyaller karıştırılmıyor — bkz.
   [[project_signal_lifecycle]].
2. Dönem ikiye bölünüp (opened_at zaman damgasına göre) HER İKİ yarıda
   BAĞIMSIZ tekrarlanıyor mu diye bakılıyor.
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
CUTOFF = "2026-07-03 19:22:16"  # commit e81aa34 — signals SL/TP fiyat-bazlı kapanış kaldırıldı


def _fetch(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT s.symbol, s.signal_type, s.realized_pnl, s.opened_at, s.closed_at,
               c.open, c.high, c.low, c.close
        FROM signals s
        JOIN cagg_{interval} c
          ON c.symbol = s.symbol AND c.bucket = s.opened_at
        WHERE s.indicators LIKE '%%RSI_Cross%%'
          AND s.status = 'closed'
          AND s.interval = %s
          AND s.realized_pnl IS NOT NULL
          AND s.closed_at >= %s
    """
    df = pd.read_sql(q, conn, params=(interval, CUTOFF))
    conn.close()
    return df


def _print_tercile_table(df: pd.DataFrame, q1: float, q2: float) -> None:
    def bucket(bp):
        return "düşük" if bp < q1 else ("orta" if bp < q2 else "yüksek")

    df = df.copy()
    df["tercil"] = df["body_pct"].apply(bucket)
    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = df.loc[df["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def run():
    frames = []
    for interval in INTERVALS:
        d = _fetch(interval)
        d["interval"] = interval
        frames.append(d)
        print(f"{interval}: {len(d):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası, OHLC join'li)")
    df = pd.concat(frames, ignore_index=True)
    print(f"\ntoplam: {len(df):,}\n")

    if len(df) < 100:
        print("Örneklem çok küçük, güvenilir tercil analizi yapılamaz.")
        return

    rng = df["high"] - df["low"]
    df["body_pct"] = np.where(rng > 0, (df["close"] - df["open"]).abs() / rng * 100, 0.0)

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    q1, q2 = df["body_pct"].quantile([0.333, 0.667])
    print(f"\n── SADECE 3 Tem sonrası, body% terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")
    _print_tercile_table(df, q1, q2)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 30:
        _print_tercile_table(first, q1, q2)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 30:
        _print_tercile_table(second, q1, q2)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
