"""
Tek-bar VPMV sıçraması testi (C99 BTHN Pine script'indeki MVPVB→MVPV→MVPVA
üçlüsünden esinlenildi — kullanıcının fikri, bkz. project_pattern_lab.md v2-7
sonrası). Bizim 5-bar/4-bar ortalamalı vpmv_pre_avg/vpmv_post_avg'dan FARKLI:
sinyal barından TAM 1 bar önce ve TAM 1 bar sonraki VPMV skoru (utils/vpmv.py
::compute_series ile, hocanın ayrı sigmoid formülü yerine bizim zaten
test edilmiş/doğru formülümüz) — aradaki sıçrama (jump = post - pre)
realized_pnl ile ilişkili mi?

Metodoloji dersi (rsi_cross_body_bt.py → split_check'te çürümesinden çıkarıldı,
project_pattern_lab.md v2-7): BAŞTAN split-period + SADECE 3 Tem 19:22:16
sonrası (commit e81aa34, signals tablosunun temiz ters-sinyal/timeout rejimi)
uygulanıyor — büyük ilk-bakış n'sine güvenip sonradan pişman olmayalım.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

INTERVALS = ["5m", "15m"]
CUTOFF = "2026-07-03 19:22:16"  # commit e81aa34
MIN_HISTORY = 60  # compute_series'in rolling normalizasyonları için makul minimum


def _fetch_signals(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, realized_pnl, opened_at
        FROM signals
        WHERE indicators LIKE '%%RSI_Cross%%'
          AND status = 'closed'
          AND interval = %s
          AND realized_pnl IS NOT NULL
          AND closed_at >= %s
    """
    df = pd.read_sql(q, conn, params=(interval, CUTOFF))
    conn.close()
    return df


def _fetch_symbol_history(symbol: str, interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"SELECT bucket AS ts, open, high, low, close, volume FROM cagg_{interval} WHERE symbol = %s ORDER BY bucket"
    df = pd.read_sql(q, conn, params=(symbol,))
    conn.close()
    return df


def _print_tercile_table(pairs: pd.DataFrame, q1: float, q2: float) -> None:
    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    pairs = pairs.copy()
    pairs["tercil"] = pairs["jump"].apply(bucket)
    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = pairs.loc[pairs["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def run():
    all_pairs = []

    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if len(hist) < MIN_HISTORY:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            series_long = compute_series(hist, "Long")
            series_short = compute_series(hist, "Short")

            for _, row in sub.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i - 1 < 0 or i + 1 >= len(hist):
                    continue
                series = series_long if row["signal_type"] == "Long" else series_short
                pre_v = series.iloc[i - 1]
                post_v = series.iloc[i + 1]
                if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                    continue
                all_pairs.append({
                    "jump": post_v - pre_v,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(all_pairs)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük, güvenilir analiz yapılamaz.")
        return

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    q1, q2 = df["jump"].quantile([0.333, 0.667])
    print(f"\n── VPMV sıçraması (post[+1] - pre[-1]) terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")
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
