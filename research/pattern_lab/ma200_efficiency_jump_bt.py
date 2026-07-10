"""
Verimlilik (ERSI/devisso_score) sıçraması testi — MA200_Cross sinyalleri
üzerinde. Kullanıcı fikri: CumΔ'nın biriktirdiği şey (ΔRSI) aslında ERSI'nin
(ΔPrice%/ΔRSI, project_devisso_ersi.md) PAYDASI — devisso_score'un MUTLAK
değeri PnL ile ilişkisiz çıkmıştı (|r|<0.05) ama VPMV sıçraması testinde
gördüğümüz gibi (mutlak skor değil YÖN/DEĞİŞİM önemli), verimliliğin sinyal
ETRAFINDAKİ DEĞİŞİMİ ayrı bir açı olabilir.

RSI_Cross yerine MA200_Cross seçildi (kullanıcı önerisi) — RSI_Cross bir DÖNÜŞ
sinyali, MA200_Cross/Supertrend bir TREND TEYİT sinyali; "bu trend az RSI
harcayarak mı ilerliyor (verimli/sağlıklı) yoksa çok RSI harcayıp az mı yol
alıyor (zorlanan/sahte)" sorusu trend sinyallerinde daha doğal. Supertrend
veri azlığından (n=136) elendi, MA200_Cross (n=330) ile devam edildi.

Verimlilik = signal_processor.py::_compute_devisso_score ile AYNI formül
(ΔPrice%/ΔRSI(14) → EMA(7) → rolling-100 percentile rank, 0-100) ama TÜM
seri için hesaplanıyor (sadece son bar değil) — sinyal barından 1 bar önce
ve 1 bar sonraki değer arasındaki sıçrama (jump) test ediliyor.

Metodoloji: SADECE 3 Tem 19:22:16 sonrası (temiz rejim) + split-period.
Örneklem küçük (330) — dikkatli yorumlanmalı.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    MIN_HISTORY, INTERVALS, CUTOFF, _fetch_symbol_history,
)


def _fetch_ma200_signals(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, realized_pnl, opened_at
        FROM signals
        WHERE indicators LIKE '%%MA200_Cross%%'
          AND status = 'closed'
          AND interval = %s
          AND realized_pnl IS NOT NULL
          AND closed_at >= %s
    """
    df = pd.read_sql(q, conn, params=(interval, CUTOFF))
    conn.close()
    return df


def _efficiency_rank_series(df: pd.DataFrame) -> pd.Series:
    """signal_processor.py::_compute_devisso_score ile AYNI formül, TÜM seri için."""
    close = df["close"].astype(float)
    rsi = calculate_rsi(df, period=14)
    price_pct = close.pct_change() * 100.0
    raw = price_pct / rsi.diff().replace(0.0, np.nan)
    smoothed = raw.ewm(span=7, adjust=False).mean()

    def _rank(x):
        return (x <= x[-1]).mean() * 100.0

    return smoothed.rolling(100, min_periods=20).apply(_rank, raw=True)


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
        sigs = _fetch_ma200_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış MA200_Cross sinyali (3 Tem 19:22 sonrası)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if len(hist) < MIN_HISTORY:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            eff_rank = _efficiency_rank_series(hist)

            for _, row in sub.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i - 1 < 0 or i + 1 >= len(hist):
                    continue
                pre_v = eff_rank.iloc[i - 1]
                post_v = eff_rank.iloc[i + 1]
                if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                    continue
                all_pairs.append({
                    "jump": post_v - pre_v,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(all_pairs)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 60:
        print("Örneklem çok küçük, güvenilir analiz yapılamaz.")
        return

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    q1, q2 = df["jump"].quantile([0.333, 0.667])
    print(f"\n── Verimlilik sıçraması (post[+1] - pre[-1]) terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")
    _print_tercile_table(df, q1, q2)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 20:
        _print_tercile_table(first, q1, q2)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 20:
        _print_tercile_table(second, q1, q2)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
