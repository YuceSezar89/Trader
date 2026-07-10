"""
HA_Cross sinyalleri üzerinde, RSI_Cross'ta doğrulanan İKİ bulgunun genelleşip
genelleşmediği testi — VPMV sıçraması (rsi_cross_vpmv_jump_bt.py, split-period
sağlam) + mum şekli (rsi_cross_candle_shape_bt.py, kısmen sağlam). HA_Cross
en büyük örneklemli aile (n=2339, 3 Tem sonrası) — RSI_Cross'tan (1739) bile
büyük, sonuçlar daha güvenilir olmalı.

Aynı disiplin: SADECE 3 Tem 19:22:16 sonrası (temiz rejim) + split-period +
OOS ekonomik etki (eşik ilk yarıdan türetilip ikinci yarıya sabit uygulanıyor).
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
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    MIN_HISTORY, INTERVALS, CUTOFF, _fetch_symbol_history,
)
from research.pattern_lab.rsi_cross_candle_shape_bt import _classify  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import _dollar_stats  # pylint: disable=wrong-import-position


def _fetch_ha_cross_signals(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, realized_pnl, opened_at
        FROM signals
        WHERE indicators LIKE '%%HA_Cross%%'
          AND status = 'closed'
          AND interval = %s
          AND realized_pnl IS NOT NULL
          AND closed_at >= %s
    """
    df = pd.read_sql(q, conn, params=(interval, CUTOFF))
    conn.close()
    return df


def _print_tercile(df: pd.DataFrame, col: str, q1: float, q2: float) -> None:
    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    d = df.copy()
    d["tercil"] = d[col].apply(bucket)
    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = d.loc[d["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def _print_category(df: pd.DataFrame, categories: list) -> None:
    print(f"{'kategori':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in categories:
        rets = df.loc[df["kategori"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:20} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_ha_cross_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış HA_Cross sinyali (3 Tem 19:22 sonrası)")

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

                bar = hist.iloc[i]
                kategori = _classify(bar)

                rows.append({
                    "jump": post_v - pre_v,
                    "kategori": kategori,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(rows)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük.")
        return

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    # ── 1. VPMV sıçraması ──
    q1, q2 = df["jump"].quantile([0.333, 0.667])
    print(f"\n══ VPMV sıçraması terciline göre (q1={q1:.1f}, q2={q2:.1f}) ══")
    _print_tercile(df, "jump", q1, q2)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]
    print(f"\n1_ilk_yari (n={len(first)}):")
    _print_tercile(first, "jump", q1, q2)
    print(f"\n2_ikinci_yari (n={len(second)}):")
    _print_tercile(second, "jump", q1, q2)

    # ── 2. Mum şekli ──
    categories = ["gövde-baskın", "üst-fitil-baskın", "alt-fitil-baskın"]
    print(f"\n══ Mum şekli (dağılım: {dict(df['kategori'].value_counts())}) ══")
    _print_category(df, categories)
    print(f"\n1_ilk_yari (n={len(first)}):")
    _print_category(first, categories)
    print(f"\n2_ikinci_yari (n={len(second)}):")
    _print_category(second, categories)

    # ── 3. Birleşik OOS ekonomik etki ──
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"\n══ OOS ekonomik etki ({mid} .. {t_max}, {oos_days:.1f} gün) ══")
    oos_threshold = float(first["jump"].quantile(0.667))
    print(f"in-sample olay: {len(first)} | SABİT VPMV sıçrama eşiği: {oos_threshold:.2f}\n")

    combined_mask = (second["kategori"] != "üst-fitil-baskın") & (second["jump"] >= oos_threshold)
    only_vpmv_mask = second["jump"] >= oos_threshold
    only_shape_mask = second["kategori"] != "üst-fitil-baskın"

    print(f"{'grup':38} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, mask in (
        ("baseline (OOS, tüm HA_Cross)", pd.Series(True, index=second.index)),
        ("sadece VPMV sıçraması filtreli", only_vpmv_mask),
        ("sadece mum-şekli filtreli", only_shape_mask),
        ("BİRLEŞİK (mum-şekli + VPMV)", combined_mask),
    ):
        sub = second[mask]
        rets = sub["realized_pnl"].to_numpy() / 100
        s = _dollar_stats(rets, oos_days)
        if s.get("n", 0) == 0:
            print(f"{name:38} {'0':>6}")
            continue
        print(f"{name:38} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")


if __name__ == "__main__":
    run()
