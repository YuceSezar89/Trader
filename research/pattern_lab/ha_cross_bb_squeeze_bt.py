"""
Bollinger Band daralması (squeeze) testi — HA_Cross sinyalleri üzerinde.
Kullanıcı fikri: HA_Cross (trend teyidi) sinyali, Bollinger bantları
DARALMIŞKEN (düşük volatilite, "patlama öncesi sessizlik") geldiyse daha
güvenilir olabilir mi?

Yöntem: `indicators/core.py::calculate_bollinger_bands` (period=20, num_std=2)
ile bant genişliği = (üst-alt)/orta*100 hesaplanıp, TÜM seri için rolling-100
percentile rank'e çevriliyor (VPMV/verimlilik sıçraması testlerindeki AYNI
yöntem) — sinyal barındaki değer DÜŞÜKSE bantlar o an tarihsel olarak dar
(squeeze), YÜKSEKSE geniş (zaten volatil) demek.

Hipotez: squeeze anında gelen sinyal (düşük tercil) → daha büyük/güvenilir
hareket. Metodoloji: SADECE 3 Tem 19:22:16 sonrası + split-period BAŞTAN.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_bollinger_bands  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import MIN_HISTORY, INTERVALS, _fetch_symbol_history  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_combined_test import _fetch_ha_cross_signals  # pylint: disable=wrong-import-position


def _bb_width_rank_series(df: pd.DataFrame) -> pd.Series:
    sma, upper, lower = calculate_bollinger_bands(df, period=20, num_std=2)
    width = (upper - lower) / sma * 100.0

    def _rank(x):
        return (x <= x[-1]).mean() * 100.0

    return width.rolling(100, min_periods=30).apply(_rank, raw=True)


def _print_tercile(df: pd.DataFrame, q1: float, q2: float) -> None:
    def bucket(v):
        return "düşük (daralmış)" if v < q1 else ("orta" if v < q2 else "yüksek (geniş)")

    d = df.copy()
    d["tercil"] = d["bb_rank"].apply(bucket)
    print(f"{'tercil':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük (daralmış)", "orta", "yüksek (geniş)"):
        rets = d.loc[d["tercil"] == name, "realized_pnl"].to_numpy() / 100
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

            bb_rank = _bb_width_rank_series(hist)

            for _, row in sub.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i >= len(bb_rank):
                    continue
                val = bb_rank.iloc[i]
                if not np.isfinite(val):
                    continue
                rows.append({
                    "bb_rank": val,
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

    q1, q2 = df["bb_rank"].quantile([0.333, 0.667])
    print(f"\n── BB genişliği (rolling-100 percentile rank) terciline göre ── (q1={q1:.1f}, q2={q2:.1f})")
    _print_tercile(df, q1, q2)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 30:
        _print_tercile(first, q1, q2)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 30:
        _print_tercile(second, q1, q2)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
