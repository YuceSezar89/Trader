"""
CumΔ testi (C99 BTHN'deki cumDelta/normCumDelta, satır 426-441) — kullanıcının
"cum delta bizi nereye götürecek" sorusu. Bu, project_turtle_traders.md'de
"flow_since_prev" olarak beklemede duran fikrin somut bir uygulaması.

Pine mantığı (birebir): dpx = RSI'nin bar-bar değişimi. Ardışık AYNI YÖNLÜ
sinyaller boyunca dpx'ler BİRİKTİRİLİYOR (cumDelta += dpx); yön değişince
(ters sinyal gelince) sıfırlanıp yeniden başlıyor (cumDelta = dpx). Yani
"son ters dönüşten beri bu yönde ne kadar RSI-momentumu birikti" sorusuna
cevap.

Hipotez: cumDelta ne kadar büyükse (aynı yönde tekrarlayan, güçlenen RSI
momentumu), sinyal o kadar güvenilir mi? Yoksa tam tersi mi (tükenme)?
Ayrıca cumDelta'nın TEK BAŞINA dpx'ten (birikimsiz, sadece son sinyalin
kendi RSI değişimi) daha mı iyi ayırt ettiği test ediliyor.

Metodoloji: BAŞTAN split-period + SADECE 3 Tem 19:22:16 sonrası (commit
e81aa34, temiz ters-sinyal/timeout rejimi). cumDelta state'i SEMBOL BAZINDA
(Pine'ın tek-sembollü indikatör mantığıyla aynı) ayrı ayrı takip ediliyor.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    MIN_HISTORY, INTERVALS, _fetch_signals, _fetch_symbol_history,
)


def _print_tercile_table(df: pd.DataFrame, col: str, q1: float, q2: float) -> None:
    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    df = df.copy()
    df["tercil"] = df[col].apply(bucket)
    print(f"{'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = df.loc[df["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def run():
    rows = []

    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if len(hist) < MIN_HISTORY:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            rsi = calculate_rsi(hist, period=14)
            dpx = rsi.diff()

            # Bu sembolün TÜM sinyallerini kronolojik sırayla işle (Pine'ın
            # tek-sembollü state machine'iyle birebir aynı)
            sub_sorted = sub.sort_values("opened_at")
            cum_delta = 0.0
            prev_type = None
            for _, row in sub_sorted.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i >= len(dpx):
                    continue
                dpx_val = dpx.iloc[i]
                if not np.isfinite(dpx_val):
                    continue

                if prev_type is None or prev_type != row["signal_type"]:
                    cum_delta = dpx_val
                else:
                    cum_delta = cum_delta + dpx_val
                prev_type = row["signal_type"]

                rows.append({
                    "dpx": abs(dpx_val),
                    "cum_delta": abs(cum_delta),
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

    q1, q2 = df["dpx"].quantile([0.333, 0.667])
    print(f"\n── SADECE dpx (birikimsiz, tek sinyalin kendi RSI değişimi) terciline göre ── (q1={q1:.2f}, q2={q2:.2f})")
    _print_tercile_table(df, "dpx", q1, q2)

    q1c, q2c = df["cum_delta"].quantile([0.333, 0.667])
    print(f"\n── cumDelta (ardışık aynı yönlü sinyallerde biriken) terciline göre ── (q1={q1c:.2f}, q2={q2c:.2f})")
    _print_tercile_table(df, "cum_delta", q1c, q2c)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}), cumDelta ══")
    if len(first) >= 30:
        _print_tercile_table(first, "cum_delta", q1c, q2c)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}), cumDelta ══")
    if len(second) >= 30:
        _print_tercile_table(second, "cum_delta", q1c, q2c)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
