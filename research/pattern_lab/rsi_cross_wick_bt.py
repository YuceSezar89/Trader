"""
A%/K%/H%/L% testi (C99 BTHN Pine script'i, kullanıcının "bu kolonlar ne anlatıyor"
sorusundan). Pine'ın kendi tanımı, dpx'in enterpole edilmiş "kesişim fiyatı"na
göre — bizim RSI_Cross sinyal tanımımızda böyle bir ara nokta yok (sinyal bar
kapanışında tetikleniyor), o yüzden referans nokta olarak barın KENDİ AÇILIŞI
kullanılıyor (en yakın karşılık gelen basitleştirme).

En özgün/yeni açı: ADVERSE excursion — sinyal mumunda pozisyon YÖNÜNE TERS ne
kadar sallanma (geri tepme/fitil) olduğu. body_pct (|close-open|/range) bunu
YÖNSÜZ ölçüyordu (hangi tarafta olduğunu ayırt etmiyordu); burada spesifik
olarak "pozisyon aleyhine" fitili ölçüyoruz:
  Long:  adverse_pct = (open - low) / open * 100
  Short: adverse_pct = (high - open) / open * 100
Hipotez: düşük adverse (temiz, çekişmesiz mum) → iyi; yüksek adverse (mum
içinde ciddi karşı hareket olmuş, "zar zor" tetiklenmiş) → kötü.

Metodoloji (body%/VPMV-jump derslerinden): BAŞTAN split-period + SADECE 3 Tem
19:22:16 sonrası (commit e81aa34, temiz ters-sinyal/timeout rejimi) —
rsi_cross_body_split_check.py'nin _fetch'i (cutoff + opened_at) yeniden kullanıldı.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_body_split_check import INTERVALS, CUTOFF, _fetch  # pylint: disable=wrong-import-position,unused-import


def _print_tercile_table(df: pd.DataFrame, q1: float, q2: float) -> None:
    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    df = df.copy()
    df["tercil"] = df["adverse_pct"].apply(bucket)
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
        print(f"{interval}: {len(d):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası)")
    df = pd.concat(frames, ignore_index=True)
    print(f"\ntoplam: {len(df):,}\n")

    if len(df) < 100:
        print("Örneklem çok küçük.")
        return

    is_long = df["signal_type"] == "Long"
    df["adverse_pct"] = np.where(
        is_long,
        (df["open"] - df["low"]) / df["open"] * 100,
        (df["high"] - df["open"]) / df["open"] * 100,
    )

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    q1, q2 = df["adverse_pct"].quantile([0.333, 0.667])
    print(f"\n── Aleyhe fitil (adverse_pct) terciline göre ── (q1={q1:.2f}, q2={q2:.2f})")
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
