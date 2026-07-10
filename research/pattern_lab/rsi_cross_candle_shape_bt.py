"""
Mum ŞEKLİ testi (kullanıcının uyarısı: A%/K%/H%/L%/body% hepsi TEK bir mumun
parçaları, ayrı ayrı test edilmemeli — bütün olarak bakılmalı).

Bir mum tam olarak 3 parçaya ayrılır (toplamda menzilin %100'ü):
  gövde%     = |close-open| / (high-low) * 100
  üst_fitil% = (high - max(open,close)) / (high-low) * 100
  alt_fitil% = (min(open,close) - low) / (high-low) * 100

Her sinyal mumu, bu üç parçadan HANGİSİ EN BÜYÜKSE o kategoriye atanıyor
(gövde-baskın / üst-fitil-baskın / alt-fitil-baskın) — parçalamadan, mumu
BÜTÜN olarak sınıflandırıp realized_pnl ile karşılaştırıyoruz. Long/Short
ayrı (üst/alt fitilin anlamı yöne göre değişir: Long'da alt fitil aleyhte,
üst fitil lehte fazladan hareket; Short'ta tersi).

Metodoloji: BAŞTAN split-period + SADECE 3 Tem 19:22:16 sonrası (commit
e81aa34, temiz ters-sinyal/timeout rejimi).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_body_split_check import INTERVALS, _fetch  # pylint: disable=wrong-import-position


def _classify(row) -> str:
    rng = row["high"] - row["low"]
    if rng <= 0:
        return "belirsiz"
    upper = max(row["open"], row["close"])
    lower = min(row["open"], row["close"])
    body = abs(row["close"] - row["open"]) / rng * 100
    upper_wick = (row["high"] - upper) / rng * 100
    lower_wick = (lower - row["low"]) / rng * 100

    parts = {"gövde-baskın": body, "üst-fitil-baskın": upper_wick, "alt-fitil-baskın": lower_wick}
    return max(parts, key=parts.get)


def _print_table(df: pd.DataFrame, categories: list[str]) -> None:
    print(f"{'kategori':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in categories:
        rets = df.loc[df["kategori"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:20} {s.get('n',0):>7} {s.get('wr',0):>6} "
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

    df["kategori"] = df.apply(_classify, axis=1)
    categories = ["gövde-baskın", "üst-fitil-baskın", "alt-fitil-baskın"]

    print(f"kategori dağılımı:\n{df['kategori'].value_counts()}\n")

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print("\n── TÜM sinyaller, mum şekline göre ──")
    _print_table(df, categories)

    print("\n── Long ──")
    _print_table(df[df["signal_type"] == "Long"], categories)

    print("\n── Short ──")
    _print_table(df[df["signal_type"] == "Short"], categories)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 30:
        _print_table(first, categories)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 30:
        _print_table(second, categories)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
