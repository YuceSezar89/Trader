"""
"Ultra hizalama" testi — HA_Cross 5m sinyalleri, üst TF'lerin (15m/1h/4h) HA
yönüyle ne kadar örtüşüyor. PythonScanner (dış proje, /Users/yusuf/Desktop/
adsız klasör 2/PythonScanner) 4h/6h/8h/12h'de HA crossover'ı ayrı ayrı
hesaplayıp bull_count>=3 aldığında "ULTRA" sinyali veriyordu — kendi TF
setimizde (5m/15m/1h/4h, DB'de zaten mevcut) aynı fikri, YENİ sinyal
üretmeden, var olan HA_Cross 5m sinyallerini üst TF onayına göre tercillere
ayırarak test ediyoruz.

Yöntem: sinyal anında (opened_at), 15m/1h/4h'de o ana kadar KAPANMIŞ en son
barın HA yönü (calculate_ha, tam geçmiş üzerinde hesaplanır — look-ahead
yok, sadece opened_at'ten önceki barlar kullanılıyor) sinyalle aynı yönde mi
diye sayılıyor (0-3 üst TF onayı). Aynı disiplin: SADECE 3 Tem 19:22:16
sonrası (temiz rejim) + split-period + OOS ekonomik etki.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.ha_cross_combined_test import (
    _fetch_ha_cross_signals,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.mtf_helpers import (  # pylint: disable=wrong-import-position
    _confirm_count,
    _fetch_dir_data,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

CONFIRM_TFS = ["15m", "1h", "4h"]
POSITION_USD = 100.0
FEE_RATE = 0.0005
ROUND_TRIP_FEE = POSITION_USD * FEE_RATE * 2


def _dollar_stats(rets: np.ndarray, days_span: float) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    pnl = rets * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    per_month = total / days_span * 30 if days_span > 0 else 0.0
    return {
        "n": len(rets),
        "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(per_month, 1),
    }


def _print_by_count(df: pd.DataFrame) -> None:
    print(f"{'üst TF onayı':14} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for count in (0, 1, 2, 3):
        rets = df.loc[df["confirm_count"] == count, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        label = f"{count}/3" + (" (ULTRA)" if count == 3 else "")
        print(
            f"{label:14} {s.get('n',0):>7} {s.get('wr',0):>6} "
            f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
        )


def run():
    rows = []
    sigs = _fetch_ha_cross_signals("5m")
    print(f"5m: {len(sigs):,} kapanmış HA_Cross sinyali (3 Tem 19:22 sonrası)\n")

    for symbol, sub in sigs.groupby("symbol"):
        dir_data = _fetch_dir_data(symbol, CONFIRM_TFS)
        if dir_data is None:
            continue

        for _, row in sub.iterrows():
            confirm_count = _confirm_count(
                dir_data, CONFIRM_TFS, row["opened_at"], want_bullish=(row["signal_type"] == "Long")
            )
            if confirm_count is None:
                continue
            rows.append(
                {
                    "confirm_count": confirm_count,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                }
            )

    df = pd.DataFrame(rows)
    print(f"toplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük, güvenilir analiz yapılamaz.")
        return

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(
        f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
        f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}\n"
    )

    print("── Üst TF (15m/1h/4h) onay sayısına göre ──")
    _print_by_count(df)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 30:
        _print_by_count(first)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 30:
        _print_by_count(second)
    else:
        print("örneklem çok küçük")

    # OOS ekonomik etki: eşik gerektirmiyor (ayrık say), ilk yarı sadece
    # örneklem büyüklüğü/yön teyidi için, asıl $ etkisi ikinci (OOS) yarıda.
    days_span = (t_max - mid).total_seconds() / 86400
    print(f"\n── OOS ($) ekonomik etki — ikinci yarı, ${POSITION_USD:.0f} pozisyon ──")
    print(f"{'grup':20} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, sub in (
        ("baseline (tümü)", second),
        ("ULTRA (3/3)", second[second["confirm_count"] == 3]),
        ("0-2/3", second[second["confirm_count"] < 3]),
    ):
        stat = _dollar_stats(sub["realized_pnl"].to_numpy() / 100, days_span)
        if stat.get("n", 0) == 0:
            print(f"{name:20} {'0':>6}")
            continue
        print(
            f"{name:20} {stat['n']:>6} {stat['wr']:>6} {stat['avg_usd']:>12} "
            f"{stat['total_usd']:>10} {stat['usd_per_month']:>10}"
        )


if __name__ == "__main__":
    run()
