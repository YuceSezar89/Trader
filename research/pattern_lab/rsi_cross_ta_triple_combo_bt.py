"""
Üçlü kombinasyon: all_up + TA "kovalama" (Hipotez B, pct>=80 & rising, 1h
veya 4h) + HA-hizalanması (1h VE 4h). Kullanıcı isteği (24 Tem 2026).

Girdi: rsi_cross_ta_ha_overlap_bt.py'nin cache'lediği (_cache_rsi_cross_ta_ha.parquet)
all_up+pct_1h/pct_4h/slope_1h/slope_4h+ha_bull_1h/ha_bull_4h verisi.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_combo_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH
from research.pattern_lab.rsi_cross_ta_percentile_bt import _decompose

_BASE_TH = 55
_EXTREME_PCT = 80


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "ort_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    df = pd.read_parquet(_HA_CACHE_PATH)
    df = df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"]).reset_index(drop=True)
    df["ha_aligned"] = (df["ha_bull_1h"] > 0.5) & (df["ha_bull_4h"] > 0.5)
    print(f"[all_up=True + tüm bileşenler hesaplanabilir popülasyon] n={len(df)}\n")

    ta_base = (df["pct_1h"] >= _BASE_TH) & (df["pct_4h"] >= _BASE_TH)
    kovalama = ((df["pct_1h"] >= _EXTREME_PCT) & (df["slope_1h"] > 0)) | \
               ((df["pct_4h"] >= _EXTREME_PCT) & (df["slope_4h"] > 0))
    ha = df["ha_aligned"]

    print("=" * 78)
    print("BİLEŞENLERİN TEK TEK VE İKİLİ SONUÇLARI (karşılaştırma için)")
    print("=" * 78)
    print(f"  TA-base (pct>={_BASE_TH}, 1h+4h)          : {_stats(df[ta_base]['fwd_ret'].to_numpy())}")
    print(f"  TA-base + kovalama                        : {_stats(df[ta_base & kovalama]['fwd_ret'].to_numpy())}")
    print(f"  TA-base + HA                              : {_stats(df[ta_base & ha]['fwd_ret'].to_numpy())}")

    print("\n" + "=" * 78)
    print("ÜÇLÜ KOMBİNASYON: TA-base + kovalama + HA-hizalı")
    print("=" * 78)
    triple_mask = ta_base & kovalama & ha
    triple = df[triple_mask]
    rest = df[~triple_mask]
    print(f"  üçlü kombinasyon: {_stats(triple['fwd_ret'].to_numpy())}")
    print(f"  geri kalan (tüm popülasyon içinde): {_stats(rest['fwd_ret'].to_numpy())}")

    df["_g"] = triple_mask
    _deep_validate("TA-base + kovalama + HA (üçlü)", triple, rest, df)

    print(f"\n  PF şüphesi (medyan): {_decompose(triple['fwd_ret'].to_numpy())}")

    if len(triple) >= 10:
        days_span = (triple["opened_at"].max() - triple["opened_at"].min()).total_seconds() / 86400
        summarize("Üçlü kombinasyon — fwd_ret% serisi", triple["fwd_ret"].to_numpy(), triple["opened_at"], days_span)

    print("\n" + "=" * 78)
    print("ÖZET TABLO — sırayla ekleme etkisi")
    print("=" * 78)
    steps = [
        ("all_up (filtresiz)", np.ones(len(df), dtype=bool)),
        ("+ TA-base (pct>=55)", ta_base),
        ("+ kovalama (pct>=80&rising)", ta_base & kovalama),
        ("+ HA-hizalı (üçlü)", ta_base & kovalama & ha),
    ]
    for label, mask in steps:
        s = _stats(df[mask]["fwd_ret"].to_numpy())
        print(f"  {label:32}: n={s['n']:>5}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")


if __name__ == "__main__":
    main()
