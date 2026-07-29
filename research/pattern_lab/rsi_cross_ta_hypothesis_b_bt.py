"""
Hipotez B: "kovalama" riski — üst-TF'nin percentile'ı ZATEN çok yüksekken
(hareket olmuş bitmiş görünüyor) VE hâlâ tırmanıyorsa, bu RSI_Cross sinyali
geç kalmış bir kovalama mı (daha zayıf), yoksa fark etmiyor mu? (23-24 Tem
2026, kullanıcı yokken tam yetkiyle test edildi.)

Referans: süleyman özçelik.html'in taObserveAt() mantığı — "high & rising"
(LONG KOVALAMA, bucket=RİSK) etiketi. Bizim slope tanımımız aynı (net[i]-net[i-2]).

percentile>=55 (1h+4h) popülasyonu (Hipotez A'nın kazananı) İÇİNDE:
  kovalama = (pct_1h>=80 AND slope_1h>0) OR (pct_4h>=80 AND slope_4h>0)
  saglikli = geri kalanı (henüz aşırı değil, veya aşırı ama zaten dönüyor)

Kullanım: python -m research.pattern_lab.rsi_cross_ta_hypothesis_b_bt
(önce rsi_cross_ta_percentile_bt.py çalıştırılmış olmalı — cache dosyasını kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_percentile_bt import _CACHE_PATH

_EXTREME_PCT = 80
_BEST_TH = 55  # Hipotez A'nın seçtiği eşik


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "ort_%": None, "medyan_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3), "medyan_%": round(float(np.median(rets)), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    df = pd.read_parquet(_CACHE_PATH)
    df = _add_all_up(df)
    df = df[df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)

    base = df[(df["pct_1h"] >= _BEST_TH) & (df["pct_4h"] >= _BEST_TH)].reset_index(drop=True)
    print(f"[percentile>={_BEST_TH} (1h+4h) popülasyonu] n={len(base)}\n")

    kovalama_mask = ((base["pct_1h"] >= _EXTREME_PCT) & (base["slope_1h"] > 0)) | \
                    ((base["pct_4h"] >= _EXTREME_PCT) & (base["slope_4h"] > 0))
    kovalama = base[kovalama_mask]
    saglikli = base[~kovalama_mask]

    print("=" * 78)
    print(f"KOVALAMA (pct>={_EXTREME_PCT} VE hâlâ tırmanıyor) vs SAĞLIKLI (geri kalan)")
    print("=" * 78)
    print(f"  kovalama : {_stats(kovalama['fwd_ret'].to_numpy())}")
    print(f"  sağlıklı : {_stats(saglikli['fwd_ret'].to_numpy())}")

    base = base.copy()
    base["_g"] = kovalama_mask
    _deep_validate("KOVALAMA (pct>=80 & rising)", kovalama, saglikli, base)

    days_span = (base["opened_at"].max() - base["opened_at"].min()).total_seconds() / 86400
    if len(kovalama) >= 10:
        summarize("KOVALAMA grubu — fwd_ret% serisi", kovalama["fwd_ret"].to_numpy(), kovalama["opened_at"], days_span)
    if len(saglikli) >= 10:
        summarize("SAĞLIKLI grubu — fwd_ret% serisi", saglikli["fwd_ret"].to_numpy(), saglikli["opened_at"], days_span)

    print("\n" + "=" * 78)
    print("KARŞI TEST: dönüş anı mı önemli? (pct>=80 içinde turnedDown vs hâlâ rising)")
    print("=" * 78)
    extreme = base[(base["pct_1h"] >= _EXTREME_PCT) | (base["pct_4h"] >= _EXTREME_PCT)]
    extreme_rising = extreme[((extreme["pct_1h"] >= _EXTREME_PCT) & (extreme["slope_1h"] > 0)) |
                              ((extreme["pct_4h"] >= _EXTREME_PCT) & (extreme["slope_4h"] > 0))]
    extreme_falling = extreme[~extreme.index.isin(extreme_rising.index)]
    print(f"  extreme+rising (kovalama)   : {_stats(extreme_rising['fwd_ret'].to_numpy())}")
    print(f"  extreme+falling (dönüyor)   : {_stats(extreme_falling['fwd_ret'].to_numpy())}")


if __name__ == "__main__":
    main()
