"""
DÜŞEN BIÇAK dışlama testi — Madde 2'nin devamı (24 Tem 2026, kullanıcı
isteği). rsi_cross_ta_regime_labels_bt.py'de DÜŞEN BIÇAK (1h VE 4h'de)
istatistiksel olarak anlamlı şekilde kötü çıktı (placebo %5.0/%1.7,
split-period istikrarlı negatif) — LONG KOVALAMA ise ZATEN en iyi filtremiz
(Hipotez B). Bu ikisi karşılıklı dışlayıcı (kovalama=yüksek percentile,
düşen bıçak=düşük percentile), yani "kovalama'yı ara" ile "düşen bıçağı
dışla" FARKLI popülasyonlar üretir — soru: dar-ama-en-iyi (kovalama) yerine
geniş-ama-güvenli (sadece düşen bıçağı dışla) daha mı iyi bir denge sunuyor?

Popülasyonlar (all_up=True içinde, 1H VEYA 4H etiketine göre):
  A. all_up (filtresiz baseline)
  B. all_up - düşen_bıçak (geniş, sadece kötüyü dışla)
  C. all_up + kovalama (dar, en iyiyi ara — mevcut filtremiz)
  D. all_up - kovalama - düşen_bıçak (ikisi de değil — "orta" popülasyon)

Kullanım: python -m research.pattern_lab.rsi_cross_ta_dusen_bicak_exclusion_bt
(önce rsi_cross_ta_regime_labels_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_regime_labels_bt import _CACHE_PATH


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
    df = df[df["all_up"]].reset_index(drop=True)
    print(f"[all_up=True popülasyonu] n={len(df)}\n")

    dusen_bicak = (df["label_1h"] == "DÜŞEN BIÇAK") | (df["label_4h"] == "DÜŞEN BIÇAK")
    kovalama = (df["label_1h"] == "LONG KOVALAMA") | (df["label_4h"] == "LONG KOVALAMA")

    pops = {
        "A. all_up (filtresiz)": np.ones(len(df), dtype=bool),
        "B. all_up - düşen_bıçak (geniş/güvenli)": ~dusen_bicak,
        "C. all_up + kovalama (dar/en iyi)": kovalama,
        "D. all_up - kovalama - düşen_bıçak (orta)": (~kovalama) & (~dusen_bicak),
    }

    print("=" * 78)
    print("ÖZET — 4 popülasyonun karşılaştırması")
    print("=" * 78)
    for label, mask in pops.items():
        s = _stats(df[mask]["fwd_ret"].to_numpy())
        print(f"  {label:42}: n={s['n']:>5}  WR%={s.get('wr','-'):>6}  "
              f"medyan%={s.get('medyan_%','-'):>7}  PF={s.get('pf','-')}")

    print("\n" + "=" * 78)
    print("DERİN DOĞRULAMA — B ve D (henüz test edilmemiş yeni popülasyonlar)")
    print("=" * 78)
    for label, mask in [("B. all_up - düşen_bıçak", ~dusen_bicak), ("D. all_up - kovalama - düşen_bıçak", (~kovalama) & (~dusen_bicak))]:
        group, rest = df[mask], df[~mask]
        df["_g"] = mask
        _deep_validate(label, group, rest, df)
        days_span = (group["opened_at"].max() - group["opened_at"].min()).total_seconds() / 86400
        summarize(f"{label} — fwd_ret% serisi", group["fwd_ret"].to_numpy(), group["opened_at"], days_span)


if __name__ == "__main__":
    main()
