"""
Kâr-yoğunlaşma (concentration) teşhis araçları — 22 Tem 2026.

Bugünkü ders: bir backtest'in toplam $/ay rakamı iyi görünse de, bu kârın
birkaç uç işlemden mi geldiği yoksa geniş/istikrarlı bir kenar mı olduğu
AYRI bir soru — placebo/split-period bunu yakalamıyor (ikisi de "gerçek mi
tesadüf mü" sorusuna cevap veriyor, "kaç işleme yayılmış" sorusuna değil).

Bu modül, HERHANGİ bir backtest'in pnl serisine uygulanabilecek, tekrar
kullanılabilir 3 teşhis sunar:
  1. top_n_contribution   — en iyi N işlemin toplam kâra katkı yüzdesi
  2. trimmed_total        — en iyi N işlem çıkarılınca geriye ne kalıyor
  3. bootstrap_monthly    — işlemleri rastgele "aylara" dağıtıp simüle
     ederek, gerçekte kaç ayın net pozitif/negatif çıktığını tahmin eder
     (tek bir toplam $/ay sayısı yerine, dağılımın TAMAMINI gösterir)

Kullanım: import edip pnl (np.ndarray) + ts (pd.Series, aynı sırada) ver.
"""

import numpy as np
import pandas as pd


def top_n_contribution(pnl: np.ndarray, n: int) -> dict:
    """En iyi N işlemin toplam kâra (pozitif toplam değil, NET toplama) katkı yüzdesi."""
    if len(pnl) == 0:
        return {"n_islem": 0, "katki_pct": None}
    sorted_pnl = np.sort(pnl)[::-1]
    total = pnl.sum()
    top_n_sum = sorted_pnl[:n].sum()
    pct = (top_n_sum / total * 100) if total != 0 else float("nan")
    return {"n_islem": min(n, len(pnl)), "top_n_toplam": round(float(top_n_sum), 2),
            "genel_toplam": round(float(total), 2), "katki_pct": round(float(pct), 1)}


def trimmed_total(pnl: np.ndarray, trim_n: int) -> dict:
    """En iyi trim_n işlem ÇIKARILINCA geriye kalan toplam/ortalama — 'şanssız/ortalama
    bir dönemde gerçekçi olarak ne beklenir' sorusuna yakın bir cevap."""
    if len(pnl) <= trim_n:
        return {"n_kalan": 0, "toplam": None}
    sorted_idx = np.argsort(pnl)[::-1]
    keep_idx = sorted_idx[trim_n:]
    remaining = pnl[keep_idx]
    return {
        "n_kalan": len(remaining), "toplam": round(float(remaining.sum()), 2),
        "ort": round(float(remaining.mean()), 2) if len(remaining) else None,
        "wr_pct": round(float((remaining > 0).mean() * 100), 1) if len(remaining) else None,
    }


def bootstrap_monthly(pnl: np.ndarray, trades_per_month: float, n_sims: int = 5000, seed: int = 42) -> dict:
    """İşlemleri (iadeli/replacement ile) rastgele örnekleyip sentetik 'ay'lar kurar —
    gerçek işlem SIRASINI değil, sadece işlem BÜYÜKLÜK DAĞILIMINI korur. Sonuç:
    simüle edilmiş aylık $ dağılımı — kaç %'i pozitif, medyan/persentiller ne."""
    if len(pnl) == 0:
        return {"n_sims": 0}
    rng = np.random.default_rng(seed)
    n_per_month = max(1, round(trades_per_month))
    monthly_totals = np.array([
        rng.choice(pnl, size=n_per_month, replace=True).sum() for _ in range(n_sims)
    ])
    return {
        "n_sims": n_sims, "trades_per_month": n_per_month,
        "pozitif_ay_pct": round(float((monthly_totals > 0).mean() * 100), 1),
        "medyan_aylik": round(float(np.median(monthly_totals)), 2),
        "p10_aylik": round(float(np.percentile(monthly_totals, 10)), 2),
        "p90_aylik": round(float(np.percentile(monthly_totals, 90)), 2),
        "ort_aylik": round(float(monthly_totals.mean()), 2),
    }


def summarize(label: str, pnl: np.ndarray, ts: pd.Series, days_span: float) -> None:
    """Tek çağrıda üç teşhisi de basar — herhangi bir backtest sonucuna eklenebilir."""
    n = len(pnl)
    trades_per_month = n / days_span * 30 if days_span > 0 else 0
    print(f"\n{'='*70}\n[Yoğunlaşma teşhisi] {label}  (n={n}, işlem/ay≈{trades_per_month:.0f})\n{'='*70}")

    for k in (1, 3, 5, 10):
        if n > k:
            t = top_n_contribution(pnl, k)
            print(f"  en iyi {k:>2} işlemin katkısı: %{t['katki_pct']:>6}  (toplam ${t['genel_toplam']})")

    for k in (5, 10):
        if n > k:
            tr = trimmed_total(pnl, k)
            print(f"  en iyi {k} işlem ÇIKARILINCA: n={tr['n_kalan']} toplam=${tr['toplam']} "
                  f"ort=${tr['ort']} WR%={tr['wr_pct']}")

    if trades_per_month >= 1:
        bs = bootstrap_monthly(pnl, trades_per_month)
        print(f"  bootstrap aylık simülasyon ({bs['n_sims']} deneme, ay başına ~{bs['trades_per_month']} işlem):")
        print(f"    pozitif ay oranı: %{bs['pozitif_ay_pct']}")
        print(f"    medyan ay: ${bs['medyan_aylik']}  (p10=${bs['p10_aylik']}  p90=${bs['p90_aylik']})")
