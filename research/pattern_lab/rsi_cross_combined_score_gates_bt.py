"""
RSI Cross combined skorunun (bkz. [[rsi_cross_combined_score_bt]]) proje
standardı 3-kapılı doğrulamasının kalan 2 kapısı — 18 Tem 2026:
1. Placebo: combined_adj değerleri rastgele karıştırılıp (sinyal-skor bağı
   kırılıp) aynı korelasyon/tercile prosedürü tekrarlanır — gerçek etki
   hesaplama artefaktı DEĞİLSE placebo korelasyonu ~0 çıkmalı.
2. Split-period: opened_at medyanına göre ilk yarı/ikinci yarı ayrı test
   edilir — etki tek bir rejime/döneme özgü değilse ikisinde de tutmalı.

Kullanım: python -m research.pattern_lab.rsi_cross_combined_score_gates_bt
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from research.pattern_lab.rsi_cross_combined_score_bt import _add_combined_score, _fetch_signals

_SEED = 42


def _tercile_report(sub: pd.DataFrame, label: str) -> None:
    if len(sub) < 30:
        print(f"    {label}: yetersiz örnek (n={len(sub)})")
        return
    rho, p = spearmanr(sub["combined_adj"], sub["realized_pnl"])
    tercile = pd.qcut(sub["combined_adj"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
    g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
        ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
    )
    alt_wr = g.loc["alt", "wr"] * 100 if "alt" in g.index else float("nan")
    ust_wr = g.loc["üst", "wr"] * 100 if "üst" in g.index else float("nan")
    print(
        f"    {label}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f}), "
        f"alt_wr=%{alt_wr:.1f} -> üst_wr=%{ust_wr:.1f}"
    )


def _report_placebo(df: pd.DataFrame) -> None:
    print("\n=== KAPI 1: PLACEBO (combined_adj rastgele karıştırıldı) ===")
    rng = np.random.default_rng(_SEED)
    shuffled = df.copy()
    shuffled["combined_adj"] = rng.permutation(shuffled["combined_adj"].values)
    for sig_type in ["Long", "Short"]:
        sub_real = df[df["signal_type"] == sig_type].dropna(subset=["combined_adj"])
        sub_fake = shuffled[shuffled["signal_type"] == sig_type].dropna(subset=["combined_adj"])
        print(f"  {sig_type}:")
        _tercile_report(sub_real, "GERÇEK")
        _tercile_report(sub_fake, "PLACEBO")


def _report_split_period(df: pd.DataFrame) -> None:
    print("\n=== KAPI 2: SPLIT-PERIOD (opened_at medyanına göre iki yarı) ===")
    median_dt = df["opened_at"].median()
    print(f"  Bölünme noktası: {median_dt}")
    first_half = df[df["opened_at"] < median_dt]
    second_half = df[df["opened_at"] >= median_dt]
    for sig_type in ["Long", "Short"]:
        print(f"  {sig_type}:")
        _tercile_report(
            first_half[first_half["signal_type"] == sig_type].dropna(subset=["combined_adj"]),
            "İLK YARI",
        )
        _tercile_report(
            second_half[second_half["signal_type"] == sig_type].dropna(subset=["combined_adj"]),
            "İKİNCİ YARI",
        )


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    print("\nRSI Cross ağırlıklı combined skoru hesaplanıyor...")
    scored = _add_combined_score(df)
    scored = scored.dropna(subset=["combined_adj"])
    print(f"Geçerli skor: {len(scored)}")

    _report_placebo(scored)
    _report_split_period(scored)


if __name__ == "__main__":
    main()
