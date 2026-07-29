"""
TF-hizalanma + erken-ayrisma bulgusunun (bkz. [[ha_cross_tf_alignment_bt]],
[[rsi_cross_tf_alignment_bt]]) ekonomik etkisi — do_break_gauss_economic_bt.py
ile AYNI konvansiyon: POSITION_USD=100, FEE_RATE=0.0005/taraf,
round-trip=%0.1. signals.realized_pnl zaten % getiri (signal_lifecycle_manager.
py:40-45, kaldıraçsız/komisyonsuz ham fiyat farkı) — burada $ 'a çevrilip
komisyon düşülüyor.

Kullanım: python -m research.pattern_lab.tf_alignment_economic_bt
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from research.pattern_lab.ha_cross_tf_alignment_bt import _add_htf_alignment as _ha_align
from research.pattern_lab.ha_cross_tf_alignment_bt import _add_early_pct as _ha_early
from research.pattern_lab.ha_cross_tf_alignment_bt import _build_cohorts as _ha_cohorts
from research.pattern_lab.ha_cross_tf_alignment_bt import _EARLY_BARS as _HA_EARLY_BARS
from research.pattern_lab.ha_cross_tf_alignment_bt import _fetch as _ha_fetch

from research.pattern_lab.rsi_cross_tf_alignment_bt import _add_htf_alignment as _rsi_align
from research.pattern_lab.rsi_cross_tf_alignment_bt import _add_early_pct as _rsi_early
from research.pattern_lab.rsi_cross_tf_alignment_bt import _build_cohorts as _rsi_cohorts
from research.pattern_lab.rsi_cross_tf_alignment_bt import _EARLY_BARS as _RSI_EARLY_BARS
from research.pattern_lab.rsi_cross_tf_alignment_bt import _fetch as _rsi_fetch

POSITION_USD = 100.0
FEE_RATE = 0.0005
ROUND_TRIP_FEE = POSITION_USD * FEE_RATE * 2


def _dollar_stats(pnl_pct: pd.Series, days_span: float) -> dict:
    if len(pnl_pct) == 0 or days_span <= 0:
        return {"n": 0}
    pnl_usd = (pnl_pct / 100.0) * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl_usd.sum())
    return {
        "n": len(pnl_pct),
        "wr": round(float((pnl_usd > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl_usd.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(total / days_span * 30, 1),
    }


def _report(df: pd.DataFrame, source: str) -> None:
    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"\n[{source}] dönem: {days_span:.1f} gün")
    print(f"{'grup':38} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")

    for interval, idf in df.groupby("interval"):
        for sig_type in ["Long", "Short"]:
            sub = idf[
                (idf["signal_type"] == sig_type)
                & (idf["aligned_count"] == 2)
                & (idf["early_rank"] >= idf["early_rank"].quantile(0.667) if len(idf) > 0 else False)
            ]
            if len(sub) < 15:
                continue
            s = _dollar_stats(sub["realized_pnl"], days_span)
            name = f"{source} {interval} {sig_type} (hizalı+üst)"
            print(
                f"{name:38} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
                f"{s['total_usd']:>10} {s['usd_per_month']:>10}"
            )


def main() -> None:
    print("HA_Cross verisi hazırlanıyor...")
    ha = _ha_fetch(_HA_EARLY_BARS)
    ha = _ha_early(ha)
    ha = _ha_cohorts(ha)
    ha = _ha_align(ha)
    _report(ha, "HA_Cross")

    print("\nRSI_Cross verisi hazırlanıyor...")
    rsi = _rsi_fetch(_RSI_EARLY_BARS)
    rsi = _rsi_early(rsi)
    rsi = _rsi_cohorts(rsi)
    rsi = _rsi_align(rsi)
    _report(rsi, "RSI_Cross")

    print(
        f"\nNot: 'ort $/işlem' ve '$/ay', TEK bir ${POSITION_USD:.0f}'lık pozisyonun "
        f"art arda açılıp kapandığı basit toplama varsayımıyla (round-trip fee "
        f"${ROUND_TRIP_FEE:.2f}) — eşzamanlı pozisyon limiti/kaldıraç/compounding "
        f"dahil değil, sadece 'ortalama işlem ne kazandırır' tahmini."
    )


if __name__ == "__main__":
    main()
