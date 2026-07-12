"""
regime_score.py'deki kompozit piyasa rejimi skorunu (piyasa genişliği 3-gün
EMA + BTC günlük SMA20 trendi, ikisi de percentile rank) RSI_Cross Long
sinyallerine uygular — threshold_optimizer.py'nin AYNI 3-kapılı disiplini
(IS/OOS + split-period + placebo) ile, tek değişkenli arama olarak.

Veri: `signals` tablosundaki GERÇEK kapanmış RSI_Cross Long sinyalleri
(alpha/beta testinde kullanılanla aynı, look-ahead yok).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.regime_score import build_regime_score  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import _fetch_with_volume  # pylint: disable=wrong-import-position
from research.pattern_lab.threshold_optimizer import (  # pylint: disable=wrong-import-position
    _fetch_signals, _run_single_var_on_df,
)

INDICATOR = "RSI_Cross(9,24)"
DIRECTION = "Long"
# Uçlardan çekilmiş percentile adayları — 10/90 gibi uç dilimler tek bir dar
# tarihsel olayı "eşik" gibi gösterip split-period'da çökebiliyor (bkz. ilk
# deneme: regime_score<=7.25, OOS ilk yarıda n=0). 20-80 aralığı daha geniş/
# daha az kırılgan alt kümeler zorluyor.
PERCENTILES = [20, 30, 40, 50, 60, 70, 80]


def run():
    print("Piyasa rejimi skoru hesaplanıyor (609 sembol, 45 gün)...")
    df_all = _fetch_with_volume()
    regime = build_regime_score(df_all)
    print(f"Rejim skoru: {len(regime):,} zaman noktası, ort={regime.mean():.1f}\n")

    sig_df = _fetch_signals(INDICATOR, DIRECTION)
    sig_df = sig_df.sort_values("opened_at").reset_index(drop=True)

    regime_df = regime.reset_index()
    regime_df.columns = ["ts", "regime_score"]
    regime_df = regime_df.sort_values("ts")

    merged = pd.merge_asof(
        sig_df, regime_df, left_on="opened_at", right_on="ts", direction="backward",
    ).dropna(subset=["regime_score"])
    print(f"{INDICATOR} — {DIRECTION}: {len(sig_df):,} sinyal, {len(merged):,} rejim skoruyla eşleşti")

    label = f"{INDICATOR} — {DIRECTION} — REJİM SKORU (20-80 persentil)"
    _run_single_var_on_df(label, merged, "regime_score", percentiles=PERCENTILES)


if __name__ == "__main__":
    run()
