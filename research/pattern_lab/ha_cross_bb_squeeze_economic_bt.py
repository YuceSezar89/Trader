"""
Ekonomik etki ölçümü — ha_cross_bb_squeeze_bt.py bulgusu (BB genişliği YÜKSEK
= güvenilir, split-period'da sağlam) için. Aynı OOS disiplini: eşik SADECE
ilk yarıdan (in-sample) türetilip ikinci yarıya (out-of-sample) sabit
uygulanıyor, $100 pozisyon + gerçek fee (do_break_gauss_economic_bt.py deseni).

Ayrıca mum-şekli filtresiyle (HA_Cross'un daha önce en iyi ekonomik sonucu
veren filtresi, +$1410/ay) birleştirilince ne olduğu da test ediliyor.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.rsi_cross_vpmv_jump_bt import MIN_HISTORY, INTERVALS, _fetch_symbol_history  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_combined_test import _fetch_ha_cross_signals  # pylint: disable=wrong-import-position
from research.pattern_lab.ha_cross_bb_squeeze_bt import _bb_width_rank_series  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_candle_shape_bt import _classify  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import _dollar_stats  # pylint: disable=wrong-import-position


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_ha_cross_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış HA_Cross sinyali (3 Tem 19:22 sonrası)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if len(hist) < MIN_HISTORY:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            bb_rank = _bb_width_rank_series(hist)

            for _, row in sub.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i >= len(bb_rank):
                    continue
                val = bb_rank.iloc[i]
                if not np.isfinite(val):
                    continue
                bar = hist.iloc[i]
                kategori = _classify(bar)
                rows.append({
                    "bb_rank": val,
                    "kategori": kategori,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(rows)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük.")
        return

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max}")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]
    if len(is_df) < 30 or len(oos_df) < 30:
        print("In-sample/out-of-sample örneklemi çok küçük.")
        return

    oos_threshold = float(is_df["bb_rank"].quantile(0.667))
    print(f"in-sample olay: {len(is_df)} | SABİT BB-genişlik eşiği: {oos_threshold:.2f}\n")

    only_bb_mask = oos_df["bb_rank"] >= oos_threshold
    only_shape_mask = oos_df["kategori"] != "üst-fitil-baskın"
    combined_mask = only_bb_mask & only_shape_mask

    print(f"{'grup':38} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, mask in (
        ("baseline (OOS, tüm HA_Cross)", pd.Series(True, index=oos_df.index)),
        ("sadece BB-genişlik(geniş) filtreli", only_bb_mask),
        ("sadece mum-şekli filtreli", only_shape_mask),
        ("BİRLEŞİK (BB-genişlik + mum-şekli)", combined_mask),
    ):
        sub = oos_df[mask]
        rets = sub["realized_pnl"].to_numpy() / 100
        s = _dollar_stats(rets, oos_days)
        if s.get("n", 0) == 0:
            print(f"{name:38} {'0':>6}")
            continue
        print(f"{name:38} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")


if __name__ == "__main__":
    run()
