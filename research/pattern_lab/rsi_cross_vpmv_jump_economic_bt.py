"""
Ekonomik etki ölçümü — rsi_cross_vpmv_jump_bt.py bulgusunun (tek-bar VPMV
sıçraması, split-period'da sağlam çıktı) GERÇEK $ etkisini ölçer.

do_break_gauss_economic_bt.py ile AYNI OOS disiplini: sıçrama eşiği (üst tercil,
66.7. percentile) SADECE ilk yarıdan (in-sample) türetilip SABİT olarak ikinci
yarıya (out-of-sample) uygulanıyor — look-ahead'siz, "geçmiş veriyle kalibre
edip bugünden itibaren canlıya alsaydık" senaryosu. $ dönüşümü de aynı
konvansiyon: POSITION_USD=100, FEE_RATE=0.0005/taraf (round-trip %0.1).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.vpmv import compute_series  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    MIN_HISTORY, INTERVALS, _fetch_signals, _fetch_symbol_history,
)
from research.pattern_lab.do_break_gauss_economic_bt import (  # pylint: disable=wrong-import-position
    POSITION_USD, ROUND_TRIP_FEE, _dollar_stats,
)


def run():
    all_pairs = []

    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if len(hist) < MIN_HISTORY:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            series_long = compute_series(hist, "Long")
            series_short = compute_series(hist, "Short")

            for _, row in sub.iterrows():
                i = ts_to_idx.get(row["opened_at"])
                if i is None or i - 1 < 0 or i + 1 >= len(hist):
                    continue
                series = series_long if row["signal_type"] == "Long" else series_short
                pre_v = series.iloc[i - 1]
                post_v = series.iloc[i + 1]
                if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                    continue
                all_pairs.append({
                    "jump": post_v - pre_v,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(all_pairs)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük, güvenilir analiz yapılamaz.")
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

    oos_threshold = float(is_df["jump"].quantile(0.667))
    print(f"in-sample olay: {len(is_df)} | SABİT (OOS'a uygulanan) sıçrama eşiği: {oos_threshold:.2f}\n")

    oos_high = oos_df[oos_df["jump"] >= oos_threshold]
    oos_rest = oos_df[oos_df["jump"] < oos_threshold]

    print(f"── Out-of-sample ekonomik etki (${POSITION_USD:.0f} pozisyon, "
          f"round-trip fee ${ROUND_TRIP_FEE:.2f}) ──")
    print(f"{'grup':32} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, sub in (
        ("baseline (OOS, tüm RSI_Cross)", oos_df),
        (f"VPMV sıçraması YÜKSEK (eşik>={oos_threshold:.1f})", oos_high),
        ("VPMV sıçraması düşük/orta", oos_rest),
    ):
        rets = sub["realized_pnl"].to_numpy() / 100
        s = _dollar_stats(rets, oos_days)
        if s.get("n", 0) == 0:
            print(f"{name:32} {'0':>6}")
            continue
        print(f"{name:32} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")


if __name__ == "__main__":
    run()
