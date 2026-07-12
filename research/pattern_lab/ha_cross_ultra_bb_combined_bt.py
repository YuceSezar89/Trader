"""
"Ultra hizalama" (v2-14, ha_cross_mtf_alignment_bt.py) ile daha önce
HA_Cross'un en iyi ekonomik filtresi olan BB-genişlik (v2-13,
ha_cross_bb_squeeze_bt.py) arasındaki İLİŞKİYİ ölçer:
1. İki filtre AYNI sinyalleri mi seçiyor (redundant) yoksa FARKLI
   popülasyonları mı seçiyor (tamamlayıcı, birleşince katkı katabilir)?
2. Standart (filtresiz) HA_Cross'a göre ikisinin ayrı ayrı ve BİRLEŞİK
   ekonomik/istatistiksel farkı ne?

Aynı disiplin: SADECE 3 Tem 19:22:16 sonrası + BB eşiği SADECE ilk yarıdan
(in-sample) türetilip ikinci yarıya (OOS) sabit uygulanıyor. Karşılaştırmayı
adil tutmak için TEK bir sinyal seti kullanılıyor: HA_Cross 5m (v2-14 ile
birebir aynı taban, v2-13'ün 5m+15m birleşiminden FARKLI — bu yüzden mutlak
$/ay rakamları v2-13'teki ile birebir kıyaslanamaz, konsept karşılaştırması
için taban sabitlendi).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.ha_cross_bb_squeeze_bt import (
    _bb_width_rank_series,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.ha_cross_combined_test import (
    _fetch_ha_cross_signals,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.ha_cross_mtf_alignment_bt import (
    CONFIRM_TFS,  # pylint: disable=wrong-import-position
)
from research.pattern_lab.mtf_helpers import (  # pylint: disable=wrong-import-position
    _confirm_count,
    _fetch_dir_data,
)
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (  # pylint: disable=wrong-import-position
    MIN_HISTORY,
    _fetch_symbol_history,
    _signal_bar_ts,
)

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
        "pf": (
            round(float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())), 3)
            if (pnl < 0).any()
            else float("inf")
        ),
        "avg_usd": round(float(pnl.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(per_month, 1),
    }


def _build_dataset() -> pd.DataFrame:
    rows = []
    sigs = _fetch_ha_cross_signals("5m")
    print(f"5m: {len(sigs):,} kapanmış HA_Cross sinyali (3 Tem 19:22 sonrası)\n")

    for symbol, sub in sigs.groupby("symbol"):
        hist_5m = _fetch_symbol_history(symbol, "5m")
        if len(hist_5m) < MIN_HISTORY:
            continue
        hist_5m = hist_5m.sort_values("ts").reset_index(drop=True)
        ts_to_idx = {t: i for i, t in enumerate(hist_5m["ts"])}
        bb_rank = _bb_width_rank_series(hist_5m)

        dir_data = _fetch_dir_data(symbol, CONFIRM_TFS)
        if dir_data is None:
            continue

        for _, row in sub.iterrows():
            i = ts_to_idx.get(_signal_bar_ts(row["opened_at"], "5m"))
            if i is None or i >= len(bb_rank):
                continue
            bb_val = bb_rank.iloc[i]
            if not np.isfinite(bb_val):
                continue

            confirm_count = _confirm_count(
                dir_data, CONFIRM_TFS, row["opened_at"], want_bullish=(row["signal_type"] == "Long")
            )
            if confirm_count is None:
                continue

            rows.append(
                {
                    "bb_rank": bb_val,
                    "confirm_count": confirm_count,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                }
            )

    return pd.DataFrame(rows)


def run():
    df = _build_dataset()
    print(f"toplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük.")
        return

    corr = df["bb_rank"].corr(df["confirm_count"])
    print(f"bb_rank ile confirm_count arasındaki korelasyon: {corr:.3f}\n")

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

    bb_threshold = float(is_df["bb_rank"].quantile(0.667))
    print(f"in-sample olay: {len(is_df)} | SABİT BB-genişlik eşiği: {bb_threshold:.2f}\n")

    bb_mask = oos_df["bb_rank"] >= bb_threshold
    ultra_mask = oos_df["confirm_count"] == 3

    print("── OOS 2x2 örtüşme tablosu (BB-geniş x ULTRA) ──")
    print(f"{'grup':30} {'n':>6} {'WR%':>6} {'PF':>7} {'ort $/işlem':>12} {'$/ay':>10}")
    groups = {
        "baseline (hepsi)": pd.Series(True, index=oos_df.index),
        "sadece BB-geniş": bb_mask & ~ultra_mask,
        "sadece ULTRA": ultra_mask & ~bb_mask,
        "İKİSİ BİRDEN (BB-geniş ∧ ULTRA)": bb_mask & ultra_mask,
        "hiçbiri": ~bb_mask & ~ultra_mask,
        "─ tek başına BB-geniş (ULTRA farketmez)": bb_mask,
        "─ tek başına ULTRA (BB farketmez)": ultra_mask,
    }
    for name, mask in groups.items():
        sub = oos_df[mask]
        s = _dollar_stats(sub["realized_pnl"].to_numpy() / 100, oos_days)
        if s.get("n", 0) == 0:
            print(f"{name:30} {'0':>6}")
            continue
        print(
            f"{name:30} {s['n']:>6} {s['wr']:>6} {s['pf']:>7} "
            f"{s['avg_usd']:>12} {s['usd_per_month']:>10}"
        )

    n_bb = int(bb_mask.sum())
    n_ultra = int(ultra_mask.sum())
    n_both = int((bb_mask & ultra_mask).sum())
    print(
        f"\nörtüşme: BB-geniş grubunun %{100*n_both/n_bb:.0f}'i aynı zamanda ULTRA "
        f"({n_both}/{n_bb}) | ULTRA grubunun %{100*n_both/n_ultra:.0f}'i aynı zamanda BB-geniş "
        f"({n_both}/{n_ultra})"
    )


if __name__ == "__main__":
    run()
