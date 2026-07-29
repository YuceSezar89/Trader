"""
TA-percentile(1h+4h) ile HA-hizalanması (tf_alignment_gate.py::_get_ha_alignment)
NE KADAR ÇAKIŞIYOR? Kullanıcının "ikisini birleştirsek?" sorusuna cevap
(24 Tem 2026). Tahmin: ikisi de aynı 1h/4h close'lardan türetilen "üst-TF
yön uyumu" ölçüyor, muhtemelen yüksek örtüşme + marjinal katkı.

HA formülü tf_alignment_gate.py::_heikin_ashi_bull ile BİREBİR aynı (recursive
HA close/open), ama live kod 60-bar pencerede taze başlatıyor — burada tüm
seri üzerinden bir kez hesaplanıyor (HA birkaç barda yakınsadığı için pratikte
fark etmez, tüm seride tek hesap çok daha hızlı).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_ha_overlap_bt
(önce rsi_cross_ta_percentile_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_percentile_bt import _CACHE_PATH

_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_BEST_TH = 55
_HA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_ta_ha.parquet")


def _ha_bull_series(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    ha_close = (o + h + l + c) / 4.0
    ha_open = ha_close.copy()
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(o)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return ha_close > ha_open


def _compute_with_ha() -> pd.DataFrame:
    if os.path.exists(_HA_CACHE_PATH):
        print(f"[cache] {_HA_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_HA_CACHE_PATH)

    cached = pd.read_parquet(_CACHE_PATH)
    cached = _add_all_up(cached)
    cached = cached[cached["all_up"]].dropna(subset=["pct_1h", "pct_4h"]).reset_index(drop=True)
    print(f"[all_up=True + percentile popülasyonu] n={len(cached)}\n")

    conn = _conn()
    cur = conn.cursor()
    symbols = cached["symbol"].unique()
    print(f"{len(symbols)} sembol için HA (1h/4h) hesaplanacak")

    bull_cols = {tf: np.full(len(cached), np.nan) for tf in _TFS}
    for si, sym in enumerate(symbols):
        sub_idx = cached.index[cached["symbol"] == sym]
        sub_times = cached.loc[sub_idx, "opened_at"]
        for tf in _TFS:
            cur.execute(
                f"SELECT bucket, open, high, low, close FROM {_TABLE[tf]} WHERE symbol=%s ORDER BY bucket ASC",
                (sym,),
            )
            rows = cur.fetchall()
            if not rows:
                continue
            d = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
            for c in ("open", "high", "low", "close"):
                d[c] = d[c].astype(float)
            bull = _ha_bull_series(d["open"].to_numpy(), d["high"].to_numpy(),
                                    d["low"].to_numpy(), d["close"].to_numpy())
            b_arr = d["bucket"].to_numpy()
            for idx, t in zip(sub_idx, sub_times):
                j = np.searchsorted(b_arr, np.datetime64(t), side="right") - 1
                if j < 0:
                    continue
                bull_cols[tf][idx] = float(bull[j])
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    for tf in _TFS:
        cached[f"ha_bull_{tf}"] = bull_cols[tf]
    cached.to_parquet(_HA_CACHE_PATH)
    return cached


def main() -> None:
    df = _compute_with_ha()
    df = df.dropna(subset=["ha_bull_1h", "ha_bull_4h"]).reset_index(drop=True)
    df["ha_aligned"] = (df["ha_bull_1h"] > 0.5) & (df["ha_bull_4h"] > 0.5)  # sinyaller hep Long
    print(f"\n[HA hesaplanabilir popülasyon] n={len(df)}\n")

    print("=" * 78)
    print("ÖRTÜŞME: TA-percentile>=55 (1h+4h) grubunda HA-hizalanma oranı ne?")
    print("=" * 78)
    ta_group = df[(df["pct_1h"] >= _BEST_TH) & (df["pct_4h"] >= _BEST_TH)]
    ta_rest = df[~((df["pct_1h"] >= _BEST_TH) & (df["pct_4h"] >= _BEST_TH))]
    print(f"  TA-percentile>=55 içinde HA-hizalı oranı : %{ta_group['ha_aligned'].mean()*100:.1f}  (n={len(ta_group)})")
    print(f"  TA-percentile<55  içinde HA-hizalı oranı : %{ta_rest['ha_aligned'].mean()*100:.1f}  (n={len(ta_rest)})")
    print(f"  Genel popülasyonda HA-hizalı oranı        : %{df['ha_aligned'].mean()*100:.1f}  (n={len(df)})")

    print("\n" + "=" * 78)
    print("TA grubunun İÇİNDE HA-hizalı vs HA-hizasız karşılaştırması")
    print("=" * 78)
    df["_g"] = (df["pct_1h"] >= _BEST_TH) & (df["pct_4h"] >= _BEST_TH) & df["ha_aligned"]
    combo = df[df["_g"]]
    ta_only_no_ha = ta_group[~ta_group["ha_aligned"]]
    print(f"  TA + HA ikisi de   : n={len(combo)}  ort_ret={combo['fwd_ret'].mean():.3f}%")
    print(f"  TA var, HA yok     : n={len(ta_only_no_ha)}  ort_ret={ta_only_no_ha['fwd_ret'].mean():.3f}%")
    _deep_validate("TA>=55 + HA-hizalı", combo, df[~df["_g"]], df)

    print("\n" + "=" * 78)
    print("KARŞILAŞTIRMA: TA-percentile>=55 TEK BAŞINA vs TA+HA KOMBİNE")
    print("=" * 78)

    def _stats(rets):
        rets = rets[~np.isnan(rets)]
        if len(rets) == 0:
            return {"n": 0}
        g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
        return {"n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
                "ort_%": round(float(rets.mean()), 3), "pf": round(float(g / l), 3) if l > 0 else float("inf")}

    print(f"  TA>=55 tek başına : {_stats(ta_group['fwd_ret'].to_numpy())}")
    print(f"  TA>=55 + HA-hizalı: {_stats(combo['fwd_ret'].to_numpy())}")

    print("\n" + "=" * 78)
    print("BASELINE: HA-hizalanması TEK BAŞINA (TA olmadan) all_up popülasyonunda işe yarıyor mu?")
    print("=" * 78)
    ha_only = df[df["ha_aligned"]]
    ha_not = df[~df["ha_aligned"]]
    print(f"  HA-hizalı  : {_stats(ha_only['fwd_ret'].to_numpy())}")
    print(f"  HA-hizasız : {_stats(ha_not['fwd_ret'].to_numpy())}")


if __name__ == "__main__":
    main()
