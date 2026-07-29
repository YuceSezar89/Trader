"""
"Kovalama'nın şüphesi": elaborate TA formülümüz (percentile-rank + 12h
döngü + slope + HA) gerçekten ek bilgi mi katıyor, yoksa sadece kripto'da
zaten bilinen kısa-vadeli momentum/serisel-korelasyon etkisini süslü bir
şekilde mi yeniden paketliyoruz? (24 Tem 2026, kullanıcı isteği)

Test: aynı all_up popülasyonunda, elaborate kovalama filtresi yerine
KABA/NAİF bir momentum ölçüsü (1h chart'ta ham trailing 24-bar % getiri,
percentile YOK, döngü-sıfırlama YOK, sadece düz sayı) ile eşik taraması
yapılıyor — IS/OOS disiplinli. Eğer naif versiyon benzer PF verirse,
elaborate makinemiz gerçekten bir şey katmıyor demektir.

Kullanım: python -m research.pattern_lab.rsi_cross_naive_momentum_bt
(önce rsi_cross_ta_ha_overlap_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_naive_momentum.parquet")

_LOOKBACK_BARS_1H = 24  # 24 saat, TA'nın 12h döngüsünün 2 katı — kaba karşılaştırma
_THRESHOLDS = [0, 1, 2, 3, 4, 5, 7, 10, 15]  # % ham getiri eşikleri
_MIN_N = 30
_PLACEBO_ITER = 300


def _fetch_1h_close(cur, symbol: str) -> pd.DataFrame:
    cur.execute("SELECT bucket, close FROM cagg_1h WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    return df


def _collect_with_raw_mom() -> pd.DataFrame:
    df = pd.read_parquet(_HA_CACHE_PATH)  # zaten all_up=True filtreli
    df = df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    print(f"[all_up=True popülasyonu] n={len(df)}\n")

    conn = _conn()
    cur = conn.cursor()
    symbols = df["symbol"].unique()
    print(f"{len(symbols)} sembol için ham 24-bar (1h) getiri hesaplanacak")

    raw_mom = np.full(len(df), np.nan)
    for si, sym in enumerate(symbols):
        sub_idx = df.index[df["symbol"] == sym]
        sub_times = df.loc[sub_idx, "opened_at"]
        d = _fetch_1h_close(cur, sym)
        if d.empty or len(d) < _LOOKBACK_BARS_1H + 1:
            continue
        close = d["close"].to_numpy()
        b_arr = d["bucket"].to_numpy()
        for idx, t in zip(sub_idx, sub_times):
            j = np.searchsorted(b_arr, np.datetime64(t), side="right") - 1
            if j < _LOOKBACK_BARS_1H:
                continue
            p_now, p_before = close[j], close[j - _LOOKBACK_BARS_1H]
            if p_before == 0:
                continue
            raw_mom[idx] = (p_now - p_before) / p_before * 100.0
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    df["raw_mom_24h"] = raw_mom
    df = df.dropna(subset=["raw_mom_24h"]).reset_index(drop=True)
    print(f"\n[ham momentum hesaplanabilir] n={len(df)}\n")
    return df


def _collect_cached() -> pd.DataFrame:
    if os.path.exists(_CACHE_PATH):
        print(f"[cache] {_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_CACHE_PATH)
    df = _collect_with_raw_mom()
    if not df.empty:
        df.to_parquet(_CACHE_PATH)
    return df


def _stats(rets):
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {"n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
            "pf": round(float(g / l), 3) if l > 0 else float("inf")}


def main() -> None:
    df = _collect_cached()
    df = df.sort_values("opened_at").reset_index(drop=True)
    mid = df["opened_at"].iloc[len(df) // 2]
    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]

    print("=" * 78)
    print("NAİF MOMENTUM (ham 24-bar 1h getiri) eşik taraması — SADECE IS")
    print("=" * 78)
    print(f"{'esik%':>6} | {'IS n':>6} {'IS PF':>8} | {'OOS n':>6} {'OOS PF':>8}")
    print("-" * 50)
    best_th, best_is_pf = None, -1.0
    for th in _THRESHOLDS:
        is_sub = is_df[is_df["raw_mom_24h"] >= th]
        oos_sub = oos_df[oos_df["raw_mom_24h"] >= th]
        is_s = _stats(is_sub["fwd_ret"].to_numpy())
        oos_s = _stats(oos_sub["fwd_ret"].to_numpy())
        print(f"{th:>6} | {is_s['n']:>6} {is_s['pf'] if is_s['pf'] is not None else '-':>8} | "
              f"{oos_s['n']:>6} {oos_s['pf'] if oos_s['pf'] is not None else '-':>8}")
        if is_s["n"] >= _MIN_N and is_s["pf"] is not None and isinstance(is_s["pf"], float) and is_s["pf"] > best_is_pf:
            best_is_pf = is_s["pf"]
            best_th = th

    print(f"\n=== IS'te en iyi ham-momentum eşiği: {best_th}% (PF={best_is_pf}) ===")
    oos_best = oos_df[oos_df["raw_mom_24h"] >= best_th]
    print(f"  OOS performansı: {_stats(oos_best['fwd_ret'].to_numpy())}")

    naive_mask = df["raw_mom_24h"] >= best_th
    naive_group, naive_rest = df[naive_mask], df[~naive_mask]
    df["_g"] = naive_mask
    _deep_validate(f"Naif momentum >= {best_th}%", naive_group, naive_rest, df)

    print("\n" + "=" * 78)
    print("KARŞILAŞTIRMA: Naif momentum vs bizim elaborate kovalama filtremiz")
    print("=" * 78)
    kovalama = ((df["pct_1h"] >= 90) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= 90) & (df["slope_4h"] > 0))
    ta_base = (df["pct_1h"] >= 55) & (df["pct_4h"] >= 55)
    kov_group = df[ta_base & kovalama]
    print(f"  Naif momentum (eşik={best_th}%)      : {_stats(naive_group['fwd_ret'].to_numpy())}")
    print(f"  Bizim kovalama (TA-base+percentile90) : {_stats(kov_group['fwd_ret'].to_numpy())}")

    overlap = len(set(naive_group.index) & set(kov_group.index))
    print(f"\n  örtüşme: naif grubun %{overlap/max(1,len(naive_group))*100:.1f}'i kovalama grubunda da var")
    print(f"           kovalama grubunun %{overlap/max(1,len(kov_group))*100:.1f}'i naif grupta da var")

    days_span = (naive_group["opened_at"].max() - naive_group["opened_at"].min()).total_seconds() / 86400
    if len(naive_group) >= 10:
        summarize(f"Naif momentum >= {best_th}% — fwd_ret% serisi", naive_group["fwd_ret"].to_numpy(), naive_group["opened_at"], days_span)

    print("\n" + "=" * 78)
    print("'EN ÇOK YÜKSELENLER İÇİNDE MUTLAKA BİR ÖRÜNTÜ VAR MI?' — naif momentum")
    print("grubunu TA durumuna göre ikiye ayırıyoruz: hâlâ tırmanıyor (kovalama-uyumlu)")
    print("vs zaten dönmüş/durmuş (percentile yüksek ama eğim negatif)")
    print("=" * 78)
    still_rising = naive_group[((naive_group["pct_1h"] >= 80) & (naive_group["slope_1h"] > 0)) |
                                ((naive_group["pct_4h"] >= 80) & (naive_group["slope_4h"] > 0))]
    already_turned = naive_group[~naive_group.index.isin(still_rising.index)]
    print(f"  Naif-momentum + HÂLÂ TIRMANIYOR (TA da onaylıyor) : {_stats(still_rising['fwd_ret'].to_numpy())}")
    print(f"  Naif-momentum + ZATEN DÖNMÜŞ/DURGUN (TA onaylamıyor): {_stats(already_turned['fwd_ret'].to_numpy())}")
    print(f"\n  (n dağılımı: hâlâ tırmanıyor={len(still_rising)}, zaten dönmüş={len(already_turned)}, toplam={len(naive_group)})")


if __name__ == "__main__":
    main()
