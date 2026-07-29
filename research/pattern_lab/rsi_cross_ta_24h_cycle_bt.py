"""
Seçenek 5: TA döngü uzunluğu — 12 saat yerine 24 saat (UTC 00:00'da tek
sıfırlama) denendiğinde percentile-uyumu bulgusu değişiyor mu? (23-24 Tem
2026, kullanıcı yokken tam yetkiyle test edildi.)

rsi_cross_ta_percentile_bt.py'nin cache'lediği all_up/fwd_ret verisini
kullanır (yeniden 5m OHLCV/compute_components hesaplamaz), sadece 1h/4h
close serilerini yeniden çekip net_ta'yı 24h döngüyle hesaplar, aynı
percentile+eşik-taraması (IS/OOS) yöntemini uygular ve 12h sonucuyla
(percentile>=55, PF=4.124, n=778) karşılaştırır.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_24h_cycle_bt
(önce rsi_cross_ta_percentile_bt.py çalıştırılmış olmalı — cache dosyasını kullanır)
"""

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_percentile_bt import _CACHE_PATH

_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_CYCLE_MS_24H = 24 * 3600 * 1000
_LOOKBACK = 200
_MIN_LOOKBACK = 50
_MIN_N = 30
_THRESHOLDS = [0, 50, 55, 60, 65, 70, 75, 80]


def _total_amount_24h(ts: pd.Series, close: np.ndarray) -> np.ndarray:
    t_ms = ts.astype("int64").to_numpy() // 10**6
    cycle = (t_ms // _CYCLE_MS_24H) * _CYCLE_MS_24H
    n = len(close)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            p = close[i - 1]
            if p != 0:
                net[i] = net[i - 1] + (close[i] - p) / p * 100.0
            else:
                net[i] = net[i - 1]
    return net


def _percentile_at(net_arr: np.ndarray, j: int) -> float:
    s = max(0, j - _LOOKBACK)
    q = net_arr[s:j]
    if len(q) < _MIN_LOOKBACK:
        return np.nan
    return float((q <= net_arr[j]).mean() * 100.0)


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "ort_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    cached = pd.read_parquet(_CACHE_PATH)
    cached = _add_all_up(cached)
    cached = cached[cached["all_up"]].reset_index(drop=True)
    print(f"[all_up=True popülasyonu, 12h-cycle cache'ten] n={len(cached)}\n")

    conn = _conn()
    cur = conn.cursor()
    symbols = cached["symbol"].unique()
    print(f"{len(symbols)} sembol için 1h/4h serisi 24h-döngüyle yeniden hesaplanacak")

    pct_cols = {tf: [np.nan] * len(cached) for tf in _TFS}
    for si, sym in enumerate(symbols):
        sub_idx = cached.index[cached["symbol"] == sym]
        sub_times = cached.loc[sub_idx, "opened_at"]
        for tf in _TFS:
            cur.execute(f"SELECT bucket, close FROM {_TABLE[tf]} WHERE symbol=%s ORDER BY bucket ASC", (sym,))
            rows = cur.fetchall()
            if not rows:
                continue
            d = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
            net_arr = _total_amount_24h(d["bucket"], d["close"].to_numpy())
            b_arr = d["bucket"].to_numpy()
            for idx, t in zip(sub_idx, sub_times):
                j = np.searchsorted(b_arr, np.datetime64(t), side="right") - 1
                if j < 0:
                    continue
                pct_cols[tf][idx] = _percentile_at(net_arr, j)
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    for tf in _TFS:
        cached[f"pct24_{tf}"] = pct_cols[tf]
    df = cached.dropna(subset=["pct24_1h", "pct24_4h"]).reset_index(drop=True)
    print(f"\n[percentile hesaplanabilir popülasyon, 24h-cycle] n={len(df)}\n")

    df = df.sort_values("opened_at").reset_index(drop=True)
    mid = df["opened_at"].iloc[len(df) // 2]
    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]
    print(f"IS: n={len(is_df)}  OOS: n={len(oos_df)}\n")

    print("=" * 78)
    print("Eşik taraması (24h-cycle, SADECE IS verisiyle)")
    print("=" * 78)
    print(f"{'esik':>6} | {'IS n':>6} {'IS WR%':>7} {'IS PF':>8} | {'OOS n':>6} {'OOS WR%':>7} {'OOS PF':>8}")
    print("-" * 70)
    best_th, best_is_pf = None, -1.0
    for th in _THRESHOLDS:
        is_sub = is_df[(is_df["pct24_1h"] >= th) & (is_df["pct24_4h"] >= th)]
        oos_sub = oos_df[(oos_df["pct24_1h"] >= th) & (oos_df["pct24_4h"] >= th)]
        is_s = _stats(is_sub["fwd_ret"].to_numpy())
        oos_s = _stats(oos_sub["fwd_ret"].to_numpy())
        print(f"{th:>6} | {is_s['n']:>6} {is_s['wr'] if is_s['wr'] is not None else '-':>7} "
              f"{is_s['pf'] if is_s['pf'] is not None else '-':>8} | "
              f"{oos_s['n']:>6} {oos_s['wr'] if oos_s['wr'] is not None else '-':>7} "
              f"{oos_s['pf'] if oos_s['pf'] is not None else '-':>8}")
        if is_s["n"] >= _MIN_N and is_s["pf"] is not None and isinstance(is_s["pf"], float) and is_s["pf"] > best_is_pf:
            best_is_pf = is_s["pf"]
            best_th = th

    print(f"\n=== IS'te en iyi görünen eşik: {best_th} (PF={best_is_pf}) ===")
    oos_best = oos_df[(oos_df["pct24_1h"] >= best_th) & (oos_df["pct24_4h"] >= best_th)]
    print(f"  OOS performansı: {_stats(oos_best['fwd_ret'].to_numpy())}")

    full_mask = (df["pct24_1h"] >= best_th) & (df["pct24_4h"] >= best_th)
    df["_g"] = full_mask
    group, rest = df[full_mask], df[~full_mask]
    _deep_validate(f"24h-cycle percentile>={best_th} (1h+4h)", group, rest, df)

    print(f"\n=== KARŞILAŞTIRMA: 12h-cycle (percentile>=55, PF=4.124, n=778) vs 24h-cycle (percentile>={best_th}) ===")
    print(f"  24h-cycle: {_stats(group['fwd_ret'].to_numpy())}")

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    summarize(f"24h-cycle percentile>={best_th} — fwd_ret% serisi", group["fwd_ret"].to_numpy(), group["opened_at"], days_span)


if __name__ == "__main__":
    main()
