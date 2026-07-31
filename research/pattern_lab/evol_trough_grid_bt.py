"""
evol_trough_grid_bt.py — evol_trough_realistic_bt.py'nin SL/TP ATR-çarpanı
ızgara taraması. Veri BİR KEZ çekilip önbelleklenir (trough tespiti + ATR de
bir kez), sadece çıkış simülasyonu her (SL,TP) kombinasyonu için tekrarlanır.

Timeout sabit 8h=32 bar (evol_trough_realistic_bt.py'de 24h'e göre hafif
daha iyi çıkmıştı). "En yüksek PF" kombinasyonuna körü körüne güvenilmiyor —
bkz. memory (ha_cross_evol_exit_sweep_bt.py, 12 Tem): eşik sıkılaştıkça PF
görünüşte artabilir ama örneklem küçülüp artefakt riski büyür. Bu yüzden her
kombinasyon için split-period'ın İKİ yarısında da PF>1 şartı ayrıca
raporlanıyor — sadece havuzlanmış PF'ye bakılmıyor.

Kullanım: python -m research.pattern_lab.evol_trough_grid_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_atr  # noqa: E402  pylint: disable=wrong-import-position

import research.pattern_lab.evol_trough_bt as base  # noqa: E402  pylint: disable=wrong-import-position
from research.pattern_lab.evol_trough_realistic_bt import (  # noqa: E402  pylint: disable=wrong-import-position
    FEE_RATE,
    MAX_POSITION_USD,
    TARGET_RISK_USD,
    _simulate_exit,
)

TIMEOUT_BARS = 32  # 8h @ 15m
SL_GRID = [1.5, 2.0, 2.5, 3.0]
TP_GRID = [1.5, 2.0, 2.5, 3.0, 4.0]


def _pf(pnls: np.ndarray) -> float:
    win = pnls[pnls > 0].sum()
    loss = -pnls[pnls < 0].sum()
    return float(win / loss) if loss > 0 else float("inf")


def run() -> None:
    conn = base._conn()  # pylint: disable=protected-access
    bad = base._bad_symbols(conn.cursor())  # pylint: disable=protected-access
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = base._fetch(bad)  # pylint: disable=protected-access
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({base.DAYS} gün)\n")

    # --- bir kerelik önbellek: sembol -> (c,l,h,ts,atr,trough_idx listesi) ---
    cache: list[tuple] = []
    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < base.MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        l = g["low"].to_numpy(float)
        h = g["high"].to_numpy(float)
        ts = g["ts"]
        atr = calculate_atr(g, period=14).to_numpy(float)
        score = base._evol_series(g)  # pylint: disable=protected-access
        troughs, _peaks = base._find_troughs_and_peaks(score)  # pylint: disable=protected-access
        valid_troughs = [
            idx for idx in troughs
            if idx + TIMEOUT_BARS < len(c) and np.isfinite(atr[idx]) and atr[idx] > 0
        ]
        if valid_troughs:
            cache.append((sym, c, l, h, ts, atr, valid_troughs))

    total_troughs = sum(len(x[6]) for x in cache)
    print(f"[tarama] {n_syms} sembol, {total_troughs} geçerli trough (tüm kombinasyonlarda ortak)\n")

    rows = []
    for sl_mult in SL_GRID:
        for tp_mult in TP_GRID:
            breakeven_wr = sl_mult / (sl_mult + tp_mult) * 100
            records = []
            for sym, c, l, h, ts, atr, valid_troughs in cache:
                for idx in valid_troughs:
                    entry_price = c[idx]
                    atr_val = atr[idx]
                    sl_price = entry_price - sl_mult * atr_val
                    tp_price = entry_price + tp_mult * atr_val
                    sl_dist = entry_price - sl_price
                    position_usd = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)
                    pnl_pct, _reason, _hold = _simulate_exit(
                        c, l, h, idx, entry_price, sl_price, tp_price, TIMEOUT_BARS
                    )
                    pnl_usd = position_usd * (pnl_pct / 100.0) - position_usd * FEE_RATE * 2
                    records.append({"ts": ts.iloc[idx], "pnl_pct": pnl_pct, "pnl_usd": pnl_usd})

            rdf = pd.DataFrame(records)
            wr = float((rdf["pnl_pct"] > 0).mean() * 100)
            pf_all = _pf(rdf["pnl_usd"].to_numpy())

            mid = rdf["ts"].median()
            is_df = rdf[rdf["ts"] < mid]
            oos_df = rdf[rdf["ts"] >= mid]
            pf_is = _pf(is_df["pnl_usd"].to_numpy())
            pf_oos = _pf(oos_df["pnl_usd"].to_numpy())

            rows.append({
                "SL": sl_mult, "TP": tp_mult, "n": len(rdf),
                "breakeven_wr": round(breakeven_wr, 1), "wr": round(wr, 1),
                "pf_all": round(pf_all, 3), "pf_IS": round(pf_is, 3), "pf_OOS": round(pf_oos, 3),
                "toplam_usd": round(float(rdf["pnl_usd"].sum()), 0),
                "robust": pf_is > 1.0 and pf_oos > 1.0,
            })

    res = pd.DataFrame(rows).sort_values("pf_all", ascending=False)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 100)
    print(res.to_string(index=False))

    print(f"\nRobust (IS VE OOS'ta PF>1) kombinasyon sayısı: {res['robust'].sum()} / {len(res)}")
    print("\n--- Sadece robust kombinasyonlar, PF'ye göre sıralı ---")
    print(res[res["robust"]].sort_values("pf_all", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
