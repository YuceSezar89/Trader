"""
HA_Cross'ta üçlü kombinasyon: all_up + TA-kovalama(1h/4h) + HA-hizalanması —
rsi_cross_ta_triple_combo_bt.py'nin HA_Cross replikasyonu (24 Tem 2026,
kullanıcı isteği).

Girdi: ha_cross_ta_percentile_bt.py'nin cache'lediği
(_cache_ha_cross_ta_percentile.parquet) all_up+pct_1h/pct_4h/slope_1h/slope_4h
verisi + burada yeniden hesaplanan HA-hizalanması (1h/4h recursive Heikin-Ashi
yönü, tf_alignment_gate.py::_heikin_ashi_bull ile birebir).

Kullanım: python -m research.pattern_lab.ha_cross_ta_triple_combo_bt
(önce ha_cross_ta_percentile_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _ha_bull_series
from research.pattern_lab.rsi_cross_ta_percentile_bt import _decompose

_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_BASE_TH = 55
_EXTREME_PCT = 80
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_ha_cross_ta_percentile.parquet")
_HA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_ha_cross_ta_ha.parquet")


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


def _compute_with_ha() -> pd.DataFrame:
    if os.path.exists(_HA_CACHE_PATH):
        print(f"[cache] {_HA_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_HA_CACHE_PATH)

    cached = pd.read_parquet(_CACHE_PATH)
    cached = _add_all_up(cached)
    cached = cached[cached["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    print(f"[all_up=True + TA bileşenleri popülasyonu] n={len(cached)}\n")

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
    df["ha_aligned"] = (df["ha_bull_1h"] > 0.5) & (df["ha_bull_4h"] > 0.5)
    print(f"\n[HA hesaplanabilir popülasyon] n={len(df)}\n")

    ta_base = (df["pct_1h"] >= _BASE_TH) & (df["pct_4h"] >= _BASE_TH)
    kovalama = ((df["pct_1h"] >= _EXTREME_PCT) & (df["slope_1h"] > 0)) | \
               ((df["pct_4h"] >= _EXTREME_PCT) & (df["slope_4h"] > 0))
    ha = df["ha_aligned"]

    print("=" * 78)
    print("ÖZET TABLO — sırayla ekleme etkisi (HA_Cross)")
    print("=" * 78)
    steps = [
        ("all_up (filtresiz)", np.ones(len(df), dtype=bool)),
        ("+ TA-base (pct>=55)", ta_base),
        ("+ kovalama (pct>=80&rising)", ta_base & kovalama),
        ("+ HA-hizalı (üçlü)", ta_base & kovalama & ha),
    ]
    for label, mask in steps:
        s = _stats(df[mask]["fwd_ret"].to_numpy())
        print(f"  {label:32}: n={s['n']:>5}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")

    triple_mask = ta_base & kovalama & ha
    triple = df[triple_mask]
    rest = df[~triple_mask]

    print("\n" + "=" * 78)
    print("ÜÇLÜ KOMBİNASYON — tam derin doğrulama")
    print("=" * 78)
    if len(triple) < 30:
        print(f"  Örneklem çok küçük (n={len(triple)}), derin doğrulama atlanıyor.")
        return

    df["_g"] = triple_mask
    _deep_validate("HA_Cross TA-base + kovalama + HA (üçlü)", triple, rest, df)

    print(f"\n  PF şüphesi (medyan): {_decompose(triple['fwd_ret'].to_numpy())}")

    days_span = (triple["opened_at"].max() - triple["opened_at"].min()).total_seconds() / 86400
    summarize("HA_Cross üçlü kombinasyon — fwd_ret% serisi", triple["fwd_ret"].to_numpy(), triple["opened_at"], days_span)


if __name__ == "__main__":
    main()
