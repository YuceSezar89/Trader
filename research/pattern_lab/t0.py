"""
Adım 2 — Patlama anı (t0) tespiti.

Kural (tek ve mekanik): t0 = [çapa−SELECTION_DAYS, çapa−BREAKOUT_FWD_H]
aralığında, ileri BREAKOUT_FWD_H getirisinin en yüksek olduğu bar.

Kontrollerin t0'ı = eşleştiği vakanın t0'ı (aynı zaman damgası — piyasa
fazı karıştırıcısını düşürür).

Çıktı: onay tablosu (kullanıcı görmeden Adım 3'e geçilmez).
"""
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab import config as C

FWD_BARS = C.BREAKOUT_FWD_H * 12  # 24h × 12 adet 5m bar


def detect_t0(case_df: pd.DataFrame, anchor: datetime) -> dict:
    """Tek vaka için t0 + bağlam. df: tek sembolün 5m barları (ts sıralı)."""
    df = case_df.reset_index(drop=True)
    close = df["close"].to_numpy()
    fwd = pd.Series(close).shift(-FWD_BARS) / pd.Series(close) - 1

    t_lo = anchor - timedelta(days=C.SELECTION_DAYS)
    t_hi = anchor - timedelta(hours=C.BREAKOUT_FWD_H)
    mask = (df["ts"] >= t_lo) & (df["ts"] <= t_hi)
    cand = fwd.where(mask.to_numpy())
    i0 = int(cand.idxmax())

    # Son 24 saatte daha büyük (kesik pencereli) koşu var mı? — bilgi amaçlı bayrak
    tail_mask = df["ts"] > t_hi
    tail_run = 0.0
    if tail_mask.any():
        tail = df.loc[tail_mask, "close"]
        tail_run = float(tail.iloc[-1] / tail.iloc[0] - 1)

    return {
        "t0": df.loc[i0, "ts"],
        "fwd_24h_pct": round(float(cand.iloc[i0]) * 100, 1),
        "pre_48h_pct": round(float(close[i0] / close[max(0, i0 - 576)] - 1) * 100, 1),
        "son24h_kesik_kosu_pct": round(tail_run * 100, 1),
    }


def run() -> pd.DataFrame:
    with open(f"{C.CORPUS_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    anchor = datetime.fromisoformat(meta["anchor"])
    layer_a = pd.read_parquet(f"{C.CORPUS_DIR}/layer_a_detail.parquet")
    layer_b = pd.read_parquet(f"{C.CORPUS_DIR}/layer_b_universe.parquet")

    # Her 5m damgasında evrende kaç sembol var (sıralama güvenilirliği)
    uni_count = layer_b.groupby("ts")["symbol"].nunique()

    rows = []
    for case, ctrl in meta["pairs"]:
        cdf = layer_a[layer_a["symbol"] == case].sort_values("ts")
        info = detect_t0(cdf, anchor)
        t0 = info["t0"]
        rows.append({
            "vaka": case,
            "t0": t0,
            "ileri_24h_%": info["fwd_24h_pct"],
            "onceki_48h_%": info["pre_48h_pct"],
            "son24h_kesik_%": info["son24h_kesik_kosu_pct"],
            "evren_n(t0)": int(uni_count.get(t0, 0)),
            "kontrol": ctrl,
        })
    out = pd.DataFrame(rows).sort_values("ileri_24h_%", ascending=False)
    out.to_parquet(f"{C.CORPUS_DIR}/t0_table.parquet", index=False)
    return out


if __name__ == "__main__":
    tbl = run()
    print(tbl.to_string(index=False))
    guven = (tbl["evren_n(t0)"] >= 250).sum()
    print(f"\nSıralama güvenilir (evren≥250): {guven}/{len(tbl)} vaka")
