"""
Trajectory Lab — VPMV DECOMPOSITION, Faz 2 (bkz. CONTEXT_LAB_STATUS.md,
8 Ağustos — Mechanism aşaması, HENÜZ threshold/Strategy/Production DEĞİL).

Soru: "Momentum S0'dan önce bozulduğunda, marketin hacim ve volatilite
davranışı buna EŞLİK EDİYOR mu, yoksa momentum TEK BAŞINA mı bozuluyor?"

KRİTİK METODOLOJİ FARKI (kullanıcı, 8 Ağustos): referans noktası artık
close_pos'un zirvesi DEĞİL — MOMENTUM'un KENDİ zirvesi. Aynı winner/
loser örneklemi korunuyor (momentum_price_order.py ile AYNI deterministik
seçim) ki önceki (close_pos-anchored) sonuçlarla KARIŞMASIN.

Metodoloji:
  1. Her dizide momentum'un (mom) KENDİ zirve pozisyonu bulunur.
  2. O pozisyondan S0'a: Δmom, Δvol, Δvlt (momentum-anchored — Faz 1a'daki
     close_pos-anchored Δvol/Δvlt ile AYNI ŞEY DEĞİL, karıştırılmasın).
  3. Öncelik korelasyon değil, BİRLİKTE HAREKET ETME ORANI + yön (sign-
     count) — ama korelasyon da (Δmom↔Δvol, Δmom↔Δvlt) ayrıca raporlanır.
  4. Threshold/optimizasyon/strateji YOK — sadece dağılım.

Kullanım:
    python -m research.trajectory_lab.momentum_anchored_delta
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd

from database.engine import async_engine
from research.trajectory_lab.signal_sequence_scan import _build_sequences, _load_snapshots
from research.trajectory_lab.vpmv_decomposition_delta import (
    N_PER_GROUP,
    _fetch_components,
    _sample_sequences,
)

_FAMILIES = [
    ("HA_Cross_Long", "HA_Cross", "Long"),
    ("RSI_Cross_Long", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long", "Supertrend(10,3.0)", "Long"),
]
_CACHE_DIR = "research/trajectory_corpus"
_COMPONENTS = ["mom", "vol", "vlt"]


def _build_sequence_frame(sequences: list[dict], comp_map: dict) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        sids = seq["window"]["signal_id"].tolist()
        if any(sid not in comp_map for sid in sids):
            continue
        row = {"symbol": seq["symbol"], "outcome": seq["outcome"]}
        for c in ["mom", "vol", "vlt", "prc"]:
            for i, sid in enumerate(sids):
                row[f"{c}_{i}"] = comp_map[sid][c]
        rows.append(row)
    return pd.DataFrame(rows)


async def _build_family_cache(label: str, indicator: str, direction: str, cache_path: str) -> pd.DataFrame:
    snapshots = _load_snapshots(indicator, direction)
    sequences = _build_sequences(snapshots)
    winner_seqs = [s for s in sequences if s["outcome"] == "winner"]
    loser_seqs = [s for s in sequences if s["outcome"] == "loser"]

    winner_sample = _sample_sequences(winner_seqs, N_PER_GROUP)
    loser_sample = _sample_sequences(loser_seqs, N_PER_GROUP)
    print(f"\n{label}: winner örneklenen={len(winner_sample)}, loser örneklenen={len(loser_sample)}")

    all_signal_ids = set()
    for seq in winner_sample + loser_sample:
        all_signal_ids.update(seq["window"]["signal_id"].tolist())
    all_signal_ids = list(all_signal_ids)
    print(f"  hedefli sinyal sayısı: {len(all_signal_ids)}")

    comp_df = await _fetch_components(all_signal_ids)
    comp_map = comp_df.set_index("signal_id")[["mom", "vol", "vlt", "prc"]].to_dict("index")
    print(f"  bileşenler hesaplanabilen: {len(comp_map)}/{len(all_signal_ids)}")

    df = _build_sequence_frame(winner_sample + loser_sample, comp_map)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  -> önbelleğe yazıldı: {cache_path}")
    return df


def _analyze(label: str, df: pd.DataFrame) -> None:
    mom_cols = [f"mom_{i}" for i in range(4)]
    df = df.copy()
    df["mom_peak"] = df[mom_cols].to_numpy().argmax(axis=1)

    for c in _COMPONENTS:
        df[f"d_{c}"] = df.apply(lambda r, c=c: r[f"{c}_3"] - r[f"{c}_{int(r['mom_peak'])}"], axis=1)

    print(f"\n===== {label} — MOMENTUM-ANCHORED Δ (momentum zirvesi → S0) =====")
    for outcome in ("winner", "loser"):
        sub = df[df["outcome"] == outcome]
        n = len(sub)
        if n == 0:
            continue
        print(f"  [{outcome}] n={n}")
        for c in _COMPONENTS:
            col = f"d_{c}"
            s = sub[col]
            r = sub["d_mom"].corr(s) if c != "mom" else 1.0
            print(
                f"    Δ{c:4} ortalama={s.mean():+7.2f}  medyan={s.median():+7.2f}  std={s.std():6.2f}  "
                f"corr(Δmom,Δ{c})={r:+.3f}"
            )
        # birlikte hareket etme oranı (threshold yok, sadece işaret)
        mom_down = sub["d_mom"] < 0
        n_down = mom_down.sum()
        if n_down > 0:
            vol_down_with = (sub.loc[mom_down, "d_vol"] < 0).mean() * 100
            vol_up_with = (sub.loc[mom_down, "d_vol"] >= 0).mean() * 100
            vlt_down_with = (sub.loc[mom_down, "d_vlt"] < 0).mean() * 100
            vlt_up_with = (sub.loc[mom_down, "d_vlt"] >= 0).mean() * 100
            print(f"    momentum düşerken (n={n_down}):")
            print(f"      volume DA düşüyor: %{vol_down_with:.1f}   volume düşmüyor/yükseliyor: %{vol_up_with:.1f}")
            print(f"      volatility düşüyor: %{vlt_down_with:.1f}   volatility yükseliyor/sabit: %{vlt_up_with:.1f}")


async def run() -> None:
    all_frames = []
    for label, indicator, direction in _FAMILIES:
        cache_path = os.path.join(_CACHE_DIR, f"momentum_anchored_{label}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            print(f"\n{label}: önbellekten yüklendi ({cache_path}, {len(df)} dizi)")
        else:
            df = await _build_family_cache(label, indicator, direction, cache_path)
        df["source"] = label
        _analyze(label, df)
        all_frames.append(df)

    pooled = pd.concat(all_frames, ignore_index=True)
    _analyze("POOLED (3 aile)", pooled)


async def _main() -> None:
    try:
        await run()
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
