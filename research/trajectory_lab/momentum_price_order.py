"""
Trajectory Lab — VPMV DECOMPOSITION, Faz 1b (bkz. CONTEXT_LAB_STATUS.md,
8 Ağustos — Mechanism aşaması, HENÜZ threshold/Strategy/Production DEĞİL).

Faz 1a'nın bulgusu: close_pos peak→S0 mekanizmasında ayrışma esas olarak
momentum + price bileşenlerinden geliyor (volume/volatility'nin rolü
yok). Faz 1b İKİ AYRI soru sorar:

1. TEMPORAL ORDER: her dizide (S-3..S0) momentum'un KENDİ zirvesi ile
   price'ın KENDİ zirvesi hangi sırada geliyor — momentum önce mi, price
   önce mi, aynı anda mı? Winner/loser'da bu dağılım farklı mı? HENÜZ
   "momentum önce gelirse loser" gibi bir sonuca GİDİLMİYOR — sadece
   dağılım gözlemleniyor.
2. REDUNDANCY (ayrı, sıra sorusundan SONRA): momentum ve price ne kadar
   aynı bilgiyi taşıyor — "ikisi de price-türevi" varsayımı test ediliyor
   (partial correlation, signal_anatomy.py'deki close_pos↔buy_pct
   yöntemiyle aynı kapalı-form formül).

Veri: vpmv_decomposition_delta.py ile AYNI fetch mantığı, ama bu sefer
sonuç parquet'e CACHE'lenir (fetch maliyeti yüksekti — 3 aile için
~220 bar/sinyal geçmiş — tekrar ödenmesin).

Kullanım:
    python -m research.trajectory_lab.momentum_price_order
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd

from database.engine import async_engine
from research.trajectory_lab.behavior_scan import _select_balanced
from research.trajectory_lab.signal_sequence_scan import (
    _build_sequences,
    _fetch_close_pos_for_signals,
    _load_snapshots,
)
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


def _partial_corr(df: pd.DataFrame, y: str, x1: str, x2: str) -> float:
    r_y1 = df[y].corr(df[x1])
    r_y2 = df[y].corr(df[x2])
    r_12 = df[x1].corr(df[x2])
    denom = np.sqrt((1 - r_y2**2) * (1 - r_12**2))
    return (r_y1 - r_y2 * r_12) / denom if denom != 0 else np.nan


def _build_sequence_frame(sequences: list[dict], close_pos_map: dict, comp_map: dict) -> pd.DataFrame:
    """Her dizi için 4 pozisyonun (S-3..S0) mom/prc/close_pos serisini
    TAM olarak saklar (sadece delta değil — zirve index'i bulmak için)."""
    rows = []
    for seq in sequences:
        sids = seq["window"]["signal_id"].tolist()
        if any(sid not in close_pos_map or sid not in comp_map for sid in sids):
            continue
        mom_series = [comp_map[sid]["mom"] for sid in sids]
        prc_series = [comp_map[sid]["prc"] for sid in sids]
        cp_series = [close_pos_map[sid] for sid in sids]
        rows.append(
            {
                "symbol": seq["symbol"],
                "outcome": seq["outcome"],
                "mom_0": mom_series[0], "mom_1": mom_series[1], "mom_2": mom_series[2], "mom_3": mom_series[3],
                "prc_0": prc_series[0], "prc_1": prc_series[1], "prc_2": prc_series[2], "prc_3": prc_series[3],
                "cp_0": cp_series[0], "cp_1": cp_series[1], "cp_2": cp_series[2], "cp_3": cp_series[3],
            }
        )
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

    cp_df = await _fetch_close_pos_for_signals(all_signal_ids)
    close_pos_map = dict(zip(cp_df["signal_id"], cp_df["close_pos"]))
    comp_df = await _fetch_components(all_signal_ids)
    comp_map = comp_df.set_index("signal_id")[["mom", "prc"]].to_dict("index")
    print(f"  close_pos: {len(close_pos_map)}/{len(all_signal_ids)}  bileşenler: {len(comp_map)}/{len(all_signal_ids)}")

    df = _build_sequence_frame(winner_sample + loser_sample, close_pos_map, comp_map)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  -> önbelleğe yazıldı: {cache_path}")
    return df


def _classify_order(mom_peak: int, prc_peak: int) -> str:
    if mom_peak < prc_peak:
        return "Momentum önce"
    if mom_peak > prc_peak:
        return "Price önce"
    return "Aynı anda"


def _analyze(label: str, df: pd.DataFrame) -> None:
    mom_cols = [f"mom_{i}" for i in range(4)]
    prc_cols = [f"prc_{i}" for i in range(4)]
    df = df.copy()
    df["mom_peak"] = df[mom_cols].to_numpy().argmax(axis=1)
    df["prc_peak"] = df[prc_cols].to_numpy().argmax(axis=1)
    df["order"] = [_classify_order(m, p) for m, p in zip(df["mom_peak"], df["prc_peak"])]
    df["mom_dist_to_s0"] = 3 - df["mom_peak"]
    df["prc_dist_to_s0"] = 3 - df["prc_peak"]

    print(f"\n===== {label} — TEMPORAL ORDER (momentum zirvesi vs price zirvesi) =====")
    for outcome in ("winner", "loser"):
        sub = df[df["outcome"] == outcome]
        n = len(sub)
        if n == 0:
            continue
        counts = sub["order"].value_counts()
        print(f"  [{outcome}] n={n}")
        for label_o in ("Momentum önce", "Price önce", "Aynı anda"):
            c = counts.get(label_o, 0)
            print(f"    {label_o:15} %{c/n*100:5.1f}  (n={c})")
        print(
            f"    ortalama mom_dist_to_s0={sub['mom_dist_to_s0'].mean():.2f}  "
            f"ortalama prc_dist_to_s0={sub['prc_dist_to_s0'].mean():.2f}"
        )

    print(f"\n===== {label} — REDUNDANCY (momentum ↔ price, S0 seviyesinde) =====")
    df["d_close_pos"] = df["cp_3"] - df[["cp_0", "cp_1", "cp_2", "cp_3"]].max(axis=1)
    df["d_mom"] = df["mom_3"] - df.apply(lambda r: r[f"mom_{int(r['mom_peak'])}"], axis=1)
    df["d_prc"] = df["prc_3"] - df.apply(lambda r: r[f"prc_{int(r['prc_peak'])}"], axis=1)
    for outcome in ("winner", "loser"):
        sub = df[df["outcome"] == outcome]
        if len(sub) < 30:
            continue
        r_raw = sub["mom_3"].corr(sub["prc_3"])
        r_delta = sub["d_mom"].corr(sub["d_prc"])
        p_mom = _partial_corr(sub, "d_close_pos", "d_mom", "d_prc")
        p_prc = _partial_corr(sub, "d_close_pos", "d_prc", "d_mom")
        print(f"  [{outcome}] corr(mom@S0,prc@S0)={r_raw:+.3f}  corr(Δmom,Δprc)={r_delta:+.3f}")
        print(f"    partial corr(Δclose_pos, Δmom | Δprc) = {p_mom:+.3f}")
        print(f"    partial corr(Δclose_pos, Δprc | Δmom) = {p_prc:+.3f}")


async def run() -> None:
    all_frames = []
    for label, indicator, direction in _FAMILIES:
        cache_path = os.path.join(_CACHE_DIR, f"momentum_price_{label}.parquet")
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
