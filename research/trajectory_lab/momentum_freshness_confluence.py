"""
Trajectory Lab — "Momentum Freshness" × close_pos/VPMV/volatilite
CONFLUENCE keşfi (bkz. CONTEXT_LAB_STATUS.md, 8-9 Ağustos — hâlâ
Mechanism/Confluence sınırında, HENÜZ Strategy/threshold/production
DEĞİL).

Soru: momentum decay'in (momentum'un KENDİ zirvesinden S0'a düşüşü)
winner/loser ayırt ediciliği close_pos / VPMV / volatilite BAĞLAMINA
göre değişiyor mu? Üçlü çapraz bucket YOK — her bağlam değişkeni AYRI
AYRI test edilir.

decay_magnitude = Momentum_peak − Momentum_S0 (pozitif = erime).
peak=S0 olan sinyaller (decay YOK) SESSİZCE atılmıyor — no_decay/
decay_present oranı ayrıca raporlanıyor (kullanıcı notu, 9 Ağustos —
winner'ların %41-46'sında zaten hiç decay yok, bu bilgiyi kaybetmeyelim).

Bu script signal_id'yi cache'e YAZAR (önceki iki script'in — momentum_
price_order.py ve momentum_anchored_delta.py — sırayla eşleştirilemeyen
cache'lerinden FARKLI olarak) — gelecekte güvenli çapraz-referans için.

Kullanım:
    python -m research.trajectory_lab.momentum_freshness_confluence
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd

from database.engine import async_engine
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
_COMPONENTS = ["mom", "vol", "vlt", "prc"]
MIN_CELL_N = 100


def _build_sequence_frame(sequences: list[dict], close_pos_map: dict, comp_map: dict) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        sids = seq["window"]["signal_id"].tolist()
        if any(sid not in close_pos_map or sid not in comp_map for sid in sids):
            continue
        row = {"symbol": seq["symbol"], "outcome": seq["outcome"]}
        for i, sid in enumerate(sids):
            row[f"signal_id_{i}"] = sid
            row[f"cp_{i}"] = close_pos_map[sid]
            for c in _COMPONENTS:
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

    cp_df = await _fetch_close_pos_for_signals(all_signal_ids)
    close_pos_map = dict(zip(cp_df["signal_id"], cp_df["close_pos"]))
    comp_df = await _fetch_components(all_signal_ids)
    comp_map = comp_df.set_index("signal_id")[_COMPONENTS].to_dict("index")
    print(f"  close_pos: {len(close_pos_map)}/{len(all_signal_ids)}  bileşenler: {len(comp_map)}/{len(all_signal_ids)}")

    df = _build_sequence_frame(winner_sample + loser_sample, close_pos_map, comp_map)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  -> önbelleğe yazıldı: {cache_path} ({len(df)} dizi)")
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mom_cols = [f"mom_{i}" for i in range(4)]
    mom_arr = df[mom_cols].to_numpy()
    peak_idx = mom_arr.argmax(axis=1)
    peak_val = mom_arr[np.arange(len(df)), peak_idx]
    df["mom_peak_idx"] = peak_idx
    df["decay_magnitude"] = peak_val - df["mom_3"]  # Momentum_peak - Momentum_S0, pozitif=erime
    df["has_decay"] = peak_idx < 3
    df["close_pos_s0"] = df["cp_3"]
    df["vpmv_s0"] = 0.35 * df["vol_3"] + 0.35 * df["mom_3"] + 0.20 * df["vlt_3"] + 0.10 * df["prc_3"]
    df["volatility_s0"] = df["vlt_3"]  # vlt zaten volatility_pct ile AYNI tanım
    df["y"] = (df["outcome"] == "winner").astype(int)
    return df


def _cohens_d(w: pd.Series, l: pd.Series) -> float:
    pooled_std = np.sqrt(((len(w) - 1) * w.var(ddof=1) + (len(l) - 1) * l.var(ddof=1)) / (len(w) + len(l) - 2))
    return (w.mean() - l.mean()) / pooled_std if pooled_std > 0 else np.nan


def _report_bucket(name: str, sub: pd.DataFrame) -> None:
    n = len(sub)
    if n < MIN_CELL_N:
        print(f"    {name:8} n={n:6} — yetersiz örneklem, atlandı")
        return
    no_decay_w = sub[(sub["outcome"] == "winner") & (~sub["has_decay"])]
    no_decay_l = sub[(sub["outcome"] == "loser") & (~sub["has_decay"])]
    w_n = (sub["outcome"] == "winner").sum()
    l_n = (sub["outcome"] == "loser").sum()
    print(
        f"    {name:8} n={n:6}  no_decay: winner=%{len(no_decay_w)/w_n*100:.1f}  "
        f"loser=%{len(no_decay_l)/l_n*100:.1f}"
    )
    decayed = sub[sub["has_decay"]]
    w = decayed.loc[decayed["outcome"] == "winner", "decay_magnitude"]
    l = decayed.loc[decayed["outcome"] == "loser", "decay_magnitude"]
    if len(w) < 30 or len(l) < 30:
        print(f"    {'':8} decay_present alt-kümesi yetersiz (winner n={len(w)}, loser n={len(l)})")
        return
    d = _cohens_d(w, l)
    print(
        f"    {'':8} decay_present: winner n={len(w)} ort={w.mean():.2f}  "
        f"loser n={len(l)} ort={l.mean():.2f}  Cohen's d={d:+.3f}"
    )


def _analyze(label: str, df: pd.DataFrame) -> None:
    df = _derive(df)
    print(f"\n===== {label} — decay_magnitude × BAĞLAM (n={len(df)}) =====")

    for ctx_name, ctx_col, ctx_kind in [
        ("close_pos", "close_pos_s0", "tertile"),
        ("VPMV", "vpmv_s0", "tertile"),
        ("volatilite", "volatility_s0", "regime"),
    ]:
        print(f"\n  --- Bağlam: {ctx_name} ---")
        if ctx_kind == "tertile":
            df["_bucket"] = pd.qcut(df[ctx_col], 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")
            order = ["LOW", "MID", "HIGH"]
        else:
            df["_bucket"] = df[ctx_col].apply(lambda v: "HIGH" if v > 70 else ("LOW" if v < 30 else "NORMAL"))
            order = ["LOW", "NORMAL", "HIGH"]
        for bucket in order:
            _report_bucket(bucket, df[df["_bucket"] == bucket])


async def run() -> None:
    all_frames = []
    for label, indicator, direction in _FAMILIES:
        cache_path = os.path.join(_CACHE_DIR, f"momentum_freshness_{label}.parquet")
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
