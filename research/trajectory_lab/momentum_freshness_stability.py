"""
Trajectory Lab — Momentum Freshness (decay_magnitude) ZAMAN-İÇİ
KARARLILIK testi (bkz. CONTEXT_LAB_STATUS.md, 9 Ağustos — Mechanism →
Temporal Validation geçişi, HENÜZ Confluence/Strategy/Production DEĞİL).

Soru: "Momentum decay'in winner/loser ayırt ediciliği (Cohen's d) zaman
içinde İSTİKRARLI mı, yoksa birkaç büyük dönem tarafından mı taşınıyor?"

decay_magnitude tanımı DEĞİŞTİRİLMEDİ (Momentum_peak - Momentum_S0).
YENİ formül/threshold YOK — sadece body_size_pct/close_pos'un zaman-içi
stability testleriyle AYNI disiplin (expanding window, her dilimde
Cohen's d + bootstrap CI + n), bu kez decay_magnitude için.

Veri: momentum_freshness_*.parquet cache'i (signal_id dahil) + sekanslar
YENİDEN ÜRETİLİP (deterministik, DB'ye gitmeden) t0 zaman damgası
signal_id üzerinden geri bağlanıyor — YENİ FETCH YOK.

Kullanım:
    python -m research.trajectory_lab.momentum_freshness_stability
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from database.engine import async_engine
from research.trajectory_lab.signal_sequence_scan import _build_sequences, _load_snapshots
from research.trajectory_lab.vpmv_decomposition_delta import N_PER_GROUP, _sample_sequences

_FAMILIES = [
    ("HA_Cross_Long", "HA_Cross", "Long"),
    ("RSI_Cross_Long", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long", "Supertrend(10,3.0)", "Long"),
]
_CACHE_DIR = "research/trajectory_corpus"
N_FOLDS = 8
N_BOOTSTRAP = 300
RNG_SEED = 42
MIN_CELL_N = 30


def _rebuild_t0_map(indicator: str, direction: str) -> dict:
    """Sekansları DB'siz, deterministik olarak yeniden üretip S0'ın
    signal_id -> t0 eşleşmesini çıkarır (cache'te t0 yoktu)."""
    snapshots = _load_snapshots(indicator, direction)
    sequences = _build_sequences(snapshots)
    winner_seqs = [s for s in sequences if s["outcome"] == "winner"]
    loser_seqs = [s for s in sequences if s["outcome"] == "loser"]
    winner_sample = _sample_sequences(winner_seqs, N_PER_GROUP)
    loser_sample = _sample_sequences(loser_seqs, N_PER_GROUP)
    t0_map = {}
    for seq in winner_sample + loser_sample:
        s0_id = seq["window"]["signal_id"].iloc[-1]
        t0_map[s0_id] = seq["opened_at"]
    return t0_map


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mom_cols = [f"mom_{i}" for i in range(4)]
    mom_arr = df[mom_cols].to_numpy()
    peak_idx = mom_arr.argmax(axis=1)
    peak_val = mom_arr[np.arange(len(df)), peak_idx]
    df["decay_magnitude"] = peak_val - df["mom_3"]
    df["has_decay"] = peak_idx < 3
    return df


def _cohens_d(w: np.ndarray, l: np.ndarray) -> float:
    ps = np.sqrt(((len(w) - 1) * w.var(ddof=1) + (len(l) - 1) * l.var(ddof=1)) / (len(w) + len(l) - 2))
    return (w.mean() - l.mean()) / ps if ps > 0 else np.nan


def _bootstrap_d_ci(w: np.ndarray, l: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple:
    ds = np.empty(n_boot)
    for i in range(n_boot):
        wb = rng.choice(w, len(w), replace=True)
        lb = rng.choice(l, len(l), replace=True)
        ds[i] = _cohens_d(wb, lb)
    return tuple(np.percentile(ds, [2.5, 97.5]))


def _analyze(label: str, df: pd.DataFrame) -> dict:
    df = _derive(df).sort_values("t0").reset_index(drop=True)
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)
    edges = np.linspace(0, n, N_FOLDS + 1).astype(int)
    fold_slices = [df.iloc[edges[i] : edges[i + 1]] for i in range(N_FOLDS)]

    print(f"\n===== {label} — decay_magnitude ZAMAN-İÇİ KARARLILIK (n={n}, {N_FOLDS} dilim) =====")
    ds = []
    for i, sub in enumerate(fold_slices, start=1):
        decayed = sub[sub["has_decay"]]
        w = decayed.loc[decayed["outcome"] == "winner", "decay_magnitude"].to_numpy()
        l = decayed.loc[decayed["outcome"] == "loser", "decay_magnitude"].to_numpy()
        no_decay_w = (sub["outcome"] == "winner") & (~sub["has_decay"])
        no_decay_l = (sub["outcome"] == "loser") & (~sub["has_decay"])
        w_total = (sub["outcome"] == "winner").sum()
        l_total = (sub["outcome"] == "loser").sum()
        if len(w) < MIN_CELL_N or len(l) < MIN_CELL_N:
            print(f"  Dilim {i} [{sub['t0'].min()} -> {sub['t0'].max()}] — yetersiz örneklem, atlandı")
            continue
        d = _cohens_d(w, l)
        lo, hi = _bootstrap_d_ci(w, l, N_BOOTSTRAP, rng)
        ds.append(d)
        sig = "pozitif(ayırt edici)" if hi < 0 else ("belirsiz" if lo < 0 < hi else "TERS")
        print(
            f"  Dilim {i} [{sub['t0'].min()} -> {sub['t0'].max()}] n={len(sub)} "
            f"(decay_present: w={len(w)},l={len(l)}) "
            f"no_decay: w=%{no_decay_w.sum()/w_total*100:.1f} l=%{no_decay_l.sum()/l_total*100:.1f} "
            f"Cohen's d={d:+.3f} CI=[{lo:+.3f},{hi:+.3f}] [{sig}]"
        )

    ds = np.array(ds)
    n_valid = len(ds)
    n_same_dir = (ds < 0).sum()  # tutarlı yön: winner ortalaması loser'dan düşük -> d negatif
    print(
        f"\n  Özet: {n_valid} geçerli dilim, {n_same_dir}/{n_valid} aynı yönde (negatif d), "
        f"ortalama d={ds.mean():+.3f} std={ds.std():.3f} min={ds.min():+.3f} max={ds.max():+.3f}"
    )
    return {"label": label, "n_valid": n_valid, "n_same_dir": n_same_dir, "d_mean": ds.mean() if n_valid else np.nan}


async def run() -> None:
    summaries = []
    all_frames = []
    for label, indicator, direction in _FAMILIES:
        cache_path = f"{_CACHE_DIR}/momentum_freshness_{label}.parquet"
        df = pd.read_parquet(cache_path)
        t0_map = _rebuild_t0_map(indicator, direction)
        df["t0"] = df["signal_id_3"].map(t0_map)
        missing = df["t0"].isna().sum()
        if missing:
            print(f"  ! {label}: {missing}/{len(df)} sinyalde t0 eşleşmedi (atlanıyor)")
            df = df.dropna(subset=["t0"])
        df["source"] = label
        all_frames.append(df)
        summaries.append(_analyze(label, df))

    pooled = pd.concat(all_frames, ignore_index=True)
    summaries.append(_analyze("POOLED (3 aile)", pooled))

    print("\n\n===== ÖZET TABLO =====")
    for s in summaries:
        print(f"  {s['label']:20} {s['n_same_dir']}/{s['n_valid']} dilim tutarlı yönde, ortalama d={s['d_mean']:+.3f}")


async def _main() -> None:
    try:
        await run()
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
