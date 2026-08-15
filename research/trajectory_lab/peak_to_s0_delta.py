"""
Trajectory Lab — sinyal-zamanı dizilerinde (S-3→S0) close_pos'un ZİRVE
yaptığı andan S0'a kadar VPMV/CVD Level'ın ne yaptığını ölçer (bkz.
CONTEXT_LAB_STATUS.md, 8 Ağustos — "loser-side erken-zirve/bozulma"
mekanizmasının sayısallaştırılması).

Metodoloji (kullanıcı, 8 Ağustos — HENÜZ threshold/bucket/ΔAUC/production
kuralı YOK, sadece ham dağılım):
  1. Her dizide (S-3,S-2,S-1,S0) close_pos'un EN YÜKSEK olduğu index =
     "zirve" (S0'ın kendisi olabilir — bu durumda Δ=0, ayrıca sayılır).
  2. Δclose_pos = close_pos(S0) - close_pos(zirve)
     ΔVPMV       = VPMV(S0)       - VPMV(zirve_index'teki_VPMV)
     ΔCVD        = CVD(S0)        - CVD(zirve_index'teki_CVD)
     (yön önemli, mutlak büyüklük değil — HENÜZ eşik yok)
  3. HEM winner HEM loser dizilerinde (asıl karşılaştırma budur — sadece
     loser'a bakmak yanıltıcı olabilir, kullanıcı notu).
  4. Sadece dağılım (ortalama/medyan/std/persentiller) + Δclose_pos ile
     ΔVPMV/ΔCVD arasındaki ham korelasyon — "VPMV birlikte mi hareket
     ediyor yoksa bağımsız mı" sorusuna kaba bir ilk cevap, bucket/
     kural DEĞİL.

Kullanım:
    python -m research.trajectory_lab.peak_to_s0_delta
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from research.trajectory_lab.behavior_scan import _select_balanced
from research.trajectory_lab.signal_sequence_scan import (
    _build_sequences,
    _fetch_close_pos_for_signals,
    _load_snapshots,
)
from database.engine import async_engine  # noqa: E402

_FAMILIES = [
    ("HA_Cross_Long", "HA_Cross", "Long"),
    ("RSI_Cross_Long", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long", "Supertrend(10,3.0)", "Long"),
]
N_PER_GROUP = 3000


def _sample_sequences(sequences: list[dict], n: int) -> list[dict]:
    if len(sequences) <= n:
        return sequences
    meta = pd.DataFrame(
        [{"symbol": s["symbol"], "opened_at": s["opened_at"], "seq_idx": i} for i, s in enumerate(sequences)]
    )
    sel = _select_balanced(meta, n)
    return [sequences[i] for i in sel["seq_idx"]]


def _compute_deltas(sequences: list[dict], close_pos_map: dict) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        window = seq["window"]
        cp_vals = [close_pos_map.get(sid, np.nan) for sid in window["signal_id"]]
        if any(pd.isna(v) for v in cp_vals):
            continue
        peak_idx = int(np.argmax(cp_vals))
        vpmv_vals = window["vpmv"].tolist()
        cvd_vals = window["cvd_level"].tolist()
        rows.append(
            {
                "symbol": seq["symbol"],
                "outcome": seq["outcome"],
                "peak_idx": peak_idx,
                "peak_is_s0": peak_idx == 3,
                "d_close_pos": cp_vals[3] - cp_vals[peak_idx],
                "d_vpmv": vpmv_vals[3] - vpmv_vals[peak_idx],
                "d_cvd": cvd_vals[3] - cvd_vals[peak_idx],
            }
        )
    return pd.DataFrame(rows)


def _print_group_stats(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"  [{label}] veri yok")
        return
    n = len(df)
    peak_s0_pct = df["peak_is_s0"].mean() * 100
    print(f"  [{label}] n={n} (zirve=S0 olan: %{peak_s0_pct:.1f})")
    for col, name in [("d_close_pos", "Δclose_pos"), ("d_vpmv", "ΔVPMV"), ("d_cvd", "ΔCVD")]:
        s = df[col]
        print(
            f"    {name:12} ortalama={s.mean():+7.2f}  medyan={s.median():+7.2f}  "
            f"std={s.std():6.2f}  p25={s.quantile(0.25):+7.2f}  p75={s.quantile(0.75):+7.2f}"
        )
    r_vpmv = df["d_close_pos"].corr(df["d_vpmv"])
    r_cvd = df["d_close_pos"].corr(df["d_cvd"])
    print(f"    corr(Δclose_pos, ΔVPMV) = {r_vpmv:+.3f}   corr(Δclose_pos, ΔCVD) = {r_cvd:+.3f}")
    # kaba yön dağılımı (henüz threshold/bucket DEĞİL — sadece işaret sayımı)
    both_down = ((df["d_close_pos"] < 0) & (df["d_vpmv"] < 0)).mean() * 100
    cp_down_vpmv_flat_up = ((df["d_close_pos"] < 0) & (df["d_vpmv"] >= 0)).mean() * 100
    print(
        f"    (yalnızca işaret sayımı, threshold değil) "
        f"close_pos↓+VPMV↓: %{both_down:.1f}   close_pos↓+VPMV≥0: %{cp_down_vpmv_flat_up:.1f}"
    )


async def run() -> None:
    all_rows = []
    for label, indicator, direction in _FAMILIES:
        snapshots = _load_snapshots(indicator, direction)
        sequences = _build_sequences(snapshots)
        winner_seqs = [s for s in sequences if s["outcome"] == "winner"]
        loser_seqs = [s for s in sequences if s["outcome"] == "loser"]

        winner_sample = _sample_sequences(winner_seqs, N_PER_GROUP)
        loser_sample = _sample_sequences(loser_seqs, N_PER_GROUP)
        print(
            f"\n{label}: uygun winner={len(winner_seqs)} (örneklenen {len(winner_sample)}), "
            f"loser={len(loser_seqs)} (örneklenen {len(loser_sample)})"
        )

        all_signal_ids = set()
        for seq in winner_sample + loser_sample:
            all_signal_ids.update(seq["window"]["signal_id"].tolist())
        print(f"  close_pos hesaplanacak hedefli sinyal sayısı: {len(all_signal_ids)}")
        cp_df = await _fetch_close_pos_for_signals(list(all_signal_ids))
        close_pos_map = dict(zip(cp_df["signal_id"], cp_df["close_pos"]))
        print(f"  close_pos hesaplanabilen: {len(close_pos_map)}/{len(all_signal_ids)}")

        w_df = _compute_deltas(winner_sample, close_pos_map)
        l_df = _compute_deltas(loser_sample, close_pos_map)
        w_df["source"] = label
        l_df["source"] = label
        all_rows.extend([w_df, l_df])

        print(f"\n===== {label} — zirve→S0 Δ dağılımı =====")
        _print_group_stats("WINNER", w_df)
        _print_group_stats("LOSER", l_df)

    pooled = pd.concat(all_rows, ignore_index=True)
    print("\n\n===== POOLED (3 aile) =====")
    _print_group_stats("WINNER", pooled[pooled["outcome"] == "winner"])
    _print_group_stats("LOSER", pooled[pooled["outcome"] == "loser"])


async def _main() -> None:
    try:
        await run()
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
