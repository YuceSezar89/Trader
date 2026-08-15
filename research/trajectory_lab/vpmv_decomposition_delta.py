"""
Trajectory Lab — VPMV DECOMPOSITION, Faz 1a (bkz. CONTEXT_LAB_STATUS.md,
8 Ağustos — Mechanism aşaması, henüz Strategy/threshold/production DEĞİL).

Soru: close_pos zirvesinden S0'a kadar ΔVPMV (loser≈-12.5, winner≈-0.7,
bkz. peak_to_s0_delta.py) HANGİ bileşenden geliyor? VPMV'nin AYNI iç
formülü (`metrics.py::vpmv_components_series`, YENİ formül YOK — sadece
0.35·vol+0.35·mom+0.20·vlt+0.10·prc'ye indirgemeden ÖNCEKİ 4 bileşen
açığa çıkarılıyor) kullanılarak momentum/volume/volatility/price
bileşenlerinin AYNI zirve→S0 metodolojisiyle (peak_to_s0_delta.py ile
BİREBİR aynı) delta'sı ölçülüyor.

KESİN SINIR (kullanıcı, 8 Ağustos): ağırlıklardan (0.35/0.35/0.20/0.10)
yola çıkıp "en çok katkı yapan bileşen" sonucuna GİDİLMİYOR — sadece
"hangisi winner/loser'da farklı davranıyor" sorusuna bakılıyor. Threshold/
Strategy/Production YOK, bu HÂLÂ Mechanism araştırması.

Kullanım:
    python -m research.trajectory_lab.vpmv_decomposition_delta
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.engine import async_engine, get_session
from research.trajectory_lab import config as C
from research.trajectory_lab import metrics as M
from research.trajectory_lab.behavior_scan import _select_balanced
from research.trajectory_lab.signal_sequence_scan import (
    _build_sequences,
    _fetch_close_pos_for_signals,
    _load_snapshots,
)

_FAMILIES = [
    ("HA_Cross_Long", "HA_Cross", "Long"),
    ("RSI_Cross_Long", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long", "Supertrend(10,3.0)", "Long"),
]
N_PER_GROUP = 3000
_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
_HISTORY_BARS = C.WARMUP_BARS  # 220 — vpmv bileşenlerinin rolling(200) ihtiyacı
_BATCH_SIZE = 150
_COMPONENTS = ["mom", "vol", "vlt", "prc"]


def _sample_sequences(sequences: list[dict], n: int) -> list[dict]:
    if len(sequences) <= n:
        return sequences
    meta = pd.DataFrame(
        [{"symbol": s["symbol"], "opened_at": s["opened_at"], "seq_idx": i} for i, s in enumerate(sequences)]
    )
    sel = _select_balanced(meta, n)
    return [sequences[i] for i in sel["seq_idx"]]


async def _fetch_signal_meta(signal_ids: list) -> pd.DataFrame:
    async with get_session() as session:
        result = await session.execute(
            text("SELECT id, symbol, interval, opened_at FROM signals WHERE id = ANY(:ids)"),
            {"ids": list(signal_ids)},
        )
        rows = result.all()
    meta = pd.DataFrame(rows, columns=["signal_id", "symbol", "interval", "opened_at"])
    return meta[meta["interval"] != "1m"].reset_index(drop=True)


async def _fetch_history_batch(batch: pd.DataFrame, interval: str) -> pd.DataFrame:
    view = _CAGG[interval]
    iv_min = _INTERVAL_MINUTES[interval]
    starts = [ts - pd.Timedelta(minutes=iv_min * _HISTORY_BARS) for ts in batch["opened_at"]]
    ends = batch["opened_at"].tolist()
    async with get_session() as session:
        result = await session.execute(
            text(
                f"""
                SELECT sw.signal_id, p.bucket, p.open, p.high, p.low, p.close,
                       p.volume, p.buy_volume, p.sell_volume
                FROM unnest(
                    CAST(:signal_ids AS int[]), CAST(:symbols AS text[]),
                    CAST(:starts AS timestamp[]), CAST(:ends AS timestamp[])
                ) AS sw(signal_id, symbol, start_ts, end_ts)
                JOIN {view} p
                  ON p.symbol = sw.symbol AND p.bucket >= sw.start_ts AND p.bucket <= sw.end_ts
                ORDER BY sw.signal_id, p.bucket
                """
            ),
            {
                "signal_ids": batch["signal_id"].tolist(),
                "symbols": batch["symbol"].tolist(),
                "starts": starts,
                "ends": ends,
            },
        )
        rows = result.all()
    return pd.DataFrame(
        rows, columns=["signal_id", "bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    )


async def _fetch_components(signal_ids: list) -> pd.DataFrame:
    meta = await _fetch_signal_meta(signal_ids)
    results = []
    for interval, grp in meta.groupby("interval"):
        if interval not in _CAGG:
            continue
        n = len(grp)
        for i in range(0, n, _BATCH_SIZE):
            batch = grp.iloc[i : i + _BATCH_SIZE]
            hist = await _fetch_history_batch(batch, interval)
            for signal_id, g in hist.groupby("signal_id"):
                if len(g) < 50:
                    continue
                g = g.sort_values("bucket").reset_index(drop=True)
                comp = M.vpmv_components_series(g, "Long")
                last = comp.iloc[-1]
                results.append(
                    {"signal_id": signal_id, "mom": last["mom"], "vol": last["vol"], "vlt": last["vlt"], "prc": last["prc"]}
                )
            print(f"  ... components {interval}: {min(i + _BATCH_SIZE, n)}/{n} işlendi", end="\r")
        print()
    return pd.DataFrame(results)


def _compute_deltas(sequences: list[dict], close_pos_map: dict, comp_map: dict) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        window = seq["window"]
        sids = window["signal_id"].tolist()
        cp_vals = [close_pos_map.get(sid, np.nan) for sid in sids]
        if any(pd.isna(v) for v in cp_vals):
            continue
        if any(sid not in comp_map for sid in sids):
            continue
        peak_idx = int(np.argmax(cp_vals))
        row = {"symbol": seq["symbol"], "outcome": seq["outcome"], "peak_idx": peak_idx}
        for c in _COMPONENTS:
            vals = [comp_map[sid][c] for sid in sids]
            row[f"d_{c}"] = vals[3] - vals[peak_idx]
        row["d_close_pos"] = cp_vals[3] - cp_vals[peak_idx]
        rows.append(row)
    return pd.DataFrame(rows)


def _print_group_stats(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"  [{label}] veri yok")
        return
    print(f"  [{label}] n={len(df)}")
    for c in _COMPONENTS:
        col = f"d_{c}"
        s = df[col]
        r = df["d_close_pos"].corr(s)
        print(
            f"    Δ{c:4} ortalama={s.mean():+7.2f}  medyan={s.median():+7.2f}  std={s.std():6.2f}  "
            f"corr(Δclose_pos,Δ{c})={r:+.3f}"
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
        print(f"\n{label}: winner örneklenen={len(winner_sample)}, loser örneklenen={len(loser_sample)}")

        all_signal_ids = set()
        for seq in winner_sample + loser_sample:
            all_signal_ids.update(seq["window"]["signal_id"].tolist())
        all_signal_ids = list(all_signal_ids)
        print(f"  hedefli sinyal sayısı: {len(all_signal_ids)}")

        cp_df = await _fetch_close_pos_for_signals(all_signal_ids)
        close_pos_map = dict(zip(cp_df["signal_id"], cp_df["close_pos"]))
        print(f"  close_pos hesaplanabilen: {len(close_pos_map)}/{len(all_signal_ids)}")

        comp_df = await _fetch_components(all_signal_ids)
        comp_map = comp_df.set_index("signal_id")[_COMPONENTS].to_dict("index")
        print(f"  bileşenler hesaplanabilen: {len(comp_map)}/{len(all_signal_ids)}")

        w_df = _compute_deltas(winner_sample, close_pos_map, comp_map)
        l_df = _compute_deltas(loser_sample, close_pos_map, comp_map)
        w_df["source"] = label
        l_df["source"] = label
        all_rows.extend([w_df, l_df])

        print(f"\n===== {label} — bileşen zirve→S0 Δ =====")
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
