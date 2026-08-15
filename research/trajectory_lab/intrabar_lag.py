"""
Trajectory Lab — sinyal barının İÇİNDE (1m alt-mumlar), momentum ile hacmin
HANGİSİ ÖNCE geldiğini test eder. Gösterge YOK — ikisi de ham mumdan:

  momentum vekili = body_size (|close-open|), en büyük olduğu dakika
  hacim vekili     = volume, en büyük olduğu dakika

lag = hacim_dakikası - momentum_dakikası
  lag > 0 → en büyük hareket ÖNCE geldi, hacim SONRA
  lag < 0 → hacim ÖNCE patladı, en büyük hareket SONRA
  lag = 0 → aynı dakika

Winner/loser'da lag dağılımı karşılaştırılır (Mann-Whitney U).

Kullanım:
    python -m research.trajectory_lab.intrabar_lag --indicator HA_Cross --direction Long
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine, get_session  # noqa: E402
from research.trajectory_lab.corpus_builder import _fetch_case_signals  # noqa: E402

_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


_BATCH_SIZE = 300


async def _fetch_intrabar_batch(batch: pd.DataFrame) -> pd.DataFrame:
    starts = batch["opened_at"].tolist()
    ends = [
        row["opened_at"] + pd.Timedelta(minutes=_INTERVAL_MINUTES[row["interval"]])
        for _, row in batch.iterrows()
    ]
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT sw.signal_id, p.timestamp, p.open, p.high, p.low, p.close,
                       p.volume
                FROM unnest(
                    CAST(:signal_ids AS int[]), CAST(:symbols AS text[]),
                    CAST(:starts AS timestamp[]), CAST(:ends AS timestamp[])
                ) AS sw(signal_id, symbol, start_ts, end_ts)
                JOIN price_data p
                  ON p.symbol = sw.symbol AND p.interval = '1m'
                 AND p.timestamp >= sw.start_ts AND p.timestamp < sw.end_ts
                ORDER BY sw.signal_id, p.timestamp
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
        rows, columns=["signal_id", "timestamp", "open", "high", "low", "close", "volume"]
    )


async def _fetch_intrabar(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Her sinyalin kendi barını oluşturan 1m alt-mumları PARTİ PARTİ çeker
    (unnest ile sinyal pencereleri, price_data'yla join) — tek sorguda
    binlerce sinyal TimescaleDB'nin chunk-exclusion'ını bozup timeout'a
    yol açıyordu (351 chunk'lı hypertable)."""
    frames = []
    n = len(signals_df)
    for i in range(0, n, _BATCH_SIZE):
        batch = signals_df.iloc[i : i + _BATCH_SIZE]
        frames.append(await _fetch_intrabar_batch(batch))
        print(f"  ... {min(i + _BATCH_SIZE, n)}/{n} sinyal işlendi", end="\r")
    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_lags(signals_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = sub_df.copy()
    sub_df["body_size"] = (sub_df["close"] - sub_df["open"]).abs()

    out = []
    for signal_id, grp in sub_df.groupby("signal_id"):
        expected_n = _INTERVAL_MINUTES[
            signals_df.loc[signals_df["signal_id"] == signal_id, "interval"].iloc[0]
        ]
        if len(grp) < expected_n:
            continue  # eksik alt-mum (gap) — güvenilmez, atla
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        momentum_min = int(grp["body_size"].idxmax())
        volume_min = int(grp["volume"].idxmax())
        out.append(
            {
                "signal_id": signal_id,
                "n_subbars": len(grp),
                "momentum_min": momentum_min,
                "volume_min": volume_min,
                "lag": volume_min - momentum_min,
            }
        )
    return pd.DataFrame(out)


def _report(label: str, lag_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    merged = lag_df.merge(signals_df[["signal_id", "outcome"]], on="signal_id")
    merged = merged[merged["outcome"].isin(["winner", "loser"])]
    print(f"\n===== {label} — intrabar momentum-vs-hacim SIRASI =====")
    print(f"Toplam sinyal (lag hesaplanabilen): {len(merged)}")
    for outcome in ("winner", "loser"):
        sub = merged.loc[merged["outcome"] == outcome, "lag"]
        if sub.empty:
            continue
        print(
            f"  {outcome:7} n={len(sub):6} ortalama lag={sub.mean():+.3f} "
            f"medyan={sub.median():+.1f} std={sub.std():.3f} "
            f"(pozitif=momentum önce, negatif=hacim önce)"
        )
    w = merged.loc[merged["outcome"] == "winner", "lag"]
    l = merged.loc[merged["outcome"] == "loser", "lag"]
    if len(w) > 0 and len(l) > 0:
        u, p = sstats.mannwhitneyu(w, l, alternative="two-sided")
        print(f"  Mann-Whitney U p-değeri (winner vs loser lag farkı) = {p:.4f}")
    same_min = (merged["momentum_min"] == merged["volume_min"]).mean() * 100
    print(f"  Aynı dakikada çakışma oranı (lag=0): %{same_min:.1f}")


async def run(indicator: str, direction: str) -> None:
    signals_df = await _fetch_case_signals(indicator, direction)
    signals_df = signals_df[signals_df["outcome"].isin(["winner", "loser"])]
    n_1m = int((signals_df["interval"] == "1m").sum())
    signals_df = signals_df[signals_df["interval"] != "1m"].reset_index(drop=True)
    print(
        f"{indicator} {direction}: {len(signals_df)} kapanmış winner/loser sinyali "
        f"(1m interval'lı {n_1m} sinyal atlandı — alt-bara ayrılamaz)"
    )

    sub_df = await _fetch_intrabar(signals_df)
    lag_df = _compute_lags(signals_df, sub_df)
    _report(f"{indicator} {direction}", lag_df, signals_df)


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default="HA_Cross")
    parser.add_argument("--direction", default="Long", choices=["Long", "Short"])
    args = parser.parse_args()
    try:
        await run(args.indicator, args.direction)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
