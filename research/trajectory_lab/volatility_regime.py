"""
Trajectory Lab — close_pos'un winner/loser ayırt ediciliği VOLATİLİTE
REJİMİNE göre değişiyor mu? (bkz. CONTEXT_LAB_STATUS.md, 7 Ağustos)

Rejim tanımı close_pos'tan TAMAMEN BAĞIMSIZ — production'ın kendi
tanımı (`metrics.py::volatility_pct_series`, `signal_processor.py::
volatility_regime`'in birebir kopyası): ATR(14)'ün 200-bar rolling
percentile-rank'i, production'ın kendi eşikleriyle (>70 high / <30 low /
normal) 3 rejime ayrılır. YENİ formül/eşik İCAT EDİLMEDİ.

Bu metrik mevcut corpus'ta yok (200-bar geçmiş gerektiriyor, corpus
penceresi trim edilmiş) — sinyal barının ÖNCESİNDEKİ ~220 barı kendi
interval'inde (5m/15m/1h/4h, cagg_X view'larından) ayrıca çekilir.

Kullanım:
    python -m research.trajectory_lab.volatility_regime
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine, get_session  # noqa: E402
from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab import metrics as M  # noqa: E402
from research.trajectory_lab.corpus_builder import _fetch_case_signals  # noqa: E402

_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_BATCH_SIZE = 150
_HISTORY_BARS = C.WARMUP_BARS  # 220 — volatility_pct_series'in 200-bar rolling'i için

_FAMILIES = [
    ("HA_Cross_Long", "HA_Cross", "Long"),
    ("RSI_Cross_Long", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long", "Supertrend(10,3.0)", "Long"),
]
_ANATOMY_FILES = {
    "HA_Cross_Long": "research/trajectory_corpus/anatomy_HA_Cross_Long.parquet",
    "RSI_Cross_Long": "research/trajectory_corpus/anatomy_RSI_Cross_9_24_Long.parquet",
    "Supertrend_Long": "research/trajectory_corpus/anatomy_Supertrend_10_3.0_Long.parquet",
}
_CACHE_DIR = "research/trajectory_corpus"


async def _fetch_history_batch(batch: pd.DataFrame, interval: str) -> pd.DataFrame:
    view = _CAGG[interval]
    iv_min = _INTERVAL_MINUTES[interval]
    starts = [row["opened_at"] - pd.Timedelta(minutes=iv_min * _HISTORY_BARS) for _, row in batch.iterrows()]
    ends = batch["opened_at"].tolist()
    async with get_session() as session:
        result = await session.execute(
            text(
                f"""
                SELECT sw.signal_id, p.bucket, p.high, p.low, p.close
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
    return pd.DataFrame(rows, columns=["signal_id", "bucket", "high", "low", "close"])


async def _fetch_volatility_pct(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Her (interval-grubu, batch) için ayrı sorgu; her sinyal grubunda
    metrics.py::volatility_pct_series AYNEN çağrılır, t=0 (son bar) değeri
    alınır."""
    results = []
    for interval, iv_group in signals_df.groupby("interval"):
        if interval not in _CAGG:
            print(f"  ! bilinmeyen interval atlandı: {interval} ({len(iv_group)} sinyal)")
            continue
        n = len(iv_group)
        for i in range(0, n, _BATCH_SIZE):
            batch = iv_group.iloc[i : i + _BATCH_SIZE]
            hist = await _fetch_history_batch(batch, interval)
            for signal_id, grp in hist.groupby("signal_id"):
                if len(grp) < 50:  # anlamlı bir rolling-percentile için minimum
                    continue
                grp = grp.sort_values("bucket").reset_index(drop=True)
                vol_pct_series = M.volatility_pct_series(grp)
                results.append({"signal_id": signal_id, "volatility_pct": float(vol_pct_series.iloc[-1])})
            print(f"  ... {interval}: {min(i + _BATCH_SIZE, n)}/{n} sinyal işlendi", end="\r")
        print()
    return pd.DataFrame(results)


def _regime(v: float) -> str:
    if v > 70:
        return "HIGH"
    if v < 30:
        return "LOW"
    return "NORMAL"


def _print_regime_report(label: str, df: pd.DataFrame) -> None:
    print(f"\n===== close_pos × Volatilite Rejimi — {label} =====")
    print(f"Toplam sinyal (volatility_pct hesaplanabilen): {len(df)}")
    print(f"Rejim dağılımı: {df['regime'].value_counts().to_dict()}\n")

    for regime in ("LOW", "NORMAL", "HIGH"):
        sub = df[df["regime"] == regime]
        if len(sub) < 30 or sub["y"].nunique() < 2:
            print(f"  {regime:8} n={len(sub):6} — yetersiz örneklem, atlandı")
            continue
        w = sub.loc[sub["y"] == 1, "close_pos"]
        l = sub.loc[sub["y"] == 0, "close_pos"]
        pooled_std = np.sqrt(
            ((len(w) - 1) * w.var(ddof=1) + (len(l) - 1) * l.var(ddof=1)) / (len(w) + len(l) - 2)
        )
        d = (w.mean() - l.mean()) / pooled_std if pooled_std > 0 else np.nan
        r = sub["close_pos"].corr(sub["y"])
        print(
            f"  {regime:8} n={len(sub):6} (winner={len(w)}, loser={len(l)}): "
            f"corr(close_pos,y)={r:+.3f}  Cohen's d={d:+.3f}  "
            f"ort.close_pos(winner)={w.mean():.1f}  ort.close_pos(loser)={l.mean():.1f}"
        )
    print()


async def run() -> None:
    all_frames = []
    for label, indicator, direction in _FAMILIES:
        cache_path = os.path.join(_CACHE_DIR, f"volpct_{label}.parquet")
        if os.path.exists(cache_path):
            merged = pd.read_parquet(cache_path)
            print(f"\n{label}: önbellekten yüklendi ({cache_path}, {len(merged)} sinyal)")
        else:
            signals_df = await _fetch_case_signals(indicator, direction)
            signals_df = signals_df[signals_df["outcome"].isin(["winner", "loser"])]
            signals_df = signals_df[signals_df["interval"] != "1m"].reset_index(drop=True)
            print(f"\n{label}: {len(signals_df)} kapanmış winner/loser sinyali (1m hariç)")

            vol_df = await _fetch_volatility_pct(signals_df)
            anatomy = pd.read_parquet(_ANATOMY_FILES[label])[["signal_id", "close_pos", "y"]]
            merged = anatomy.merge(vol_df, on="signal_id", how="inner")
            merged["regime"] = merged["volatility_pct"].apply(_regime)
            merged["source"] = label

            os.makedirs(_CACHE_DIR, exist_ok=True)
            merged.to_parquet(cache_path, index=False)
            print(f"  -> önbelleğe yazıldı: {cache_path}")

        _print_regime_report(label, merged)
        all_frames.append(merged)

    pooled = pd.concat(all_frames, ignore_index=True)
    _print_regime_report("POOLED (3 aile)", pooled)


async def _main() -> None:
    try:
        await run()
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
