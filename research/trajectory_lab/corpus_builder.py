"""
Trajectory Lab — Adım 1: gerçek kapanmış sinyaller için trajectory corpus'u
üretir.

Sinyal seçimi pattern_lab/threshold_optimizer.py::_fetch_signals ile AYNI
desen (indicators+signal_type, status='closed'). Outcome etiketi
signal_performance.return_t5_pct'ten (Pattern Lab ile AYNI temiz hedef).
Kline penceresi price_data (1m) / cagg_X view'larından (5m/15m/1h/4h/6h/8h/12h)
— database/crud.py'nin KENDİSİ import edilmiyor (no production dependency
ilkesi, bkz. README), ama şema/view adları aynı (price_data, cagg_5m vb.).

Kullanım:
    python -m research.trajectory_lab.corpus_builder --indicator "HA_Cross" --direction Long
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import timedelta

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine, get_session  # noqa: E402
from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab import metrics as M  # noqa: E402

_CAGG_MAP = {
    "5m": "cagg_5m",
    "15m": "cagg_15m",
    "1h": "cagg_1h",
    "4h": "cagg_4h",
    "6h": "cagg_6h",
    "8h": "cagg_8h",
    "12h": "cagg_12h",
}
_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


async def _fetch_case_signals(indicator: str, direction: str) -> pd.DataFrame:
    """signals JOIN signal_performance — pattern_lab/threshold_optimizer.py::
    _fetch_signals ile aynı filtre (status='closed'), outcome Pattern Lab'ın
    kendi temiz hedefinden (return_t5_pct)."""
    async with get_session() as session:
        result = await session.execute(
            text(
                f"""
                SELECT s.id AS signal_id, s.symbol, s.interval, s.opened_at,
                       sp.{C.OUTCOME_METRIC} AS outcome_value
                FROM signals s
                JOIN signal_performance sp ON sp.signal_id = s.id
                WHERE s.indicators = :indicator
                  AND s.signal_type = :direction
                  AND s.status = 'closed'
                  AND sp.{C.OUTCOME_METRIC} IS NOT NULL
                ORDER BY s.opened_at
                """
            ),
            {"indicator": indicator, "direction": direction},
        )
        rows = result.all()
    df = pd.DataFrame(rows, columns=["signal_id", "symbol", "interval", "opened_at", "outcome_value"])

    def _label(v: float) -> str:
        if v >= C.WINNER_THRESHOLD:
            return "winner"
        if v <= C.LOSER_THRESHOLD:
            return "loser"
        return "neutral"

    df["outcome"] = df["outcome_value"].apply(_label)
    return df


async def _fetch_window_klines(symbol: str, interval: str, center: "pd.Timestamp") -> pd.DataFrame:
    """center (sinyalin opened_at'ı) etrafında [-WARMUP_BARS-WINDOW_PRE,
    +WINDOW_POST] barlık pencere. 1m -> price_data, diğerleri -> cagg_X view."""
    iv_min = _INTERVAL_MINUTES.get(interval, 5)
    start = center - timedelta(minutes=iv_min * (C.WARMUP_BARS + C.WINDOW_PRE))
    end = center + timedelta(minutes=iv_min * C.WINDOW_POST)

    if interval == "1m":
        query = """
            SELECT timestamp AS bucket, open, high, low, close, volume, buy_volume, sell_volume
            FROM price_data
            WHERE symbol = :symbol AND interval = :interval
              AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp
        """
        params = {"symbol": symbol, "interval": interval, "start": start, "end": end}
    else:
        view = _CAGG_MAP.get(interval)
        if view is None:
            return pd.DataFrame()
        query = f"""
            SELECT bucket, open, high, low, close, volume, buy_volume, sell_volume
            FROM {view}
            WHERE symbol = :symbol AND bucket BETWEEN :start AND :end
            ORDER BY bucket
        """
        params = {"symbol": symbol, "start": start, "end": end}

    async with get_session() as session:
        result = await session.execute(text(query), params)
        rows = result.all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows, columns=["bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    )
    df["t_offset"] = (
        (df["bucket"] - center).dt.total_seconds() / 60 / iv_min
    ).round().astype(int)
    return df


async def build_corpus(indicator: str, direction: str) -> pd.DataFrame:
    signals_df = await _fetch_case_signals(indicator, direction)
    active_metrics = [name for name, cfg in C.METRIC_PROVIDERS.items() if cfg["active"]]

    long_rows = []
    for _, sig in signals_df.iterrows():
        klines = await _fetch_window_klines(sig["symbol"], sig["interval"], sig["opened_at"])
        if klines.empty or len(klines) < C.WARMUP_BARS:
            continue
        window = klines[(klines["t_offset"] >= -C.WINDOW_PRE) & (klines["t_offset"] <= C.WINDOW_POST)]
        if window.empty:
            continue

        for metric_name in active_metrics:
            series = M.PROVIDERS[metric_name](klines, direction)
            trimmed = series.loc[window.index]
            for t_offset, value in zip(window["t_offset"], trimmed):
                if pd.isna(value):
                    continue
                long_rows.append(
                    {
                        "signal_id": sig["signal_id"],
                        "symbol": sig["symbol"],
                        "t0": sig["opened_at"],
                        "outcome": sig["outcome"],
                        "t_offset": int(t_offset),
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

    corpus = pd.DataFrame(long_rows)
    os.makedirs(C.CORPUS_DIR, exist_ok=True)
    out_path = os.path.join(C.CORPUS_DIR, f"{indicator.replace('(', '_').replace(')', '')}_{direction}.parquet")
    corpus.to_parquet(out_path, index=False)
    print(f"{len(signals_df)} sinyal, {len(corpus)} satır -> {out_path}")
    return corpus


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", required=True)
    parser.add_argument("--direction", required=True, choices=["Long", "Short"])
    args = parser.parse_args()
    try:
        await build_corpus(args.indicator, args.direction)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
