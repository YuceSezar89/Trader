"""
Devisso score backfill — cagg_5m / cagg_15m kullanarak.

29 Haz 2026 23:50 (commit 88af9e7) öncesi sinyaller ters formülle (ΔRSI/ΔPrice%)
hesaplanmıştı. Bu script hem NULL kayıtları hem de o tarihten önceki (dolu ama
yanlış formüllü) kayıtları düzeltilmiş formülle yeniden hesaplayıp üzerine yazar.

Kullanım:
    python scripts/backfill_devisso.py [--dry-run]
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from indicators.core import calculate_rsi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB = dict(host="127.0.0.1", port=6432, dbname="trader_panel", user="yusuf", password="123")

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m"}
_BARS_NEEDED = 130


def _compute_devisso(df: pd.DataFrame) -> float | None:
    try:
        if len(df) < 30:
            return None
        close = df["close"].astype(float)
        rsi = calculate_rsi(df, period=14)
        price_pct = close.pct_change() * 100.0
        raw = price_pct / rsi.diff().replace(0.0, np.nan)
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.dropna()
        if len(valid) < 20:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        rank = float((recent < current).sum()) / len(recent)
        return round(rank * 100.0, 2)
    except Exception:
        return None


def _fetch_bars(cur, symbol: str, interval: str, opened_at: datetime) -> pd.DataFrame | None:
    cagg = _CAGG.get(interval)
    if not cagg:
        return None
    cur.execute(f"""
        SELECT bucket AS open_time, open, high, low, close, volume
        FROM {cagg}
        WHERE symbol = %s AND bucket <= %s
        ORDER BY bucket DESC
        LIMIT %s
    """, (symbol, opened_at, _BARS_NEEDED))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)
    return df


def run(dry_run: bool = False) -> None:
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    upd = conn.cursor()

    cur.execute("""
        SELECT id, symbol, interval, opened_at, signal_type
        FROM signals
        WHERE (devisso_score IS NULL OR opened_at < '2026-06-29 23:50:35')
          AND interval IN ('5m', '15m')
        ORDER BY symbol, interval, opened_at
    """)
    signals = cur.fetchall()
    total = len(signals)
    logger.info("%d sinyal backfill gerekiyor", total)

    ok = skip = fail = 0

    for i, sig in enumerate(signals, 1):
        sid      = sig["id"]
        symbol   = sig["symbol"]
        interval = sig["interval"]
        opened_at = sig["opened_at"]

        df = _fetch_bars(cur, symbol, interval, opened_at)
        if df is None or len(df) < 30:
            skip += 1
            continue

        score = _compute_devisso(df)
        if score is None:
            skip += 1
            continue

        if not dry_run:
            upd.execute(
                "UPDATE signals SET devisso_score = %s WHERE id = %s",
                (score, sid)
            )

        ok += 1

        if i % 500 == 0:
            if not dry_run:
                conn.commit()
            logger.info("[%d/%d] güncellendi=%d  atlandı=%d  hata=%d", i, total, ok, skip, fail)

    if not dry_run:
        conn.commit()

    conn.close()
    logger.info("Tamamlandı — güncellendi=%d  atlandı=%d  hata=%d / toplam=%d", ok, skip, fail, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="DB'ye yazma, sadece hesapla")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
