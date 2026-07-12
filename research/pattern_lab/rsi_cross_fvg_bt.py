"""
RSI_Cross sinyallerinde `signals.fvg_tfs` alanını (sinyal anında CANLI hesaplanıp
saklanmış — signal_processor.py::_compute_fvg, 6 TF'de (1m-1d) sinyal yönünde
aktif FVG var mı, look-ahead riski yok) filtre/rejim olarak test eder.

Baseline gözlem çarpıcı: Long'da FVG YOKsa ort. PnL NEGATİF (-0.094), VARsa
+0.498 — bugünün en büyük ham ayrımı. threshold_optimizer'ın 3-kapılı
disipliniyle (IS/OOS+split-period+placebo) doğrulanıyor.

fvg_var: fvg_tfs != '-' (sinyal yönünde en az bir TF'de aktif FVG var) → 1, yoksa 0.
"""

import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.threshold_optimizer import (
    _run_single_var_on_df,  # pylint: disable=wrong-import-position
)

INDICATOR = "RSI_Cross(9,24)"


def _fetch_signals_with_fvg(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT opened_at, realized_pnl, fvg_tfs
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL AND fvg_tfs IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    df["fvg_var"] = (df["fvg_tfs"] != "-").astype(float)
    return df


def run() -> None:
    for direction in ("Long", "Short"):
        df = _fetch_signals_with_fvg(INDICATOR, direction)
        if len(df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(df)}), atlanıyor")
            continue
        print(
            f"{INDICATOR} — {direction}: {len(df):,} sinyal (fvg_var oranı={df['fvg_var'].mean():.2%})"
        )

        label = f"{INDICATOR} — {direction} — fvg_var (sinyal anında aktif FVG)"
        _run_single_var_on_df(label, df, "fvg_var")


if __name__ == "__main__":
    run()
