"""
RSI_Cross sinyallerinde `signals.candle_pattern`'daki BELTHOLD formasyonunu
(pandas_ta_classic::cdl_pattern, sinyal anında CANLI hesaplanıp saklanmış —
look-ahead riski yok) test eder — en büyük örneklemli tekil formasyon
(+BELTHOLD n=1589, -BELTHOLD n=1562, ikisi de ham ortalamada pozitif).

belthold_confirm: BELTHOLD'un YÖNÜ sinyalin yönüyle eşleşiyor mu (+BELTHOLD
+ Long, -BELTHOLD + Short = "teyit eden kararlı mum") — FVG'nin "sinyal
yönünde" tanımıyla aynı mantık, ham "BELTHOLD var mı" değil.

threshold_optimizer'ın 3-kapılı disipliniyle (IS/OOS+split-period+placebo).
"""
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position

INDICATOR = "RSI_Cross(9,24)"


def _fetch_signals_with_pattern(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT opened_at, realized_pnl, candle_pattern
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL AND candle_pattern IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()

    want_sign = "+" if direction == "Long" else "-"
    df["belthold_confirm"] = df["candle_pattern"].str.contains(f"\\{want_sign}BELTHOLD", regex=True).astype(float)
    return df


def run() -> None:
    for direction in ("Long", "Short"):
        df = _fetch_signals_with_pattern(INDICATOR, direction)
        if len(df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(df)}), atlanıyor")
            continue
        print(f"{INDICATOR} — {direction}: {len(df):,} sinyal (belthold_confirm oranı={df['belthold_confirm'].mean():.2%})")

        label = f"{INDICATOR} — {direction} — belthold_confirm (yön-eşleşen BELTHOLD)"
        _run_single_var_on_df(label, df, "belthold_confirm")


if __name__ == "__main__":
    run()
