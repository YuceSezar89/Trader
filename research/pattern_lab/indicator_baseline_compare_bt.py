"""
ML vizyonunun 2. adımı (project_ml_vision.md): "RSI/MA200/ST arasında basit
istatistik — hangisi daha iyi t5-t10 getirisi üretiyor?" sorusuna cevap.
signal_performance/signals tablosu 3.5 haftadır doluyor (50k+ satır, 11 Tem
kontrolü) — artık cevaplanabilir. 4 sinyal ailesi × Long/Short baseline
PF/WR/n karşılaştırması, kronolojik ilk/ikinci yarı ile sağlamlık notu
(placebo/OOS değil — bu bir eşik ARAMASI değil, ham baseline karşılaştırması).
"""
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross", "Supertrend(10,3.0)"]
DIRECTIONS = ["Long", "Short"]


def _fetch(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT opened_at, realized_pnl FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    return df


def run() -> None:
    rows = []
    for ind in INDICATORS:
        for d in DIRECTIONS:
            df = _fetch(ind, d)
            if len(df) == 0:
                continue
            rets = df["realized_pnl"].to_numpy() / 100
            s = _stats(rets)

            mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
            first = df[df["opened_at"] < mid]["realized_pnl"].to_numpy() / 100
            second = df[df["opened_at"] >= mid]["realized_pnl"].to_numpy() / 100
            s1, s2 = _stats(first), _stats(second)

            rows.append({
                "indikator": ind, "yön": d, "n": s["n"], "WR%": s["wr"],
                "ort%": s["ort_%"], "PF": s["pf"],
                "PF_ilk_yari": s1.get("pf", 0), "PF_ikinci_yari": s2.get("pf", 0),
            })

    out = pd.DataFrame(rows).sort_values("PF", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
