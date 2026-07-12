"""
`st_confirmed` filtresi testi — panelin "sadece ST onaylı" checkbox'ının
gerçek ekonomik etkisi var mı diye bakar. st_confirmed, Supertrend'in
KENDİ sinyali için her zaman True (anlamsız) ama RSI_Cross/HA_Cross/
MA200_Cross için GERÇEK bir ayrım: sinyalin KENDİ zaman diliminde
(sinyalin ait olduğu interval, üst TF değil) Supertrend'in (`indicators/
core.py::calculate_supertrend` — TradingView/ChartPrime uyumlu, RMA
seed=SMA, standart ATR-Supertrend'in doğru başlatılmış hali) yönü sinyalle
uyuşuyor mu (`signal_processor.py:503-511`).

st_confirmed sabit bir boolean (aranacak/kalibre edilecek bir eşik yok),
bu yüzden threshold_optimizer'ın percentile arama/placebo aygıtına gerek
yok — doğrudan True/False karşılaştırması + split-period sağlamlık yeterli.
"""

import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross"]
DIRECTIONS = ["Long", "Short"]


def _fetch(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT st_confirmed, realized_pnl, opened_at
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
          AND st_confirmed IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


def _report(df: pd.DataFrame) -> None:
    confirmed = df[df["st_confirmed"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    unconfirmed = df[
        df["st_confirmed"] == False
    ]  # noqa: E712  pylint: disable=singleton-comparison

    baseline = _pf(df)
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(
        f"{'baseline (tümü)':20} {baseline.get('n',0):>7} {baseline.get('wr',0):>6} "
        f"{baseline.get('ort_%',0):>8} {baseline.get('pf',0):>7}"
    )
    s_c = _pf(confirmed)
    print(
        f"{'st_confirmed=True':20} {s_c.get('n',0):>7} {s_c.get('wr',0):>6} "
        f"{s_c.get('ort_%',0):>8} {s_c.get('pf',0):>7}"
    )
    s_u = _pf(unconfirmed)
    print(
        f"{'st_confirmed=False':20} {s_u.get('n',0):>7} {s_u.get('wr',0):>6} "
        f"{s_u.get('ort_%',0):>8} {s_u.get('pf',0):>7}"
    )

    if len(df) < 60:
        return
    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    first_c = confirmed[confirmed["opened_at"] < mid]
    second_c = confirmed[confirmed["opened_at"] >= mid]
    first_u = unconfirmed[unconfirmed["opened_at"] < mid]
    second_u = unconfirmed[unconfirmed["opened_at"] >= mid]
    print("\n-- split-period (st_confirmed=True) --")
    print(f"  ilk yarı:    PF={_pf(first_c).get('pf',0):>6} (n={_pf(first_c).get('n',0)})")
    print(f"  ikinci yarı: PF={_pf(second_c).get('pf',0):>6} (n={_pf(second_c).get('n',0)})")
    print("-- split-period (st_confirmed=False) --")
    print(f"  ilk yarı:    PF={_pf(first_u).get('pf',0):>6} (n={_pf(first_u).get('n',0)})")
    print(f"  ikinci yarı: PF={_pf(second_u).get('pf',0):>6} (n={_pf(second_u).get('n',0)})")


def run():
    for indicator in INDICATORS:
        for direction in DIRECTIONS:
            df = _fetch(indicator, direction)
            print(f"\n{'='*70}\n{indicator} — {direction}  (n={len(df):,})\n{'='*70}")
            if len(df) < 60:
                print("Örneklem çok küçük.")
                continue
            _report(df)


if __name__ == "__main__":
    run()
