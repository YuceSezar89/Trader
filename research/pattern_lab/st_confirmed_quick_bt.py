"""
st_confirmed hizli testi — eski st_confirmed_filtered_bt.py'nin (rsi_cross_vpmv_jump_bt
importunda sembol-basina yavas sorgu donguisu barindiran) temiz/hizli yeniden
yazimi. st_confirmed zaten DB'de hazir bir boolean (sinyalin KENDI interval'inde
Supertrend yonu sinyalle uyusuyor mu, signal_processor.py:503-511) — ek
hesaplama/resample gerekmiyor, tek sorguyla test edilebilir.

Kullanım: python -m research.pattern_lab.st_confirmed_quick_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

from config import Config


def _fetch(indicator: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT signal_type, interval, opened_at, st_confirmed, realized_pnl
        FROM signals
        WHERE indicators = %s AND status = 'closed' AND realized_pnl IS NOT NULL
          AND st_confirmed IS NOT NULL
    """
    df = pd.read_sql(q, conn, params=(indicator,))
    conn.close()
    return df


def _report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        conf = sub[sub["st_confirmed"]]
        noconf = sub[~sub["st_confirmed"]]
        print(
            f"  {sig_type}: onaylı n={len(conf)} ort_pnl={conf['realized_pnl'].mean():+.4f} "
            f"wr={(conf['realized_pnl']>0).mean()*100:.1f}%  |  "
            f"onaysız n={len(noconf)} ort_pnl={noconf['realized_pnl'].mean():+.4f} "
            f"wr={(noconf['realized_pnl']>0).mean()*100:.1f}%"
        )
        if len(conf) >= 10 and len(noconf) >= 10:
            u, p = mannwhitneyu(conf["realized_pnl"], noconf["realized_pnl"], alternative="two-sided")
            print(f"    Mann-Whitney p-value: {p:.4f}")


def _report_split(df: pd.DataFrame) -> None:
    mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", df[df["opened_at"] < mid]),
        ("ikinci_yari", df[df["opened_at"] >= mid]),
    ]:
        _report(half_df, f"SPLIT-PERIOD {half_name} ({len(half_df)} sinyal)")


def main() -> None:
    for indicator in ["HA_Cross", "RSI_Cross(9,24)"]:
        print(f"\n{'='*60}\n{indicator}\n{'='*60}")
        df = _fetch(indicator)
        print(f"Toplam sinyal: {len(df)}")
        _report(df, "ANA TEST (tüm dönem)")
        _report_split(df)


if __name__ == "__main__":
    main()
