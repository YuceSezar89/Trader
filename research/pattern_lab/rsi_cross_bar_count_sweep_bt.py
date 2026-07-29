"""
Bar sayısı taraması — 18 Tem 2026, kullanıcı "5 bar çok fazla" dedi (15m'de
75dk). Alt-TF'ye inmek işe yaramamıştı ([[rsi_cross_alttf_ranking_bt]]) — bu
script farklı bir soru soruyor: TF DEĞİŞTİRMEDEN, sinyalin KENDİ TF'sinde
daha AZ bar (1/2/3/4/5) beklemek, sıralama gücünün ne kadarını koruyor?

Kullanım: python -m research.pattern_lab.rsi_cross_bar_count_sweep_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_MIN_COHORT_SIZE = 2
_COHORT_KEY = ["signal_type", "interval", "opened_at"]
_BAR_COUNTS = [1, 2, 3, 4, 5]
_INTERVAL_MIN = {"5m": 5, "15m": 15}


def _fetch_interval(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    offset_exprs = []
    for n in _BAR_COUNTS:
        offset = n * _INTERVAL_MIN[interval] - 1
        offset_exprs.append(
            f"""(SELECT p.close FROM price_data p
                 WHERE p.symbol = s.symbol AND p.interval = '1m' AND p.timestamp > s.opened_at
                 ORDER BY p.timestamp ASC LIMIT 1 OFFSET {offset}) AS price_{n}bar"""
        )
    select_extra = ",\n               ".join(offset_exprs)
    q = f"""
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl,
               {select_extra}
        FROM signals s
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.interval = %s
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(interval,))
    conn.close()
    return df


def _fetch() -> pd.DataFrame:
    pieces = [_fetch_interval(iv) for iv in _INTERVAL_MIN]
    return pd.concat(pieces, ignore_index=True)


def _add_pcts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    for n in _BAR_COUNTS:
        df[f"pct_{n}bar"] = (df[f"price_{n}bar"] - df["open_price"]) / df["open_price"] * 100.0 * side
    return df


def _build_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    for n in _BAR_COUNTS:
        df[f"rank_{n}bar"] = df.groupby(_COHORT_KEY)[f"pct_{n}bar"].rank(pct=True)
    return df


def main() -> None:
    print("RSI_Cross(9,24) 5m/15m verisi çekiliyor (1-5 bar arası tüm ölçümler)...")
    df = _fetch()
    df = _add_pcts(df)
    cohorts = _build_ranks(df)
    print(f"Kohort içi sinyal: {len(cohorts)}\n")

    for interval in ["5m", "15m"]:
        idf = cohorts[cohorts["interval"] == interval]
        dakika = _INTERVAL_MIN[interval]
        print(f"=== {interval} (1 bar = {dakika} dakika) ===")
        for sig_type in ["Long", "Short"]:
            sub = idf[idf["signal_type"] == sig_type]
            if len(sub) < 30:
                continue
            print(f"  {sig_type}:")
            for n in _BAR_COUNTS:
                rank_col = f"rank_{n}bar"
                rho, p = spearmanr(sub[rank_col], sub["realized_pnl"])
                tercile = pd.qcut(sub[rank_col], 3, labels=["alt", "orta", "üst"], duplicates="drop")
                g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
                    ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
                )
                ust = g.loc["üst"] if "üst" in g.index else None
                ust_str = (
                    f"üst_wr={ust['wr']*100:.1f}% üst_pnl={ust['ort_pnl']:+.3f}"
                    if ust is not None else "üst: yok"
                )
                print(f"    {n} bar ({n*dakika:>3}dk): rho={rho:+.3f} (p={p:.4f})  {ust_str}")
        print()


if __name__ == "__main__":
    main()
