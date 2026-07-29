"""
R-Score (Guc Siralamasi panelindeki Sortino/Omega/Calmar/Sharpe karmasi,
live_data_manager.py::_ranking_r_score ile AYNI formul, 1h TF, 14 bar
pencere) sinyal kalitesini tahmin ediyor mu? 18 Tem 2026, alfa/beta hayal
kirikligindan sonra kullanici istegi. r_score signals tablosunda saklanmiyor
— RSI_Cross(9,24) sinyalleri icin geriye donuk, backward-safe yeniden
hesaplaniyor.

Kullanım: python -m research.pattern_lab.r_score_signal_quality_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_R_PERIOD = 14
_EPS = 1e-12


def _fetch_signals() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT id, symbol, signal_type, opened_at, realized_pnl
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND status = 'closed'
          AND realized_pnl IS NOT NULL AND open_price IS NOT NULL AND open_price > 0
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _fetch_real_trades() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = "SELECT signal_id, pnl_usd FROM paper_trades WHERE strategy = 'rsi_cross_live' AND status = 'closed'"
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _r_score_series(symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, bucket, close
        FROM cagg_1h
        WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()

    pieces = []
    for symbol, sub in df.groupby("symbol"):
        sub = sub.sort_values("bucket").reset_index(drop=True)
        if len(sub) < _R_PERIOD + 2:
            continue
        close = sub["close"].astype(float)
        log_ret = np.log(close + _EPS).diff()

        avg = log_ret.rolling(_R_PERIOD).mean()
        std = log_ret.rolling(_R_PERIOD).std() + _EPS
        sharpe = avg / std

        neg_ret = log_ret.where(log_ret < 0)
        neg_std = neg_ret.rolling(_R_PERIOD, min_periods=2).std() + _EPS
        sortino = avg / neg_std

        roll_max = close.rolling(_R_PERIOD + 1).max()
        roll_min = close.rolling(_R_PERIOD + 1).min()
        max_dd = (roll_max - roll_min) / (roll_max + _EPS)
        calmar = avg / (max_dd + _EPS)

        pos_ret = log_ret.where(log_ret >= 0, 0.0)
        neg_ret_z = log_ret.where(log_ret < 0, 0.0)
        gains = pos_ret.rolling(_R_PERIOD).sum()
        losses = (-neg_ret_z).rolling(_R_PERIOD).sum() + _EPS
        omega = gains / losses

        r_score = sortino * 0.40 + omega * 0.30 + calmar * 0.20 + sharpe * 0.10
        pieces.append(pd.DataFrame({"symbol": symbol, "bucket": sub["bucket"], "r_score": r_score}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "bucket", "r_score"]
    )


def _add_r_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=5)
    end = df["opened_at"].max()

    print("  1h R-Score serisi hesaplanıyor...")
    r_series = _r_score_series(symbols, start, end)

    pieces = []
    for symbol, sub in df.groupby("symbol"):
        sub_r = r_series[r_series["symbol"] == symbol].sort_values("bucket")
        if sub_r.empty:
            sub = sub.copy()
            sub["r_score"] = np.nan
            pieces.append(sub)
            continue
        merged = pd.merge_asof(
            sub.sort_values("opened_at"),
            sub_r[["bucket", "r_score"]],
            left_on="opened_at",
            right_on="bucket",
            direction="backward",
        )
        pieces.append(merged.drop(columns=["bucket"]))
    return pd.concat(pieces, ignore_index=True)


def _report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["r_score"])
        if len(sub) < 30:
            print(f"  {sig_type}: yetersiz örnek")
            continue
        rho, p = spearmanr(sub["r_score"], sub["realized_pnl"])
        print(f"  {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")
        tercile = pd.qcut(sub["r_score"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean() * 100
        )
        print("   ", g.to_string().replace("\n", "\n    "))


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    scored = _add_r_score(df)
    scored = scored.dropna(subset=["r_score"])
    print(f"Geçerli R-Score: {len(scored)}")

    _report(scored, "HAM realized_pnl ile ilişki")

    real = _fetch_real_trades()
    merged = scored.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"\n=== GERÇEK $ DOĞRULAMASI (n={len(merged)}) ===")
    for sig_type in ["Long", "Short"]:
        sub = merged[merged["signal_type"] == sig_type]
        if len(sub) < 15:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        tercile = pd.qcut(sub["r_score"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True).agg(
            n=("pnl_usd", "count"),
            ort_usd=("pnl_usd", "mean"),
            toplam_usd=("pnl_usd", "sum"),
            wr=("pnl_usd", lambda s: (s > 0).mean() * 100),
        )
        print(f"  {sig_type}:")
        print("   ", g.to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
