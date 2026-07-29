"""
"RSI Cross hizalanması" — BİNER sayım değil, VPMV'nin combined skoruyla BİREBİR
aynı yapı: her TF'de RSI9-RSI24 farkı (spread) alınır, normalize_momentum_0_100
(utils/preprocessing.py — production'da momentum bileşeni için kullanılan AYNI
fonksiyon) ile 0-100'e normalize edilir, _RANKING_TF_WEIGHTS (5m:0.35, 15m:0.30,
1h:0.20, 4h:0.15) ile ağırlıklı ortalanır. 18 Tem 2026, bkz.
[[rsi_cross_state_alignment_bt]] (ikili sayım versiyonu, cascade_count=4 gerçek
$'da net pozitif çıkmıştı — bu script AYNI fikri sayısal/dereceli skora çevirip
tekrar test ediyor).

Kullanım: python -m research.pattern_lab.rsi_cross_combined_score_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_rsi
from utils.preprocessing import normalize_momentum_0_100

_TF_TABLES = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_TF_WEIGHTS = {"5m": 0.35, "15m": 0.30, "1h": 0.20, "4h": 0.15}
_FAST = Config.RSI_FAST_WINDOW
_SLOW = Config.RSI_SLOW_WINDOW


def _fetch_signals() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT id, symbol, signal_type, interval, opened_at, open_price, realized_pnl
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


def _rsi_cross_score_series(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    table = _TF_TABLES[tf]
    q = f"""
        SELECT symbol, bucket, open, high, low, close
        FROM {table}
        WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()

    pieces = []
    for symbol, sub in df.groupby("symbol"):
        sub = sub.sort_values("bucket").reset_index(drop=True)
        if len(sub) < _SLOW + 15:
            continue
        rsi_fast = calculate_rsi(sub, period=_FAST)
        rsi_slow = calculate_rsi(sub, period=_SLOW)
        spread = rsi_fast - rsi_slow
        score = normalize_momentum_0_100(spread)
        pieces.append(pd.DataFrame({"symbol": symbol, "bucket": sub["bucket"], "score": score}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "bucket", "score"]
    )


def _add_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)
    end = df["opened_at"].max()

    for tf in _TF_TABLES:
        print(f"  {tf} RSI Cross skoru (spread + normalize_momentum_0_100) hesaplanıyor...")
        score = _rsi_cross_score_series(tf, symbols, start, end)
        pieces = []
        for symbol, sub in df.groupby("symbol"):
            sub_score = score[score["symbol"] == symbol].sort_values("bucket")
            if sub_score.empty:
                sub = sub.copy()
                sub[f"score_{tf}"] = np.nan
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"),
                sub_score[["bucket", "score"]],
                left_on="opened_at",
                right_on="bucket",
                direction="backward",
            ).rename(columns={"score": f"score_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)

    total_w = np.zeros(len(df))
    weighted_sum = np.zeros(len(df))
    for tf in _TF_TABLES:
        col = df[f"score_{tf}"]
        has = col.notna()
        weighted_sum += np.where(has, col.fillna(0) * _TF_WEIGHTS[tf], 0)
        total_w += np.where(has, _TF_WEIGHTS[tf], 0)
    df["combined"] = np.where(total_w > 0, weighted_sum / np.where(total_w > 0, total_w, 1), np.nan)

    # Yön-ayarlı: Long icin yuksek combined iyi, Short icin dusuk combined iyi
    # (VPMV panel_numeric_filters testindeki AYNI kural, [[project_panel_numeric_filters]])
    df["combined_adj"] = np.where(df["signal_type"] == "Long", df["combined"], 100 - df["combined"])
    return df


def _report_correlation(df: pd.DataFrame) -> None:
    print("\n=== combined_adj (yön-ayarlı skor) vs realized_pnl ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["combined_adj"])
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["combined_adj"], sub["realized_pnl"])
        print(f"  {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")
        tercile = pd.qcut(sub["combined_adj"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(g.to_string().replace("\n", "\n    "))


def _report_real_dollars(df: pd.DataFrame) -> None:
    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd) ===")
    real = _fetch_real_trades()
    merged = df.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kesişim: {len(merged)}")
    for sig_type in ["Long", "Short"]:
        sub = merged[merged["signal_type"] == sig_type].dropna(subset=["combined_adj"])
        if len(sub) < 15:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        print(f"  {sig_type} (n={len(sub)}):")
        tercile = pd.qcut(sub["combined_adj"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True).agg(
            n=("pnl_usd", "count"),
            ort_usd=("pnl_usd", "mean"),
            toplam_usd=("pnl_usd", "sum"),
            wr=("pnl_usd", lambda s: (s > 0).mean() * 100),
        )
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    print("\nRSI Cross ağırlıklı combined skoru hesaplanıyor (VPMV combined ile birebir aynı yapı)...")
    scored = _add_combined_score(df)

    _report_correlation(scored)
    _report_real_dollars(scored)


if __name__ == "__main__":
    main()
