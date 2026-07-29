"""
Supertrend(10,3.0) Short icin meanrev_state rejim filtresi + bugunku
TF-hizalanma/erken-ayrisma istiflemesi — [[project_turtle_traders]] 11 Tem
"2 SAGLAM BULGU"nun ikincisi: "Supertrend Short + meanrev_state<=0" (asiri-
alim DEGILse). 18 Tem 2026, is_consolidating/RSI_Cross Long testinin (bkz.
[[rsi_cross_long_consolidating_bt]]) Supertrend/Short karsiligi.

meanrev_state: devissotrader_agents_bt.py::_mean_reversion_state_series ile
BIREBIR AYNI formul — RSI(SMA,14) + Bollinger(20,2) pozisyonu. +1=asiri-satim,
-1=asiri-alim, 0=notr. Short icin filtre: meanrev_state<=0 (asiri-alim
DEGIL — cunku asiri-alimda Short zaten "dogal", filtre onu degil tam tersini
eliyor... NOT: orijinal script yonunu netlestirmek icin -1/+1 isaretine dikkat).

NOT: paper_trades'te 'supertrend' stratejisi yok — sadece signals.realized_pnl
(zayif proxy) ile test edilebiliyor, gercek $ dogrulamasi bu sinyal turu icin
mumkun degil.

Kullanım: python -m research.pattern_lab.supertrend_short_meanrev_bt
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
_EARLY_BARS = 5
_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15}
_HTF_TABLES = {"1h": "cagg_1h", "4h": "cagg_4h"}

OVERSOLD, OVERBOUGHT = 30, 70
BB_WIDTH_MIN = 0.03
BAR_DURATION = pd.Timedelta(minutes=15)
DAYS = 45


def _fetch_interval(interval: str, early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    early_offset_1m = early_bars * _INTERVAL_MIN[interval] - 1
    q = """
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl, pd.close AS price_early
        FROM signals s
        JOIN LATERAL (
            SELECT p.close
            FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m'
              AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC
            LIMIT 1 OFFSET %s
        ) pd ON true
        WHERE s.indicators = 'Supertrend(10,3.0)' AND s.interval = %s
          AND s.signal_type = 'Short'
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(early_offset_1m, interval))
    conn.close()
    return df


def _fetch(early_bars: int) -> pd.DataFrame:
    pieces = [_fetch_interval(iv, early_bars) for iv in _INTERVAL_MIN]
    return pd.concat(pieces, ignore_index=True)


def _add_early_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_pct = (df["price_early"] - df["open_price"]) / df["open_price"] * 100.0
    df["early_pct"] = raw_pct * -1.0  # Short: dusus = lehte
    return df


def _build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    df["early_rank"] = df.groupby(_COHORT_KEY)["early_pct"].rank(pct=True)
    return df


def _compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("bucket").reset_index(drop=True)
    o, h, l, c = (df[x].to_numpy() for x in ["open", "high", "low", "close"])
    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    if n > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    df["ha_bull"] = ha_close > ha_open
    return df[["bucket", "ha_bull"]]


def _fetch_htf_ha(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    table = _HTF_TABLES[tf]
    q = f"SELECT symbol, bucket, open, high, low, close FROM {table} WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s ORDER BY symbol, bucket"
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()
    pieces = [_compute_ha(sub).assign(symbol=symbol) for symbol, sub in df.groupby("symbol")]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["bucket", "ha_bull", "symbol"])


def _add_htf_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)
    end = df["opened_at"].max()
    for tf in _HTF_TABLES:
        ha = _fetch_htf_ha(tf, symbols, start, end)
        pieces = []
        for symbol, sub in df.groupby("symbol"):
            ha_sym = ha[ha["symbol"] == symbol].sort_values("bucket")
            if ha_sym.empty:
                sub = sub.copy()
                sub[f"ha_bull_{tf}"] = np.nan
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"), ha_sym[["bucket", "ha_bull"]],
                left_on="opened_at", right_on="bucket", direction="backward",
            ).rename(columns={"ha_bull": f"ha_bull_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)
    # Short: hizali = HA BEARISH (ha_bull=False)
    aligned_1h = df["ha_bull_1h"] == False  # noqa: E712
    aligned_4h = df["ha_bull_4h"] == False  # noqa: E712
    df["aligned_count"] = aligned_1h.astype(int) + aligned_4h.astype(int)
    df.loc[df["ha_bull_1h"].isna() | df["ha_bull_4h"].isna(), "aligned_count"] = np.nan
    return df


def _calculate_rsi_sma(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _mean_reversion_state_series(g: pd.DataFrame) -> pd.Series:
    close = g["close"]
    rsi = _calculate_rsi_sma(close, period=14)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)

    mr_up = (rsi < OVERSOLD) & (bb_position < 0.1) & (bb_width > BB_WIDTH_MIN)
    mr_down = (rsi > OVERBOUGHT) & (bb_position > 0.9) & (bb_width > BB_WIDTH_MIN)
    return (mr_up.astype(int) - mr_down.astype(int)).fillna(0)


def _fetch_regime(symbols: list) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, close
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < 50:
            continue
        state = _mean_reversion_state_series(g)
        out.append(pd.DataFrame({"symbol": sym, "ts": g["ts"], "meanrev_state": state.astype(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "ts", "meanrev_state"])


def _merge_regime(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cutoff"] = df["opened_at"] - BAR_DURATION
    return pd.merge_asof(
        df.sort_values("cutoff"), regime_df.sort_values("ts"),
        left_on="cutoff", right_on="ts", by="symbol", direction="backward",
    )


def main() -> None:
    print("Supertrend(10,3.0) Short verisi çekiliyor...")
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    print(f"Toplam sinyal: {len(df)}")

    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")
    if len(cohorts) < 30:
        print("Yetersiz örneklem, çıkılıyor.")
        return

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("meanrev_state (15m, hocanın kodu) hesaplanıyor...")
    regime = _fetch_regime(aligned["symbol"].unique().tolist())
    merged = _merge_regime(aligned, regime)

    print("\n=== BACKTEST realized_pnl: kademeli filtre ===")
    s1 = merged[merged["aligned_count"] == 2]
    print(f"  (1) +hizalı (1h+4h bearish):       n={len(s1)}, ort_pnl={s1['realized_pnl'].mean():+.4f}, wr={(s1['realized_pnl']>0).mean()*100:.1f}%" if len(s1) > 0 else "  (1) örnek yok")

    s2 = s1[s1["early_rank"] >= s1["early_rank"].quantile(0.667)] if len(s1) > 0 else s1
    print(f"  (2) +çok-ayrışan (üst 1/3):        n={len(s2)}, ort_pnl={s2['realized_pnl'].mean():+.4f}, wr={(s2['realized_pnl']>0).mean()*100:.1f}%" if len(s2) > 0 else "  (2) örnek yok")

    s3 = s2[s2["meanrev_state"] <= 0.0] if len(s2) > 0 else s2
    print(f"  (3) +meanrev_state<=0 (aşırı-alım DEĞİL): n={len(s3)}, ort_pnl={s3['realized_pnl'].mean():+.4f}, wr={(s3['realized_pnl']>0).mean()*100:.1f}%" if len(s3) > 0 else "  (3) örnek yok")

    rho, p = spearmanr(merged.dropna(subset=["meanrev_state"])["meanrev_state"], merged.dropna(subset=["meanrev_state"])["realized_pnl"])
    print(f"\n  (tek başına) rho(meanrev_state, realized_pnl)={rho:+.3f} (p={p:.4f}), n={merged['meanrev_state'].notna().sum()}")


if __name__ == "__main__":
    main()
