"""
HA_Cross icin devisso_score (ERSI) testi — [[ha_cross_tf_alignment_bt]] (2-TF,
1h+4h) + [[ha_cross_early_divergence_bt]] (erken-ayrisma) uzerine ucuncu bir
boyut olarak devisso_score ekleniyor. Hafizada CELISKILI gecmis var: 9 Tem
Supertrend rolling-cohort pilotunda "en guclu" cikmisti, 12 Tem'de ("ERSI
cürüdü") basarisiz oldu — bu yuzden VARSAYILMADAN, bu spesifik sinyal
kumesinde (HA_Cross, TF-hizali) yeniden test ediliyor.

devisso_score = ERSI (RSI Verimliligi): Δprice%/ΔRSI, EMA(7), kendi son 100
barina gore percentile (0-100). signal_processor.py:254-280.

Uc test:
1. devisso_rank (kohort-ici percentile) tek basina realized_pnl ile korelasyonlu mu
2. Zaten hizali (1h+4h) + cok ayrisan (early_rank ust) sinyaller icinde
   devisso_score ekstra ayrim katiyor mu
3. early_rank + devisso_rank birlesik (ortalama) skoru, early_rank'in tek
   basinasindan daha mi guclu

Kullanım: python -m research.pattern_lab.ha_cross_devisso_bt
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
_INTERVAL_MIN = {"5m": 5, "15m": 15}
_HTF_TABLES = {"1h": "cagg_1h", "4h": "cagg_4h"}


def _fetch_interval(interval: str, early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    early_offset_1m = early_bars * _INTERVAL_MIN[interval] - 1
    q = """
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl, s.devisso_score, pd.close AS price_early
        FROM signals s
        JOIN LATERAL (
            SELECT p.close
            FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m'
              AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC
            LIMIT 1 OFFSET %s
        ) pd ON true
        WHERE s.indicators = 'HA_Cross' AND s.interval = %s
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
          AND s.devisso_score IS NOT NULL
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
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    df["early_pct"] = raw_pct * side
    return df


def _build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    df["early_rank"] = df.groupby(_COHORT_KEY)["early_pct"].rank(pct=True)
    df["devisso_rank"] = df.groupby(_COHORT_KEY)["devisso_score"].rank(pct=True)
    df["composite_rank"] = (df["early_rank"] + df["devisso_rank"]) / 2.0
    return df


def _placebo(df: pd.DataFrame, col: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[col] = out.groupby(_COHORT_KEY)[col].transform(lambda s: rng.permutation(s.values))
    return out


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
    aligned_1h = (df["signal_type"] == "Long") == (df["ha_bull_1h"] == True)  # noqa: E712
    aligned_4h = (df["signal_type"] == "Long") == (df["ha_bull_4h"] == True)  # noqa: E712
    df["aligned_count"] = aligned_1h.astype(int) + aligned_4h.astype(int)
    df.loc[df["ha_bull_1h"].isna() | df["ha_bull_4h"].isna(), "aligned_count"] = np.nan
    return df


def _report_correlation(df: pd.DataFrame, col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=[col])
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub[col], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _report_buckets(df: pd.DataFrame, col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=[col])
        if len(sub) < 30:
            continue
        tercile = pd.qcut(sub[col], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  [{label}] {sig_type}:")
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal (devisso_score dolu): {len(cohorts)}")

    print("\n=== TEST 1: devisso_rank TEK BAŞINA (bu sinyal kümesinde) ===")
    _report_correlation(cohorts, "devisso_rank", "gerçek")
    _report_buckets(cohorts, "devisso_rank", "gerçek")
    print("\n  -- placebo --")
    _report_correlation(_placebo(cohorts, "devisso_rank"), "devisso_rank", "placebo")

    print("\n1h/4h hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("\n=== TEST 2: TAM HİZALI (1h+4h) + ÇOK AYRIŞAN (early_rank üst) içinde devisso_score ekstra ayrım katıyor mu ===")
    for sig_type in ["Long", "Short"]:
        sub = aligned[
            (aligned["signal_type"] == sig_type)
            & (aligned["aligned_count"] == 2)
            & (aligned["early_rank"] >= aligned["early_rank"].quantile(0.667))
        ]
        if len(sub) < 30:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        tercile = pd.qcut(sub["devisso_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  {sig_type} (hizalı+çok-ayrışan içinde, n={len(sub)}):")
        print(g.to_string().replace("\n", "\n    "))

    print("\n=== TEST 3: BİRLEŞİK (early_rank+devisso_rank ortalaması) vs SADECE early_rank ===")
    print("  -- composite_rank --")
    _report_correlation(aligned, "composite_rank", "composite")
    _report_buckets(aligned, "composite_rank", "composite")
    print("  -- sadece early_rank (referans) --")
    _report_correlation(aligned, "early_rank", "early_only")


if __name__ == "__main__":
    main()
