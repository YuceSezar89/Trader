"""
HA_Cross icin 5-TF hizalanma testi — [[ha_cross_tf_alignment_bt]]'nin (1h+4h,
aligned_count 0-2) genisletilmis hali. Hocanin orijinal tarifi olan
_HTF_CONFIRM_TFS = [4h,6h,8h,12h,1d] (signal_processor.py:138) artik TAM
kapsaniyor: 4h cagg_4h'ten, 6h/8h/12h/1d ise DB'de CA olarak hic olmadigi icin
ham 1m price_data'dan kendi resample'imizle turetiliyor (18 Tem 2026).

Yontem: 4h icin oldugu gibi CA'dan, digerleri icin 1m'den pandas resample +
ozyinelemeli Heikin Ashi. aligned_count artik 0-5 arasi (once 0-2 idi).

Kullanım: python -m research.pattern_lab.ha_cross_tf_alignment5_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.ha_cross_early_divergence_bt import (
    _EARLY_BARS,
    _add_early_pct,
    _build_cohorts,
    _fetch,
)

_CA_TFS = {"4h": "cagg_4h"}
_RESAMPLE_TFS = {"6h": 360, "8h": 480, "12h": 720, "1d": 1440}
_ORIGIN = "2000-01-01 03:00:00"


def _compute_ha(df: pd.DataFrame, time_col: str = "bucket") -> pd.DataFrame:
    df = df.sort_values(time_col).reset_index(drop=True)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    if n > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    df["ha_bull"] = ha_close > ha_open
    return df[[time_col, "ha_bull"]].rename(columns={time_col: "bucket"})


def _fetch_ca_ha(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    table = _CA_TFS[tf]
    q = f"""
        SELECT symbol, bucket, open, high, low, close
        FROM {table}
        WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()
    pieces = [
        _compute_ha(sub).assign(symbol=symbol) for symbol, sub in df.groupby("symbol")
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["bucket", "ha_bull", "symbol"]
    )


def _fetch_1m(symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, timestamp, open, high, low, close
        FROM price_data
        WHERE symbol = ANY(%s) AND interval = '1m' AND timestamp BETWEEN %s AND %s
        ORDER BY symbol, timestamp
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()
    return df


def _resample_ha(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    pieces = []
    for symbol, sub in df_1m.groupby("symbol"):
        sub = sub.set_index("timestamp").sort_index()
        res = sub.resample(f"{minutes}min", origin=pd.Timestamp(_ORIGIN)).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if res.empty:
            continue
        res = res.reset_index().rename(columns={"timestamp": "bucket"})
        ha = _compute_ha(res)
        ha["symbol"] = symbol
        pieces.append(ha)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["bucket", "ha_bull", "symbol"]
    )


def _merge_alignment(df: pd.DataFrame, ha: pd.DataFrame, tf: str) -> pd.DataFrame:
    pieces = []
    for symbol, sub in df.groupby("symbol"):
        ha_sym = ha[ha["symbol"] == symbol].sort_values("bucket")
        if ha_sym.empty:
            sub = sub.copy()
            sub[f"ha_bull_{tf}"] = np.nan
            pieces.append(sub)
            continue
        merged = pd.merge_asof(
            sub.sort_values("opened_at"),
            ha_sym[["bucket", "ha_bull"]],
            left_on="opened_at",
            right_on="bucket",
            direction="backward",
        ).rename(columns={"ha_bull": f"ha_bull_{tf}"})
        pieces.append(merged.drop(columns=["bucket"]))
    return pd.concat(pieces, ignore_index=True)


def _add_htf_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=15)  # HA isinmasi icin pay
    end = df["opened_at"].max()

    for tf in _CA_TFS:
        print(f"  {tf} (CA'dan) Heikin Ashi hesaplanıyor...")
        ha = _fetch_ca_ha(tf, symbols, start, end)
        df = _merge_alignment(df, ha, tf)

    print("  1m veri çekiliyor (6h/8h/12h/1d türetmek için, biraz sürebilir)...")
    df_1m = _fetch_1m(symbols, start, end)
    print(f"  {len(df_1m)} adet 1m bar çekildi")

    for tf, minutes in _RESAMPLE_TFS.items():
        print(f"  {tf} (1m'den türetilmiş) Heikin Ashi hesaplanıyor...")
        ha = _resample_ha(df_1m, minutes)
        df = _merge_alignment(df, ha, tf)

    all_tfs = list(_CA_TFS) + list(_RESAMPLE_TFS)
    aligned_cols = []
    for tf in all_tfs:
        col = f"ha_bull_{tf}"
        aligned = (df["signal_type"] == "Long") == (df[col] == True)  # noqa: E712
        aligned_cols.append(aligned.astype(int))

    df["aligned_count"] = sum(aligned_cols)
    na_mask = df[[f"ha_bull_{tf}" for tf in all_tfs]].isna().any(axis=1)
    df.loc[na_mask, "aligned_count"] = np.nan
    df["n_tfs"] = len(all_tfs)
    return df


def _report_standalone(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["aligned_count"])
        if len(sub) < 30:
            continue
        print(f"  {sig_type}:")
        for cnt, g in sub.groupby("aligned_count"):
            print(
                f"    aligned_count={int(cnt)}/5: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )
        rho, p = spearmanr(sub["aligned_count"], sub["realized_pnl"])
        print(f"    rho(aligned_count, realized_pnl)={rho:+.3f} (p={p:.4f})")


def _report_combined(df: pd.DataFrame, min_aligned: int) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type) & (df["aligned_count"] >= min_aligned)]
        if len(sub) < 30:
            print(f"  {sig_type}: aligned_count>={min_aligned} icin yetersiz ornek (n={len(sub)})")
            continue
        tercile = pd.qcut(sub["early_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  {sig_type} (aligned_count>={min_aligned}/5, n={len(sub)}):")
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("\n4h/6h/8h/12h/1d Heikin Ashi hizalanması hesaplanıyor (sürebilir)...")
    aligned = _add_htf_alignment(cohorts)

    print("\n=== TEK BAŞINA: aligned_count (0-5) arttıkça performans ===")
    _report_standalone(aligned)

    print("\n=== BİRLEŞİK: TAM hizalı (5/5) sinyallerde erken-ayrışma tercile ===")
    _report_combined(aligned, min_aligned=5)

    print("\n=== BİRLEŞİK (gevşetilmiş): en az 4/5 hizalı sinyallerde erken-ayrışma tercile ===")
    _report_combined(aligned, min_aligned=4)


if __name__ == "__main__":
    main()
