"""
HA_Cross icin TF hizalanma testi — hocanin "N TF ayni yonde hizalansin" fikrinin
gercek Heikin Ashi ile test edilmis hali. 6h/8h/12h/1d tarihsel veride yok
(sadece cagg_5m/15m/1h/4h var, bkz. [[project_pattern_lab]] 17 Tem notu) — bu
yuzden elde mevcut EN YUKSEK iki TF (1h, 4h) kullanildi.

Yontem: her HA_Cross sinyali icin, sinyal aninda en son KAPANMIS 1h ve 4h
barinin gercek (ozyinelemeli) Heikin Ashi yonu hesaplanir. aligned_count = kac
tanesi sinyal yonuyle uyusuyor (0/1/2).

Iki test:
1. Tek basina: aligned_count arttikca realized_pnl/win-rate iyilesiyor mu
2. Kohort testiyle birlikte: [[ha_cross_early_divergence_bt]]'teki "en cok
   ayrisani sec" yontemine, hizalanmayi ek filtre olarak eklersek sonuc daha mi
   guclu (sadece hizali olan sinyallerde tercile analizi tekrarlanir)

Kullanım: python -m research.pattern_lab.ha_cross_tf_alignment_bt
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

_HTF_TABLES = {"1h": "cagg_1h", "4h": "cagg_4h"}


def _compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Tek sembol icin ozyinelemeli Heikin Ashi yonunu hesaplar (bucket'a gore sirali)."""
    df = df.sort_values("bucket").reset_index(drop=True)
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
    return df[["bucket", "ha_bull"]]


def _fetch_htf_ha(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    table = _HTF_TABLES[tf]
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
        ha = _compute_ha(sub)
        ha["symbol"] = symbol
        pieces.append(ha)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["bucket", "ha_bull", "symbol"]
    )


def _add_htf_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)  # HA'nin isinmasi icin pay
    end = df["opened_at"].max()

    for tf in _HTF_TABLES:
        print(f"  {tf} Heikin Ashi hesaplanıyor...")
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
                sub.sort_values("opened_at"),
                ha_sym[["bucket", "ha_bull"]],
                left_on="opened_at",
                right_on="bucket",
                direction="backward",
            ).rename(columns={"ha_bull": f"ha_bull_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)

    aligned_1h = (df["signal_type"] == "Long") == (df["ha_bull_1h"] == True)  # noqa: E712
    aligned_4h = (df["signal_type"] == "Long") == (df["ha_bull_4h"] == True)  # noqa: E712
    df["aligned_count"] = aligned_1h.astype(int) + aligned_4h.astype(int)
    df.loc[df["ha_bull_1h"].isna() | df["ha_bull_4h"].isna(), "aligned_count"] = np.nan
    return df


def _report_standalone(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["aligned_count"])
        if len(sub) < 30:
            continue
        print(f"  {sig_type}:")
        for cnt, g in sub.groupby("aligned_count"):
            print(
                f"    aligned_count={int(cnt)}: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )
        rho, p = spearmanr(sub["aligned_count"], sub["realized_pnl"])
        print(f"    rho(aligned_count, realized_pnl)={rho:+.3f} (p={p:.4f})")


def _report_combined(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type) & (df["aligned_count"] == 2)]
        if len(sub) < 30:
            print(f"  {sig_type}: aligned_count=2 icin yetersiz ornek (n={len(sub)})")
            continue
        tercile = pd.qcut(sub["early_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  {sig_type} (sadece aligned_count=2, n={len(sub)}):")
        print(g.to_string().replace("\n", "\n    "))


def _report_by_interval(df: pd.DataFrame) -> None:
    for interval, idf in df.groupby("interval"):
        print(f" -- interval={interval} --")
        for sig_type in ["Long", "Short"]:
            sub = idf[
                (idf["signal_type"] == sig_type)
                & (idf["aligned_count"] == 2)
                & (idf["early_rank"] >= idf["early_rank"].quantile(0.667) if len(idf) > 0 else False)
            ]
            if len(sub) < 15:
                print(f"    {sig_type}: yetersiz örnek (n={len(sub)})")
                continue
            print(
                f"    {sig_type}: n={len(sub)}, ort_pnl={sub['realized_pnl'].mean():+.4f}, "
                f"wr={(sub['realized_pnl']>0).mean()*100:.1f}%"
            )


def main() -> None:
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("\n1h/4h Heikin Ashi hizalanması hesaplanıyor (biraz sürebilir)...")
    aligned = _add_htf_alignment(cohorts)

    print("\n=== TEK BAŞINA: aligned_count arttıkça performans ===")
    _report_standalone(aligned)

    print("\n=== BİRLEŞİK: sadece TAM hizalı (1h+4h ikisi de uyumlu) sinyallerde erken-ayrışma tercile ===")
    _report_combined(aligned)

    print("\n=== BİRLEŞİK — INTERVAL'A GÖRE KIRILIM (hizalı + üst 1/3 ayrışan) ===")
    _report_by_interval(aligned)


if __name__ == "__main__":
    main()
