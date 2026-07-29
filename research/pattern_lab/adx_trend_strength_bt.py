"""
ADX(14) — indicators/core.py::calculate_adx (TradingView/Wilder uyumlu,
do_kirilimi.py'nin zaten kullandigi AYNI fonksiyon) — RSI_Cross(9,24)
sinyallerinde hic filtre olarak kullanilmiyordu, sadece sinyal ureten anda
dict'e ekleniyor ve DB'ye kaydedilmeden dusuyordu (signals tablosunda adx
kolonu yok). 18 Tem 2026, kullanici gozlemi uzerine test ediliyor: sinyal
KENDI TF'sinde, sinyal anindaki (backward-safe) ADX degeri realized_pnl'i
tahmin ediyor mu? Hipotez: dusuk ADX (yatay/sikismis piyasa) = kesisim
sinyalleri guvenilmez, yuksek ADX (gercek trend) = daha guvenilir.

Kullanım: python -m research.pattern_lab.adx_trend_strength_bt
"""

import warnings

import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_adx

_TF_TABLES = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_ADX_LEN = 14


def _fetch_signals() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT id, symbol, signal_type, interval, opened_at, realized_pnl
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND status = 'closed'
          AND realized_pnl IS NOT NULL AND open_price IS NOT NULL AND open_price > 0
          AND interval IN ('5m', '15m', '1h', '4h')
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


def _adx_series(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    table = _TF_TABLES[tf]
    q = f"""
        SELECT symbol, bucket, high, low, close
        FROM {table}
        WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()

    pieces = []
    for symbol, sub in df.groupby("symbol"):
        sub = sub.sort_values("bucket").reset_index(drop=True)
        if len(sub) < _ADX_LEN * 2 + 5:
            continue
        adx, _plus, _minus = calculate_adx(sub, adxlen=_ADX_LEN, dilen=_ADX_LEN)
        pieces.append(pd.DataFrame({"symbol": symbol, "bucket": sub["bucket"], "adx": adx}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "bucket", "adx"]
    )


def _add_adx(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for interval, idf in df.groupby("interval"):
        print(f"  {interval} ADX(14) hesaplanıyor...")
        symbols = idf["symbol"].unique().tolist()
        start = idf["opened_at"].min() - pd.Timedelta(days=5)
        end = idf["opened_at"].max()
        adx_series = _adx_series(interval, symbols, start, end)

        for symbol, sub in idf.groupby("symbol"):
            sub_adx = adx_series[adx_series["symbol"] == symbol].sort_values("bucket")
            if sub_adx.empty:
                sub = sub.copy()
                sub["adx"] = float("nan")
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"),
                sub_adx[["bucket", "adx"]],
                left_on="opened_at",
                right_on="bucket",
                direction="backward",
            )
            pieces.append(merged.drop(columns=["bucket"]))
    return pd.concat(pieces, ignore_index=True)


def _report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["adx"])
        if len(sub) < 30:
            print(f"  {sig_type}: yetersiz örnek")
            continue
        rho, p = spearmanr(sub["adx"], sub["realized_pnl"])
        print(f"  {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")
        tercile = pd.qcut(sub["adx"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean() * 100
        )
        print("   ", g.to_string().replace("\n", "\n    "))


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor (5m/15m/1h/4h)...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    scored = _add_adx(df)
    scored = scored.dropna(subset=["adx"])
    print(f"Geçerli ADX: {len(scored)}")

    _report(scored, "HAM realized_pnl ile ilişki (kendi TF'sinde ADX)")

    _report(scored[scored["interval"] == "5m"], "SADECE 5m")
    _report(scored[scored["interval"] == "15m"], "SADECE 15m")

    real = _fetch_real_trades()
    merged = scored.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"\n=== GERÇEK $ DOĞRULAMASI (n={len(merged)}) ===")
    for sig_type in ["Long", "Short"]:
        sub = merged[merged["signal_type"] == sig_type]
        if len(sub) < 15:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        tercile = pd.qcut(sub["adx"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
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
