"""
Güç Sıralaması (Ranking) panelindeki "hizalanma" kavramının gerçek bir edge mi
yoksa TF'ler arası doğal korelasyondan ötürü çoğunlukla true çıkan gürültülü
bir bayrak mı olduğunu test eder — 18 Tem 2026, kullanıcı sorusu.

_ranking_compute_symbol (live_data_manager.py) ile BİREBİR aynı tanım:
5m/15m/1h/4h'de RSI(14) hesaplanıp yön = (rsi>=50) alınıyor, kaç TF aynı
yönde ("alignment_count") ve sinyalin kendi yönüyle uyuşuyor mu ("dir_match")
inceleniyor. Bu alanlar signals tablosunda saklanmıyor (sadece rank_score/
vs_btc saklanıyor) — bu yüzden RSI(9,24) sinyal popülasyonu için geriye dönük
YENİDEN HESAPLANIYOR (bkz. [[rsi_cross_tf_alignment_bt]] aynı yöntem, HA yerine
RSI kullanıyor), look-ahead güvenli backward merge_asof ile.

Kullanım: python -m research.pattern_lab.rsi_ranking_alignment_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_rsi

_TF_TABLES = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_RSI_PERIOD = 14


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


def _rsi_direction_series(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
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
        if len(sub) < _RSI_PERIOD + 5:
            continue
        rsi = calculate_rsi(sub, period=_RSI_PERIOD)
        pieces.append(pd.DataFrame({
            "symbol": symbol,
            "bucket": sub["bucket"],
            "rsi_bull": rsi >= 50,
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "bucket", "rsi_bull"]
    )


def _add_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)
    end = df["opened_at"].max()

    for tf in _TF_TABLES:
        print(f"  {tf} RSI(14) hesaplanıyor...")
        rsi_dir = _rsi_direction_series(tf, symbols, start, end)
        pieces = []
        for symbol, sub in df.groupby("symbol"):
            sub_rsi = rsi_dir[rsi_dir["symbol"] == symbol].sort_values("bucket")
            if sub_rsi.empty:
                sub = sub.copy()
                sub[f"rsi_bull_{tf}"] = np.nan
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"),
                sub_rsi[["bucket", "rsi_bull"]],
                left_on="opened_at",
                right_on="bucket",
                direction="backward",
            ).rename(columns={"rsi_bull": f"rsi_bull_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)

    dir_cols = [f"rsi_bull_{tf}" for tf in _TF_TABLES]
    valid = df[dir_cols].notna().all(axis=1)
    n_bull = df[dir_cols].sum(axis=1)
    n_bear = len(dir_cols) - n_bull
    df["alignment_count"] = np.where(valid, np.maximum(n_bull, n_bear), np.nan)
    df["ranking_direction"] = np.where(valid, np.where(n_bull >= n_bear, "long", "short"), None)
    df["fully_aligned"] = df["alignment_count"] == len(dir_cols)
    df["dir_match"] = np.where(
        valid,
        (df["signal_type"].str.lower() == df["ranking_direction"]),
        np.nan,
    )
    return df


def _report_base_rate(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["alignment_count"])
    print(f"\n=== TEMEL ORAN: alignment_count dağılımı (n={len(valid)}) ===")
    dist = valid["alignment_count"].value_counts(normalize=True).sort_index() * 100
    for cnt, pct in dist.items():
        print(f"  alignment_count={int(cnt)}: %{pct:.1f}")
    print(f"  TAM hizalı (4/4) oranı: %{(valid['alignment_count']==4).mean()*100:.1f}")
    print(f"  dir_match (ranking_direction == sinyal yönü) oranı: %{valid['dir_match'].mean()*100:.1f}")


def _report_predictive(df: pd.DataFrame) -> None:
    print("\n=== TEK BAŞINA: alignment_count arttıkça performans (yön uyumu şartı yok) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type)].dropna(subset=["alignment_count"])
        if len(sub) < 30:
            continue
        print(f"  {sig_type}:")
        for cnt, g in sub.groupby("alignment_count"):
            print(
                f"    alignment_count={int(cnt)}: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )
        rho, p = spearmanr(sub["alignment_count"], sub["realized_pnl"])
        print(f"    rho(alignment_count, realized_pnl)={rho:+.3f} (p={p:.4f})")

    print("\n=== dir_match ŞARTLI: ranking yönü sinyal yönüyle uyuşuyor mu ===")
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type)].dropna(subset=["dir_match"])
        if len(sub) < 30:
            continue
        print(f"  {sig_type}:")
        for match, g in sub.groupby("dir_match"):
            label = "uyuşuyor" if match else "uyuşmuyor"
            print(
                f"    {label}: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )


def _report_real_dollars(df: pd.DataFrame) -> None:
    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd) ===")
    real = _fetch_real_trades()
    merged = df.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kesişim: {len(merged)}")
    for sig_type in ["Long", "Short"]:
        sub = merged[(merged["signal_type"] == sig_type)].dropna(subset=["dir_match"])
        if len(sub) < 15:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        print(f"  {sig_type}:")
        for match, g in sub.groupby("dir_match"):
            label = "uyuşuyor" if match else "uyuşmuyor"
            print(
                f"    {label}: n={len(g)}, ort_$={g['pnl_usd'].mean():+.3f}, "
                f"toplam_$={g['pnl_usd'].sum():+.2f}, wr={(g['pnl_usd']>0).mean()*100:.1f}%"
            )


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    print("\nGüç Sıralaması formülüyle BİREBİR aynı RSI-hizalanma hesaplanıyor...")
    aligned = _add_alignment(df)

    _report_base_rate(aligned)
    _report_predictive(aligned)
    _report_real_dollars(aligned)


if __name__ == "__main__":
    main()
