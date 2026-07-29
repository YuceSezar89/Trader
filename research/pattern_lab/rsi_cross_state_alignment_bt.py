"""
"RSI Cross hizalanması" — 18 Tem 2026, kullanıcı önerisi: Güç Sıralaması'ndaki
RSI(14)>=50 eşiği yerine, RSI_Cross(9,24) sinyalinin KENDİ mantığını (RSI9 >
RSI24 mü) 4h/1h/15m/5m TF'lerinde durum olarak kullanıp hizalanma test etmek.
HA hizalanması (1h+4h Heikin Ashi) zaten doğrulanmıştı (bkz.
[[project_tf_alignment_early_divergence]]) — bu, RSI9-vs-RSI24 versiyonunun
hiç denenmemiş hali. Yöntem [[rsi_ranking_alignment_bt]] ile birebir aynı,
sadece yön tanımı (rsi_fast>rsi_slow, Config.RSI_FAST_WINDOW/SLOW_WINDOW=9/24,
production ile birebir) değişti.

Kullanım: python -m research.pattern_lab.rsi_cross_state_alignment_bt
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


def _rsi_cross_state_series(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
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
        if len(sub) < _SLOW + 5:
            continue
        rsi_fast = calculate_rsi(sub, period=_FAST)
        rsi_slow = calculate_rsi(sub, period=_SLOW)
        pieces.append(pd.DataFrame({
            "symbol": symbol,
            "bucket": sub["bucket"],
            "cross_bull": rsi_fast > rsi_slow,
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["symbol", "bucket", "cross_bull"]
    )


def _add_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)
    end = df["opened_at"].max()

    for tf in _TF_TABLES:
        print(f"  {tf} RSI9/RSI24 durumu hesaplanıyor...")
        state = _rsi_cross_state_series(tf, symbols, start, end)
        pieces = []
        for symbol, sub in df.groupby("symbol"):
            sub_state = state[state["symbol"] == symbol].sort_values("bucket")
            if sub_state.empty:
                sub = sub.copy()
                sub[f"cross_bull_{tf}"] = np.nan
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"),
                sub_state[["bucket", "cross_bull"]],
                left_on="opened_at",
                right_on="bucket",
                direction="backward",
            ).rename(columns={"cross_bull": f"cross_bull_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)

    dir_cols = [f"cross_bull_{tf}" for tf in _TF_TABLES]
    valid = df[dir_cols].notna().all(axis=1)
    n_bull = df[dir_cols].sum(axis=1)
    n_bear = len(dir_cols) - n_bull
    df["alignment_count"] = np.where(valid, np.maximum(n_bull, n_bear), np.nan)
    df["ranking_direction"] = np.where(valid, np.where(n_bull >= n_bear, "long", "short"), None)
    df["dir_match"] = np.where(
        valid,
        (df["signal_type"].str.lower() == df["ranking_direction"]),
        np.nan,
    )
    # 4h->1h->15m->5m kademeli tam-uyum: her TF'nin kendi sinyal yönüyle uyuşup
    # uyuşmadığı ayrı ayrı da tutuluyor (kademeli onay sayısı için)
    side_bull = df["signal_type"] == "Long"
    cascade_match = pd.DataFrame({
        tf: (df[f"cross_bull_{tf}"] == side_bull) for tf in _TF_TABLES
    })
    df["cascade_count"] = np.where(valid, cascade_match[["4h", "1h", "15m", "5m"]].sum(axis=1), np.nan)
    return df


def _report_base_rate(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["alignment_count"])
    print(f"\n=== TEMEL ORAN: alignment_count dağılımı (n={len(valid)}) ===")
    dist = valid["alignment_count"].value_counts(normalize=True).sort_index() * 100
    for cnt, pct in dist.items():
        print(f"  alignment_count={int(cnt)}: %{pct:.1f}")
    print(f"  dir_match oranı: %{valid['dir_match'].mean()*100:.1f}")


def _report_predictive(df: pd.DataFrame) -> None:
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

    print("\n=== KADEMELİ ONAY SAYISI (4h/1h/15m/5m'den kaçı sinyal yönüyle uyuşuyor) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type)].dropna(subset=["cascade_count"])
        if len(sub) < 30:
            continue
        print(f"  {sig_type}:")
        for cnt, g in sub.groupby("cascade_count"):
            print(
                f"    cascade_count={int(cnt)}: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )
        rho, p = spearmanr(sub["cascade_count"], sub["realized_pnl"])
        print(f"    rho(cascade_count, realized_pnl)={rho:+.3f} (p={p:.4f})")


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
        print(f"  {sig_type} (dir_match):")
        for match, g in sub.groupby("dir_match"):
            label = "uyuşuyor" if match else "uyuşmuyor"
            print(
                f"    {label}: n={len(g)}, ort_$={g['pnl_usd'].mean():+.3f}, "
                f"toplam_$={g['pnl_usd'].sum():+.2f}, wr={(g['pnl_usd']>0).mean()*100:.1f}%"
            )
        sub2 = merged[(merged["signal_type"] == sig_type)].dropna(subset=["cascade_count"])
        print(f"  {sig_type} (cascade_count):")
        for cnt, g in sub2.groupby("cascade_count"):
            print(
                f"    cascade_count={int(cnt)}: n={len(g)}, ort_$={g['pnl_usd'].mean():+.3f}, "
                f"toplam_$={g['pnl_usd'].sum():+.2f}, wr={(g['pnl_usd']>0).mean()*100:.1f}%"
            )


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_signals()
    print(f"Toplam sinyal: {len(df)}")

    print(f"\nRSI Cross (RSI{_FAST}/RSI{_SLOW}) durumuyla 4h/1h/15m/5m hizalanma hesaplanıyor...")
    aligned = _add_alignment(df)

    _report_base_rate(aligned)
    _report_predictive(aligned)
    _report_real_dollars(aligned)


if __name__ == "__main__":
    main()
