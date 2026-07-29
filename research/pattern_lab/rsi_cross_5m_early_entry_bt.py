"""
Erken giriş testi — 5m RSI_Cross(9,24) sinyali geldiğinde, 15m onayını (aynı
sembol+yön RSI_Cross'un birkaç bar içinde 15m'de de gelmesi) beklemeden erken
girmek mantıklı mı? Kullanıcının hedefi: harekete erken dahil olmak, 15m'nin
"büyüteceğini" sinyalin kendi erken fiyat hareketinden (early_pct, bkz.
rsi_cross_early_divergence_bt.py) önceden kestirebiliyor muyuz.

Not: price_data sadece '1m' interval tutuyor (CA'lar 5m/15m/1h/4h — hiçbiri bu
script'te kullanılmıyor). early_pct bu yüzden 1m barlardan ZAMAN OFSETİYLE
hesaplanıyor (5m sinyalde "5 bar sonrası" = 25 dakika sonrası), interval eşleşmesi
ARANMIYOR — [[project_pattern_lab]]'daki önceki script'te bulunan interval-mismatch
bug'ının düzeltilmiş hali.

escalated: aynı (symbol, signal_type) için, 5m sinyalinden sonraki
_ESCALATION_WINDOW_MIN dakika içinde 15m RSI_Cross(9,24) sinyali de geldiyse True.

Kullanım: python -m research.pattern_lab.rsi_cross_5m_early_entry_bt
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

from config import Config

_EARLY_BARS = 5  # 5m sinyalin kendi TF'sine göre "5 bar" = 25 dakika
_INTERVAL_MIN = 5
_ESCALATION_WINDOW_MIN = 30  # 15m onayı için bekleme penceresi


def _fetch_5m_signals() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    early_offset_1m = _EARLY_BARS * _INTERVAL_MIN - 1  # 1m barlarda OFFSET
    q = """
        SELECT s.symbol, s.signal_type, s.opened_at, s.open_price, s.realized_pnl,
               pd.close AS price_early
        FROM signals s
        JOIN LATERAL (
            SELECT p.close
            FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m'
              AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC
            LIMIT 1 OFFSET %s
        ) pd ON true
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.interval = '5m'
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(early_offset_1m,))
    conn.close()
    return df


def _fetch_15m_signals(symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, opened_at
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND interval = '15m'
          AND symbol = ANY(%s) AND opened_at BETWEEN %s AND %s
    """
    df = pd.read_sql(q, conn, params=(symbols, start, end + timedelta(minutes=_ESCALATION_WINDOW_MIN)))
    conn.close()
    return df


def _add_early_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_pct = (df["price_early"] - df["open_price"]) / df["open_price"] * 100.0
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    df["early_pct"] = raw_pct * side
    return df


def _add_escalation_flag(df5: pd.DataFrame) -> pd.DataFrame:
    df5 = df5.copy()
    df15 = _fetch_15m_signals(
        df5["symbol"].unique().tolist(), df5["opened_at"].min(), df5["opened_at"].max()
    )
    df5["escalated"] = False
    for (symbol, sig_type), sub15 in df15.groupby(["symbol", "signal_type"]):
        mask = (df5["symbol"] == symbol) & (df5["signal_type"] == sig_type)
        idx = df5[mask].index
        for i in idx:
            t0 = df5.at[i, "opened_at"]
            window = sub15[
                (sub15["opened_at"] > t0)
                & (sub15["opened_at"] <= t0 + timedelta(minutes=_ESCALATION_WINDOW_MIN))
            ]
            df5.at[i, "escalated"] = len(window) > 0
    return df5


def _report(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        esc = sub[sub["escalated"]]
        non_esc = sub[~sub["escalated"]]
        print(
            f"  {sig_type}: n={len(sub)}, escalated={len(esc)} ({len(esc)/len(sub)*100:.1f}%), "
            f"non_escalated={len(non_esc)}"
        )
        print(
            f"    escalated     ort_pnl={esc['realized_pnl'].mean():+.4f}  "
            f"wr={(esc['realized_pnl'] > 0).mean()*100:.1f}%"
        )
        print(
            f"    non_escalated ort_pnl={non_esc['realized_pnl'].mean():+.4f}  "
            f"wr={(non_esc['realized_pnl'] > 0).mean()*100:.1f}%"
        )
        if len(esc) >= 10 and len(non_esc) >= 10:
            u, p = mannwhitneyu(esc["realized_pnl"], non_esc["realized_pnl"], alternative="two-sided")
            print(f"    Mann-Whitney U p-value (pnl farkı anlamlı mı): {p:.4f}")

        # early_pct'in escalation'ı tahmin gücü (5m aninda, gelecegi bilmeden)
        rho, p2 = spearmanr(sub["early_pct"], sub["escalated"].astype(int))
        print(f"    early_pct vs escalated: rho={rho:+.3f} (p={p2:.4f})")
        rho2, p3 = spearmanr(sub["early_pct"], sub["realized_pnl"])
        print(f"    early_pct vs realized_pnl: rho={rho2:+.3f} (p={p3:.4f})")


def _report_corr(df: pd.DataFrame, col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub[col], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _placebo_early_pct(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """early_pct'i signal_type icinde rastgele karistirir - gercek eslesme
    kaybolunca korelasyon da kaybolmali."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["early_pct"] = out.groupby("signal_type")["early_pct"].transform(
        lambda s: rng.permutation(s.values)
    )
    return out


def main() -> None:
    df = _fetch_5m_signals()
    df = _add_early_pct(df)
    print(f"Toplam 5m RSI_Cross(9,24) sinyali (kapanmış, {_EARLY_BARS*_INTERVAL_MIN}dk sonrası fiyat mevcut): {len(df)}")

    print(f"15m onayı aranıyor (sonraki {_ESCALATION_WINDOW_MIN}dk içinde)...")
    df = _add_escalation_flag(df)

    print("\n=== ESCALATED (15m onayı geldi) vs NON-ESCALATED ===")
    _report(df)

    print("\n=== PLACEBO (early_pct rastgele karıştırıldı) ===")
    _report_corr(_placebo_early_pct(df), "early_pct", "placebo")

    print("\n=== SPLIT-PERIOD (early_pct vs realized_pnl, dönem ikiye bölündü) ===")
    mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", df[df["opened_at"] < mid]),
        ("ikinci_yari", df[df["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)} sinyal) --")
        _report_corr(half_df, "early_pct", half_name)

    print("\n=== SADECE UZUN SUREN ISLEMLER (25dk+ - ortusme riskini eler) ===")
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    dur = pd.read_sql(
        "SELECT symbol, opened_at, duration_minutes FROM signals "
        "WHERE indicators='RSI_Cross(9,24)' AND interval='5m' AND status='closed'",
        conn,
    )
    conn.close()
    df_dur = df.merge(dur, on=["symbol", "opened_at"], how="left")
    long_only = df_dur[df_dur["duration_minutes"] >= _EARLY_BARS * _INTERVAL_MIN]
    print(f"(toplam {len(df_dur)} sinyalden {len(long_only)} tanesi 25dk+ surmus)")
    _report_corr(long_only, "early_pct", "sadece_uzun")


if __name__ == "__main__":
    main()
