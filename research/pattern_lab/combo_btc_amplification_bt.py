"""
BTC dominansı / altcoin amplifikasyonu testi — kullanıcının canlı gözlemi
(24 Tem 2026): "BTC 1 düşerse altcoinler 4 düşüyor" (dominans yükseliş
rejimi). Daha önce sadece HAFTALIK BTC getirisi test edilmiş ve zayıf
korelasyon (-0.016) bulunmuştu ([[project_short_split_period_weakness_24tem]])
— bu, BTC'nin KENDİ hareketinin büyüklüğünün değil, altcoinlerin BTC'ye
göre AMPLİFİYE tepki vermesinin önemli olabileceğini düşündürdü.

Burada İŞLEM BAZINDA, sinyal anındaki BTC'nin KISA VADELİ (1h trailing)
getirisi test ediliyor — [[project_combo_clustering_feature_24tem]]'deki
concurrent_count (kalabalık) ile karşılaştırmalı: ikisi aynı şeyi mi
ölçüyor (redundan) yoksa BTC'nin kendi hareketi EK bilgi mi taşıyor?

Kullanım: python -m research.pattern_lab.combo_btc_amplification_bt
"""

import os

import numpy as np
import pandas as pd
import psycopg2

from config import Config

_FILES = {
    ("RSI_Cross", "Long"): "_cache_replay_long.parquet",
    ("RSI_Cross", "Short"): "_cache_replay_short.parquet",
    ("HA_Cross", "Long"): "_cache_replay_ha_long.parquet",
    ("HA_Cross", "Short"): "_cache_replay_ha_short.parquet",
}
_BASE = os.path.dirname(__file__)
_WINDOW_HOURS = 1.0


def _load() -> pd.DataFrame:
    frames = []
    for (indicator, direction), fname in _FILES.items():
        df = pd.read_parquet(os.path.join(_BASE, fname))
        df["indicator"] = indicator
        df["direction"] = direction
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    return df.sort_values("opened_at").reset_index(drop=True)


def _fetch_btc_1h() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("SELECT bucket, close FROM cagg_5m WHERE symbol='BTCUSDT' ORDER BY bucket ASC")
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    df["bucket"] = pd.to_datetime(df["bucket"])
    return df


def _trailing_return(df_btc: pd.DataFrame, ts: pd.Timestamp, window_hours: float) -> float:
    idx = df_btc["bucket"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return np.nan
    lookback_idx = df_btc["bucket"].searchsorted(ts - pd.Timedelta(hours=window_hours), side="right") - 1
    if lookback_idx < 0:
        return np.nan
    p_now = df_btc["close"].iloc[idx]
    p_before = df_btc["close"].iloc[lookback_idx]
    if p_before == 0:
        return np.nan
    return (p_now - p_before) / p_before * 100.0


def main() -> None:
    df = _load()
    btc = _fetch_btc_1h()
    print(f"Toplam n={len(df)} işlem, BTC serisi n={len(btc)}\n")

    df["btc_ret_1h"] = [
        _trailing_return(btc, ts, _WINDOW_HOURS) for ts in df["opened_at"]
    ]
    df = df.dropna(subset=["btc_ret_1h"])

    for direction in ["Short", "Long"]:
        sub = df[df["direction"] == direction]
        corr = np.corrcoef(sub["btc_ret_1h"], sub["pnl_usd"])[0, 1]
        print(f"{direction} (n={len(sub)}): korelasyon(BTC 1h getiri, pnl_usd) = {corr:+.3f}")

        q = pd.qcut(sub["btc_ret_1h"], 4, duplicates="drop")
        g = sub.groupby(q, observed=True).agg(
            n=("pnl_usd", "size"), wr=("pnl_usd", lambda x: round((x > 0).mean() * 100, 1)),
            toplam=("pnl_usd", "sum"), ort=("pnl_usd", "mean"),
        )
        print(g)
        print()

    # concurrent_count ile karşılaştırma (Short için, en güçlü sinyal orada)
    print("=" * 78)
    print("KARŞILAŞTIRMA: BTC 1h getiri vs concurrent_count (Short, korelasyon)")
    print("=" * 78)
    short = df[df["direction"] == "Short"].sort_values("opened_at").reset_index(drop=True)
    times = short["opened_at"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    window_ns = 4 * 3600 * 1e9
    counts = np.zeros(len(times), dtype=int)
    left = 0
    for i in range(len(times)):
        while times[i] - times[left] > window_ns:
            left += 1
        counts[i] = i - left
    short["concurrent_count"] = counts
    corr_cc_pnl = np.corrcoef(short["concurrent_count"], short["pnl_usd"])[0, 1]
    corr_btc_cc = np.corrcoef(short["btc_ret_1h"], short["concurrent_count"])[0, 1]
    print(f"  concurrent_count vs pnl_usd: {corr_cc_pnl:+.3f} (referans, önceki bulgu)")
    print(f"  BTC 1h getiri vs concurrent_count: {corr_btc_cc:+.3f} (redundan mı, bağımsız mı?)")


if __name__ == "__main__":
    main()
