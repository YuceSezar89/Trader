"""
Adım 1 — Korpus: veriyi bir kez çek, dondur, bir daha DB'ye gitme.

Katman A: 20 vaka + 20 likidite-eşli kontrol × CORPUS_DAYS × 5m OHLCV+taker
          (Binance REST, 1 çağrı/sn — taker alanları tarihsel barlarda da gelir)
Katman B: tüm evren × CORPUS_DAYS × 5m close (cagg_5m — sıralama/ayrışma için)

Zaman ekseni: proje geleneği, naif İstanbul zamanı ("ts" kolonu).
Çapa (anchor): kurulum anında bir kez dondurulur, meta.json'a yazılır.
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from config import Config
from research.pattern_lab import config as C

random.seed(C.SEED)


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def freeze_anchor() -> datetime:
    """Çapa = şimdi, 5 dakikaya aşağı yuvarlanmış (naif İstanbul)."""
    now = datetime.now()
    return now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)


def universe_stats(anchor: datetime) -> pd.DataFrame:
    """Evren: seçim penceresinde getiri + bar kapsamı + medyan günlük ciro."""
    t_start = anchor - timedelta(days=C.SELECTION_DAYS)
    q = """
        SELECT symbol,
               (array_agg(close ORDER BY bucket))[1]                 AS first_close,
               (array_agg(close ORDER BY bucket DESC))[1]            AS last_close,
               COUNT(*)                                              AS n_bars,
               percentile_cont(0.5) WITHIN GROUP (
                   ORDER BY day_quote)                               AS med_daily_quote
        FROM (
            SELECT symbol, bucket, close,
                   SUM(close * volume) OVER (PARTITION BY symbol, date_trunc('day', bucket)) AS day_quote
            FROM cagg_1h
            WHERE bucket >= %s AND bucket < %s
        ) x
        GROUP BY symbol
    """
    with _conn() as conn:
        df = pd.read_sql(q, conn, params=(t_start, anchor))
    df["ret_pct"] = (df["last_close"] / df["first_close"] - 1) * 100
    expected_bars = C.SELECTION_DAYS * 24
    df["coverage"] = df["n_bars"] / expected_bars
    df = df[(df["first_close"] > 0) & (df["coverage"] >= C.MIN_BAR_COVERAGE)].copy()
    df["vol_decile"] = pd.qcut(df["med_daily_quote"], C.VOL_DECILES, labels=False)
    return df.sort_values("ret_pct", ascending=False).reset_index(drop=True)


def select_groups(stats: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    """Vakalar: getiri top-N. Kontroller: orta dilim ∩ vakanın hacim ondalığı.

    Döner: (vaka listesi, [(vaka, eşleşen kontrol), ...])
    Ondalıkta aday kalmazsa komşu ondalığa taşılır (kural burada, yazılı).
    """
    cases = stats.head(C.N_CASES)
    lo = stats["ret_pct"].quantile(C.MID_QUANTILE[0])
    hi = stats["ret_pct"].quantile(C.MID_QUANTILE[1])
    pool = stats[(stats["ret_pct"] >= lo) & (stats["ret_pct"] <= hi)]
    pool = pool[~pool["symbol"].isin(cases["symbol"])]

    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, case in cases.iterrows():
        dec = case["vol_decile"]
        for widen in range(C.VOL_DECILES):
            cand = pool[
                (pool["vol_decile"].sub(dec).abs() <= widen)
                & (~pool["symbol"].isin(used))
            ]
            if not cand.empty:
                pick = cand.sample(1, random_state=C.SEED + len(pairs)).iloc[0]
                used.add(pick["symbol"])
                pairs.append((case["symbol"], pick["symbol"]))
                break
        else:
            raise RuntimeError(f"{case['symbol']} için kontrol adayı bulunamadı")
    return cases["symbol"].tolist(), pairs


def fetch_layer_a(symbols: list[str], anchor: datetime) -> pd.DataFrame:
    """40 sembol × CORPUS_DAYS × 5m OHLCV+taker — DB'den (price_data 1m → 5m resample).
    REST'e HİÇ gitmez: OHLCV price_data'da aylarca geriye gidiyor, buy_volume ise
    şema düzeltmesinden beri (~9 gün) mevcut — bu, t0'ların 48h bağlam penceresinin
    çoğunu zaten kapsıyor (4 Tem gece dersi: REST gereksizdi, bkz. memory)."""
    t_start = anchor - timedelta(days=C.CORPUS_DAYS)
    with _conn() as conn:
        q = """
            SELECT symbol, timestamp AS ts, open, high, low, close, volume, buy_volume, sell_volume
            FROM price_data
            WHERE interval = '1m' AND symbol = ANY(%s)
              AND timestamp >= %s AND timestamp < %s
            ORDER BY symbol, timestamp
        """
        raw = pd.read_sql(q, conn, params=(symbols, t_start, anchor))

    def _sum_or_nan(s: pd.Series) -> float:
        return float(s.sum()) if s.notna().any() else float("nan")

    frames = []
    for sym, g in raw.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        agg = g.resample("5min").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"),
            buy_volume=("buy_volume", _sum_or_nan),
            sell_volume=("sell_volume", _sum_or_nan),
        ).dropna(subset=["open"])
        agg["symbol"] = sym
        frames.append(agg.reset_index())
        print(f"  {sym}: {len(agg)} bar (DB)")

    out = pd.concat(frames, ignore_index=True)
    cols = ["symbol", "ts", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    return out[cols].sort_values(["symbol", "ts"]).reset_index(drop=True)


def fetch_layer_b(symbols: list[str], anchor: datetime) -> pd.DataFrame:
    """Evren × CORPUS_DAYS × 5m close — cagg_5m (sıralama için)."""
    t_start = anchor - timedelta(days=C.CORPUS_DAYS)
    q = """
        SELECT symbol, bucket AS ts, close
        FROM cagg_5m
        WHERE bucket >= %s AND bucket < %s AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    with _conn() as conn:
        return pd.read_sql(q, conn, params=(t_start, anchor, symbols))


def build() -> None:
    os.makedirs(C.CORPUS_DIR, exist_ok=True)
    anchor = freeze_anchor()
    print(f"Çapa donduruldu: {anchor}")

    stats = universe_stats(anchor)
    print(f"Evren: {len(stats)} sembol (kapsam ≥ {C.MIN_BAR_COVERAGE:.0%})")

    cases, pairs = select_groups(stats)
    controls = [c for _, c in pairs]
    print(f"Vaka: {len(cases)} | Kontrol: {len(controls)}")

    layer_b = fetch_layer_b(stats["symbol"].tolist(), anchor)
    layer_b.to_parquet(f"{C.CORPUS_DIR}/layer_b_universe.parquet", index=False)
    print(f"Katman B: {len(layer_b):,} satır ({layer_b['symbol'].nunique()} sembol) → parquet")

    layer_a = fetch_layer_a(cases + controls, anchor)
    layer_a.to_parquet(f"{C.CORPUS_DIR}/layer_a_detail.parquet", index=False)
    taker_cov = layer_a["buy_volume"].notna().mean()
    print(f"Katman A: {len(layer_a):,} satır, taker kapsamı {taker_cov:.1%} → parquet")

    meta = {
        "anchor": anchor.isoformat(),
        "built_at": datetime.now().isoformat(),
        "cases": cases,
        "pairs": pairs,
        "universe_size": len(stats),
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    with open(f"{C.CORPUS_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    stats.to_parquet(f"{C.CORPUS_DIR}/universe_stats.parquet", index=False)
    print("meta.json + universe_stats yazıldı. Korpus donduruldu.")


if __name__ == "__main__":
    build()
