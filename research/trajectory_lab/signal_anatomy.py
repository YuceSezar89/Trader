"""
Trajectory Lab — "Sinyal Mumu Anatomisi": bileşenler ARASI redundancy
analizi (bkz. CONTEXT_LAB_STATUS.md, 7 Ağustos). Amaç yeni bir
formül/indikatör İCAT ETMEK DEĞİL — sinyal barının anatomisini oluşturan
bileşenlerden hangisinin bağımsız bilgi taşıdığını, hangisinin redundant
olduğunu görmek. Outcome (y) korelasyonu sadece yan bilgi — asıl analiz
bileşenler ARASI korelasyon matrisi.

Üç kategori:
  1. Mutlak/geometrik: body_size, range_size, body_pct(=body/range),
     close_pos, üst/alt fitil, volume_level, buy_pct
  2. Bar-içi sıralama: lag (intrabar_lag.py ile aynı yöntem, momentum
     dakikası vs hacim dakikası)
  3. Sembol-içi sıra dışılık: body_size_pct, range_size_pct,
     volume_level_pct (percentile-rank, stage2_pctrank.py ile aynı yöntem)

3 ailede AYRI AYRI çalıştırılır (havuzlanmaz) — redundancy yapısının
aileye özgü mü yoksa genel mi olduğunu görmek için (kullanıcı talimatı,
7 Ağustos): HA_Cross_Long (ana referans), RSI_Cross_Long (çapraz
doğrulama), Supertrend_Long (üçüncü/farklı karakterde aile).

Kullanım:
    python -m research.trajectory_lab.signal_anatomy
    python -m research.trajectory_lab.signal_anatomy --indicator HA_Cross --direction Long
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine, get_session  # noqa: E402
from research.trajectory_lab.corpus_builder import _fetch_case_signals  # noqa: E402

_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_BATCH_SIZE = 300
MIN_SIGNALS_PER_SYMBOL = 30

_FAMILIES = [
    ("HA_Cross_Long (ana referans)", "HA_Cross", "Long"),
    ("RSI_Cross_Long (çapraz doğrulama)", "RSI_Cross(9,24)", "Long"),
    ("Supertrend_Long (üçüncü/farklı karakter)", "Supertrend(10,3.0)", "Long"),
]

# aynı kavramın farklı temsilleri — ham büyüklük vs kendi percentile-rank'i
_SAME_CONCEPT_PAIRS = [
    ("body_size", "body_size_pct"),
    ("range_size", "range_size_pct"),
    ("volume_level", "volume_level_pct"),
]
# geometrik oranların hangi ham büyüklüklerle ilişkisine özellikle bakılacak
_RATIO_VS_RAW = [
    ("body_pct", "body_size"),
    ("body_pct", "range_size"),
    ("close_pos", "body_size"),
    ("close_pos", "range_size"),
    ("close_pos", "upper_wick_pct"),
    ("close_pos", "lower_wick_pct"),
    ("upper_wick_pct", "range_size"),
    ("lower_wick_pct", "range_size"),
]

ALL_COLS = [
    "body_size", "body_size_pct", "range_size", "range_size_pct",
    "volume_level", "volume_level_pct", "body_pct", "close_pos",
    "upper_wick_pct", "lower_wick_pct", "buy_pct", "lag",
]


async def _fetch_subbars_batch(batch: pd.DataFrame) -> pd.DataFrame:
    starts = batch["opened_at"].tolist()
    ends = [
        row["opened_at"] + pd.Timedelta(minutes=_INTERVAL_MINUTES[row["interval"]])
        for _, row in batch.iterrows()
    ]
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT sw.signal_id, p.timestamp, p.open, p.high, p.low, p.close,
                       p.volume, p.buy_volume, p.sell_volume
                FROM unnest(
                    CAST(:signal_ids AS int[]), CAST(:symbols AS text[]),
                    CAST(:starts AS timestamp[]), CAST(:ends AS timestamp[])
                ) AS sw(signal_id, symbol, start_ts, end_ts)
                JOIN price_data p
                  ON p.symbol = sw.symbol AND p.interval = '1m'
                 AND p.timestamp >= sw.start_ts AND p.timestamp < sw.end_ts
                ORDER BY sw.signal_id, p.timestamp
                """
            ),
            {
                "signal_ids": batch["signal_id"].tolist(),
                "symbols": batch["symbol"].tolist(),
                "starts": starts,
                "ends": ends,
            },
        )
        rows = result.all()
    return pd.DataFrame(
        rows,
        columns=[
            "signal_id", "timestamp", "open", "high", "low", "close",
            "volume", "buy_volume", "sell_volume",
        ],
    )


async def _fetch_subbars(signals_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    n = len(signals_df)
    for i in range(0, n, _BATCH_SIZE):
        batch = signals_df.iloc[i : i + _BATCH_SIZE]
        frames.append(await _fetch_subbars_batch(batch))
        print(f"  ... {min(i + _BATCH_SIZE, n)}/{n} sinyal işlendi", end="\r")
    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_anatomy(signals_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = sub_df.copy()
    sub_df["body_size"] = (sub_df["close"] - sub_df["open"]).abs()

    rows = []
    for signal_id, grp in sub_df.groupby("signal_id"):
        expected_n = _INTERVAL_MINUTES[
            signals_df.loc[signals_df["signal_id"] == signal_id, "interval"].iloc[0]
        ]
        if len(grp) < expected_n:
            continue
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        momentum_min = int(grp["body_size"].idxmax())
        volume_min = int(grp["volume"].idxmax())
        lag = volume_min - momentum_min

        o, h, low_, c = grp["open"].iloc[0], grp["high"].max(), grp["low"].min(), grp["close"].iloc[-1]
        vol = grp["volume"].sum()
        bv, sv = grp["buy_volume"].sum(), grp["sell_volume"].sum()

        rng = h - low_
        body = abs(c - o)
        if rng <= 0:
            continue
        body_pct = body / rng * 100.0
        close_pos = (c - low_) / rng * 100.0
        upper_wick_pct = (h - max(o, c)) / rng * 100.0
        lower_wick_pct = (min(o, c) - low_) / rng * 100.0
        total_v = bv + sv
        buy_pct = (bv / total_v * 100.0) if total_v > 0 else np.nan

        rows.append(
            {
                "signal_id": signal_id,
                "symbol": signals_df.loc[signals_df["signal_id"] == signal_id, "symbol"].iloc[0],
                "body_size": body,
                "range_size": rng,
                "volume_level": vol,
                "body_pct": body_pct,
                "close_pos": close_pos,
                "upper_wick_pct": upper_wick_pct,
                "lower_wick_pct": lower_wick_pct,
                "buy_pct": buy_pct,
                "lag": lag,
            }
        )
    return pd.DataFrame(rows)


def _add_symbol_pctrank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank kolonlarını sadece MIN_SIGNALS_PER_SYMBOL şartını
    sağlayan sembollerde doldurur — şartı sağlamayan semboller SATIR olarak
    ATILMAZ (sadece pct kolonları NaN kalır), böylece az sinyalli aileler
    (Supertrend gibi) diğer (percentile-rank olmayan) bileşenler için tüm
    örneklemi korur. corr() NaN'ları pairwise-complete olarak yok sayar."""
    df = df.copy()
    sizes = df.groupby("symbol").size()
    valid = set(sizes[sizes >= MIN_SIGNALS_PER_SYMBOL].index)
    is_valid = df["symbol"].isin(valid)
    for base in ("body_size", "range_size", "volume_level"):
        pct_col = f"{base}_pct"
        df[pct_col] = np.nan
        df.loc[is_valid, pct_col] = df.loc[is_valid].groupby("symbol")[base].rank(pct=True) * 100.0
    print(
        f"  (percentile-rank için yeterli sembol-içi örneklemi olan sinyal: "
        f"{int(is_valid.sum())}/{len(df)})"
    )
    return df


def _partial_corr(df: pd.DataFrame, y: str, x1: str, x2: str) -> float:
    """Birinci-derece partial correlation (kapalı-form): r(y,x1|x2).
    y binary olsa da bu SADECE korelasyonların birbirinden ne kadar
    bağımsız olduğunu ölçmek için kullanılıyor — 'tahmin gücü' olarak
    yorumlanmıyor (kullanıcı notu, 7 Ağustos)."""
    r_y1 = df[y].corr(df[x1])
    r_y2 = df[y].corr(df[x2])
    r_12 = df[x1].corr(df[x2])
    denom = np.sqrt((1 - r_y2**2) * (1 - r_12**2))
    return (r_y1 - r_y2 * r_12) / denom if denom != 0 else np.nan


def _print_partial_corr(df: pd.DataFrame) -> None:
    print("--- PARTIAL CORRELATION: close_pos vs buy_pct BAĞIMSIZLIĞI (Stage 2 DEĞİL, sadece bağımsızlık ölçümü) ---")
    r_y_cp = df["y"].corr(df["close_pos"])
    r_y_bp = df["y"].corr(df["buy_pct"])
    r_cp_bp = df["close_pos"].corr(df["buy_pct"])
    p_cp_given_bp = _partial_corr(df, "y", "close_pos", "buy_pct")
    p_bp_given_cp = _partial_corr(df, "y", "buy_pct", "close_pos")
    print(f"  corr(y, close_pos)                    = {r_y_cp:+.4f}")
    print(f"  corr(y, buy_pct)                       = {r_y_bp:+.4f}")
    print(f"  corr(close_pos, buy_pct)               = {r_cp_bp:+.4f}")
    print(f"  partial corr(y, close_pos | buy_pct)   = {p_cp_given_bp:+.4f}")
    print(f"  partial corr(y, buy_pct | close_pos)   = {p_bp_given_cp:+.4f}")
    drop_cp = 1 - abs(p_cp_given_bp) / abs(r_y_cp) if r_y_cp != 0 else np.nan
    drop_bp = 1 - abs(p_bp_given_cp) / abs(r_y_bp) if r_y_bp != 0 else np.nan
    print(f"  close_pos'un buy_pct kontrolüyle düşüşü: %{drop_cp*100:.1f}")
    print(f"  buy_pct'in close_pos kontrolüyle düşüşü: %{drop_bp*100:.1f}")
    print()


def _print_family_report(label: str, df: pd.DataFrame) -> None:
    print(f"\n===== Sinyal Mumu Anatomisi — {label} =====")
    print(f"Analiz edilen sinyal sayısı: {len(df)}")
    print(f"Sınıf dağılımı: {df['y'].value_counts().to_dict()}\n")

    corr = df[ALL_COLS].corr().round(2)
    print("--- TAM KORELASYON MATRİSİ (bileşenler arası) ---")
    print(corr)
    print()

    print("--- AYNI KAVRAMIN FARKLI TEMSİLLERİ (ham vs kendi percentile-rank'i) ---")
    for a, b in _SAME_CONCEPT_PAIRS:
        r = df[a].corr(df[b])
        note = "YÜKSEK redundancy" if abs(r) >= 0.7 else ("orta" if abs(r) >= 0.4 else "DÜŞÜK — bağımsız bilgi olabilir")
        print(f"  corr({a}, {b}) = {r:+.3f}  [{note}]")
    print()

    print("--- GEOMETRİK ORANLARIN HAM BÜYÜKLÜKLERLE İLİŞKİSİ ---")
    for a, b in _RATIO_VS_RAW:
        r = df[a].corr(df[b])
        print(f"  corr({a}, {b}) = {r:+.3f}")
    print()

    print("--- OUTCOME (y) KORELASYONU — SADECE YAN BİLGİ, Stage 2 DEĞİL ---")
    for col in ALL_COLS:
        r = df[col].corr(df["y"])
        print(f"  corr({col}, y) = {r:+.3f}")
    print()

    _print_partial_corr(df)


_ANATOMY_CACHE_DIR = "research/trajectory_corpus"


def _cache_path(indicator: str, direction: str) -> str:
    safe = indicator.replace("(", "_").replace(")", "").replace(",", "_")
    return os.path.join(_ANATOMY_CACHE_DIR, f"anatomy_{safe}_{direction}.parquet")


async def run_family(label: str, indicator: str, direction: str, use_cache: bool = True) -> None:
    cache_path = _cache_path(indicator, direction)
    if use_cache and os.path.exists(cache_path):
        anatomy = pd.read_parquet(cache_path)
        print(f"\n{label}: önbellekten yüklendi ({cache_path}, {len(anatomy)} sinyal)")
    else:
        signals_df = await _fetch_case_signals(indicator, direction)
        signals_df = signals_df[signals_df["outcome"].isin(["winner", "loser"])]
        n_1m = int((signals_df["interval"] == "1m").sum())
        signals_df = signals_df[signals_df["interval"] != "1m"].reset_index(drop=True)
        print(f"\n{label}: {len(signals_df)} kapanmış winner/loser sinyali (1m atlandı: {n_1m})")

        sub_df = await _fetch_subbars(signals_df)
        anatomy = _build_anatomy(signals_df, sub_df)
        anatomy = _add_symbol_pctrank(anatomy)

        outcome_map = signals_df.set_index("signal_id")["outcome"]
        anatomy["y"] = anatomy["signal_id"].map(outcome_map).eq("winner").astype(int)

        os.makedirs(_ANATOMY_CACHE_DIR, exist_ok=True)
        anatomy.to_parquet(cache_path, index=False)
        print(f"  -> önbelleğe yazıldı: {cache_path}")

    _print_family_report(label, anatomy)


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default=None)
    parser.add_argument("--direction", default="Long", choices=["Long", "Short"])
    args = parser.parse_args()
    try:
        if args.indicator:
            await run_family(f"{args.indicator} {args.direction}", args.indicator, args.direction)
        else:
            for label, indicator, direction in _FAMILIES:
                await run_family(label, indicator, direction)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
