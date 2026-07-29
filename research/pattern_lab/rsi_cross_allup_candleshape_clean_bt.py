"""
all_up + mum-şekli — TEMİZ yöntemle yeniden doğrulama (21 Tem 2026).

rsi_cross_allup_live_formula_bt.py'nin BİREBİR AYNI bileşen formülü
(utils/vpmv.py::compute_components — yönlü buy/sell hacmi, RSI diff
momentum, canlı sistemle uyumlu) — SADECE hedef değişti:
  ESKİ: realized_pnl (status='closed' zorunlu, kirli/çıkış-mekanizması-bağımlı)
  YENİ: sabit 24-bar ileri getiri (status önemsiz, TÜM sinyaller)

all_up + mum-şekli, bugün çürüyen 7 tek-değişkenli statik-eşik filtresinden
YAPISAL olarak farklı: 4 değişkenin (hacim/momentum/volatilite/fiyat)
ÖNCEKİ AYNI SEMBOL sinyaline göre GÖRECELİ deltası + kategori birleşimi —
sabit global eşik yerine kendi geçmişine göre hareketli referans kullanıyor.

Kullanım: python -m research.pattern_lab.rsi_cross_allup_candleshape_clean_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import truncate_after_gap
from utils.vpmv import compute_components

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_BARS_NEEDED = 220
_FORWARD_BARS = 24
_GAP_HOURS_THRESHOLD = 200
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")
DIRECTIONS = ["Long", "Short"]


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _bad_symbols(cur) -> set[str]:
    cur.execute(
        """
        WITH gaps AS (
            SELECT symbol, EXTRACT(EPOCH FROM (curr_ts-prev_ts))/3600 AS saat
            FROM (
                SELECT symbol, timestamp AS curr_ts,
                       LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                FROM price_data WHERE interval='1m'
            ) t
            WHERE prev_ts IS NOT NULL
        )
        SELECT DISTINCT symbol FROM gaps WHERE saat > %s
        """,
        (_GAP_HOURS_THRESHOLD,),
    )
    return {r[0] for r in cur.fetchall()}


def _fetch_signals(cur, indicator: str, direction: str, exclude_symbols: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, opened_at, open_price
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY symbol, opened_at
        """,
        (indicator, direction),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id", "symbol", "interval", "opened_at", "open_price"])
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)].reset_index(drop=True)
    return df


def _fetch_entry_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    iv_min = _INTERVAL_MIN.get(interval, 1)
    lookback_minutes = iv_min * (_BARS_NEEDED + 5) + 5
    cur.execute(
        "SELECT timestamp AS open_time, open, high, low, close, volume, buy_volume, sell_volume "
        "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
        "ORDER BY timestamp DESC LIMIT %s",
        (symbol, opened_at, lookback_minutes),
    )
    rows = cur.fetchall()
    if not rows or len(rows) < 60:
        return None
    df1m = pd.DataFrame(
        rows, columns=["open_time", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    )
    df1m = df1m.iloc[::-1].reset_index(drop=True)
    df1m = truncate_after_gap(df1m)
    if len(df1m) < 60:
        return None

    if interval == "1m":
        df = df1m
    else:
        d = df1m.copy()
        d["ts"] = pd.to_datetime(d["open_time"], unit="ms")
        d = d.set_index("ts")
        rule = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[interval]
        df = (
            d.resample(rule, label="left", closed="left", origin="epoch")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                  "volume": "sum", "buy_volume": "sum", "sell_volume": "sum"})
            .dropna()
            .reset_index(drop=True)
        )
    if len(df) < 60:
        return None
    return df.tail(_BARS_NEEDED).reset_index(drop=True)


def _classify_candle(last: pd.Series) -> str:
    rng = last["high"] - last["low"]
    if rng <= 0:
        return "belirsiz"
    upper = max(last["open"], last["close"])
    lower = min(last["open"], last["close"])
    body = abs(last["close"] - last["open"]) / rng * 100
    upper_wick = (last["high"] - upper) / rng * 100
    lower_wick = (lower - last["low"]) / rng * 100
    parts = {"govde": body, "ust_fitil": upper_wick, "alt_fitil": lower_wick}
    return max(parts, key=parts.get)


def _components(df: pd.DataFrame, direction: str) -> dict | None:
    try:
        vol_s, mom_s, vlt_s, prc_s = compute_components(df, direction)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    vals = (vol_s, mom_s, vlt_s, prc_s)
    if any(pd.isna(v) for v in vals):
        return None
    return {"vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s, "kategori": _classify_candle(df.iloc[-1])}


def _fetch_forward_price(cur, symbol: str, interval: str, after, n_bars: int) -> float | None:
    if interval == "1m":
        cur.execute(
            "SELECT close FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp >= %s "
            "ORDER BY timestamp ASC LIMIT 1 OFFSET %s",
            (symbol, after, n_bars - 1),
        )
    else:
        table = _CAGG[interval]
        cur.execute(
            f"SELECT close FROM {table} WHERE symbol=%s AND bucket >= %s ORDER BY bucket ASC LIMIT 1 OFFSET %s",
            (symbol, after, n_bars - 1),
        )
    row = cur.fetchone()
    return float(row[0]) if row else None


def _collect(conn, indicator: str, direction: str, bad_symbols: set[str]) -> pd.DataFrame:
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, indicator, direction, bad_symbols)
    if sig_df.empty:
        return sig_df
    side = 1.0 if direction == "Long" else -1.0
    records = []
    for i, row in enumerate(sig_df.itertuples()):
        df = _fetch_entry_bars(cur, row.symbol, row.interval, row.opened_at)
        if df is None:
            continue
        comp = _components(df, direction)
        if comp is None:
            continue
        fwd_price = _fetch_forward_price(cur, row.symbol, row.interval, row.opened_at, _FORWARD_BARS)
        if fwd_price is None:
            continue
        fwd_ret = (fwd_price - row.open_price) / row.open_price * 100.0 * side
        records.append({
            "symbol": row.symbol, "interval": row.interval,
            "opened_at": row.opened_at, "fwd_ret": fwd_ret, **comp,
        })
        if (i + 1) % 2000 == 0:
            print(f"    ... {i+1}/{len(sig_df)} ({direction})")
    return pd.DataFrame(records)


def _add_all_up(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        df[f"d_{col}"] = df.groupby("symbol")[col].diff()
    df = df.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    df["all_up"] = (df["d_vol"] > 0) & (df["d_mom"] > 0) & (df["d_volat"] > 0) & (df["d_price"] > 0)

    # 21 Tem — do_open_streak'teki "3 ardışık" mantığının all_up'a uyarlanmışı:
    # tek adımlık artış (N-1 -> N) yerine, AYNI SEMBOLÜN art arda 3 sinyali
    # boyunca 4 metriğin de kesintisiz arttığı (N-2<N-1<N) daha sıkı bir bayrak.
    df = df.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        prev1 = df.groupby("symbol")[col].shift(1)
        prev2 = df.groupby("symbol")[col].shift(2)
        df[f"streak3_{col}"] = (prev2 < prev1) & (prev1 < df[col])
    df["all_up_streak3"] = (
        df["streak3_vol"] & df["streak3_mom"] & df["streak3_volat"] & df["streak3_price"]
    )
    return df


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _report(name: str, sub: pd.DataFrame) -> None:
    s = _stats(sub["fwd_ret"].to_numpy())
    print(f"  {name:30} n={s.get('n',0):>6}  WR%={s.get('wr','-'):>6}  ort%={s.get('ort_%','-'):>7}  PF={s.get('pf','-')}")


def _deep_validate(label: str, group: pd.DataFrame, rest: pd.DataFrame, full_df: pd.DataFrame) -> None:
    """Placebo (etiket karıştırma, PF farkı) + kronolojik yarı-yarı + rejim split."""
    print(f"\n  -- derin doğrulama: {label} (n={len(group)}) vs geri kalan (n={len(rest)}) --")
    _report(label, group)
    _report("geri kalanı (hepsi)", rest)

    pf_grp = _stats(group["fwd_ret"].to_numpy()).get("pf", 0) or 0
    pf_rest = _stats(rest["fwd_ret"].to_numpy()).get("pf", 0) or 0
    real_gap = pf_grp - pf_rest

    rng = np.random.default_rng(42)
    n_true = len(group)
    fwd = full_df["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        perm = rng.permutation(len(full_df))
        fake = np.zeros(len(full_df), dtype=bool)
        fake[perm[:n_true]] = True
        pf_t = _stats(fwd[fake]).get("pf", 0) or 0
        pf_f = _stats(fwd[~fake]).get("pf", 0) or 0
        if abs(pf_t - pf_f) >= abs(real_gap):
            count_ge += 1
    print(f"    placebo: gerçek PF farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(group) >= 40:
        g_sorted = group.sort_values("opened_at")
        mid = g_sorted["opened_at"].iloc[len(g_sorted)//2]
        first = g_sorted[g_sorted["opened_at"] < mid]
        second = g_sorted[g_sorted["opened_at"] >= mid]
        print(f"    split-period(kronolojik yarı-yarı): ilk yarı PF={_stats(first['fwd_ret'].to_numpy()).get('pf',0):>6} "
              f"(n={len(first)})  ikinci yarı PF={_stats(second['fwd_ret'].to_numpy()).get('pf',0):>6} (n={len(second)})")
    else:
        print("    split-period: yetersiz örnek")

    crash = group[group["opened_at"] < _REGIME_SPLIT]
    recovery = group[group["opened_at"] >= _REGIME_SPLIT]
    for rl, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 50:
            print(f"    Split — {rl} (n={len(sub)}): {_stats(sub['fwd_ret'].to_numpy())}")
        else:
            print(f"    Split — {rl}: yetersiz örnek (n={len(sub)})")


def _run_one(conn, indicator: str, direction: str, bad_symbols: set[str]) -> None:
    print(f"\n[{direction}] sinyaller toplanıyor...")
    raw = _collect(conn, indicator, direction, bad_symbols)
    if raw.empty:
        print(f"\n{indicator} — {direction}: veri yok.")
        return
    df = _add_all_up(raw)
    print(f"\n{'='*78}\n{indicator} — {direction} (TEMİZ hedef, canlı formül)  (n={len(df):,})\n{'='*78}")

    _report("baseline (tümü)", df)
    _report("all_up=True", df[df["all_up"]])
    _report("all_up=False", df[~df["all_up"]])

    print("\n  -- mum-şekli (tüm veri) --")
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        _report(f"kategori={kat}", df[df["kategori"] == kat])

    all_up_df = df[df["all_up"]]
    print("\n  -- kombinasyon: all_up=True İÇİNDE mum-şekli --")
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        sub = all_up_df[all_up_df["kategori"] == kat]
        _report(f"all_up=T + {kat}", sub)

    if len(all_up_df) < 60:
        return

    # 1) DÜZ all_up=True — mining YOK, en büyük/en az önyargılı örnek
    _deep_validate("all_up=True", all_up_df, df[~df["all_up"]], df)

    # 1b) TF bazlı kırılım — havuzlanmış sonuç hangi TF'lerin ortalamasıysa
    # onu maskeleyebilir; canlıya alınacaksa hangi TF gerçekten taşıyor bilmek lazım
    print("\n  -- all_up=True TF bazlı kırılım --")
    for interval in sorted(df["interval"].unique()):
        tf_all_up = all_up_df[all_up_df["interval"] == interval]
        tf_rest = df[(df["interval"] == interval) & (~df["all_up"])]
        if len(tf_all_up) < 100:
            print(f"    [{interval}] yetersiz örnek (n={len(tf_all_up)}), atlanıyor")
            continue
        _deep_validate(f"all_up=True [{interval}]", tf_all_up, tf_rest, df[df["interval"] == interval])

    # 2) En iyi kategori-kombosu — DİKKAT: 3 kategoriden "en iyisi" seçildiği
    # için mining riski taşır, placebo/split buna göre daha temkinli okunmalı
    cat_pf = {}
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        sub = all_up_df[all_up_df["kategori"] == kat]
        if len(sub) >= 20:
            cat_pf[kat] = _stats(sub["fwd_ret"].to_numpy()).get("pf", 0) or 0
    if not cat_pf:
        return
    best_kat = max(cat_pf, key=cat_pf.get)
    best = all_up_df[all_up_df["kategori"] == best_kat]
    rest = df[~((df["all_up"]) & (df["kategori"] == best_kat))]
    _deep_validate(f"all_up=T + {best_kat} (en iyi kombo, mining riskli)", best, rest, df)

    # 3) all_up_streak3 — do_open_streak'teki "3 ardışık" mantığının all_up'a
    # uyarlanmışı: tek adım yerine art arda 3 sinyal boyunca kesintisiz artış
    streak3_df = df[df["all_up_streak3"]]
    print(f"\n  -- all_up_streak3 (3 sinyal boyunca kesintisiz artış, n={len(streak3_df)}) --")
    if len(streak3_df) >= 30:
        _deep_validate("all_up_streak3=True", streak3_df, df[~df["all_up_streak3"]], df)
    else:
        print("    örneklem çok küçük, atlanıyor.")


def main() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")
    for direction in DIRECTIONS:
        _run_one(conn, "RSI_Cross(9,24)", direction, bad)
    conn.close()


if __name__ == "__main__":
    main()
