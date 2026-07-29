"""
all_up + mum-şekli testini, CANLI sistemin GERÇEK formülüyle (utils/vpmv.py::
compute_components) tekrar eder. rsi_cross_allup_candleshape_bt.py'deki
_vpmv_components ile İKİ farkı vardı:

  - Hacim: research script toplam `volume` kullanıyordu; canlı sistem
    Long'da buy_volume, Short'ta sell_volume (yönlü taker hacmi) kullanıyor.
  - Momentum: research script RSI-50 (seviye) kullanıyordu; canlı sistem
    RSI'nin bar-bar DEĞİŞİMİNİ (rsi.diff()) kullanıyor.

cagg_5m/15m/1h/4h tablolarında buy_volume/sell_volume YOK — bu yüzden ham 1m
price_data'dan (buy_volume/sell_volume dahil) kendimiz interval'a agregasyon
yapıyoruz, live_data_manager'daki toplama mantığıyla birebir (sum ile).

Kullanım: python -m research.pattern_lab.rsi_cross_allup_live_formula_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import truncate_after_gap
from research.pattern_lab.vol_exhaustion_bt import _stats
from utils.vpmv import compute_components

_GAP_HOURS_THRESHOLD = 200
_BARS_NEEDED = 220
_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
DIRECTIONS = ["Long", "Short"]


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
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
        SELECT id, symbol, interval, opened_at, realized_pnl
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY symbol, opened_at
        """,
        (indicator, direction),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id", "symbol", "interval", "opened_at", "realized_pnl"])
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)].reset_index(drop=True)
    return df


def _fetch_entry_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    """opened_at'e kadarki son ~220 bar'ı, buy_volume/sell_volume DAHİL, canlıyla
    aynı şekilde 1m'den agrege ederek döner."""
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
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "buy_volume": "sum",
                    "sell_volume": "sum",
                }
            )
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
    return {
        "vol": vol_s,
        "mom": mom_s,
        "volat": vlt_s,
        "price": prc_s,
        "kategori": _classify_candle(df.iloc[-1]),
    }


def _collect(conn, indicator: str, direction: str, bad_symbols: set[str]) -> pd.DataFrame:
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, indicator, direction, bad_symbols)
    if sig_df.empty:
        return sig_df
    records = []
    for row in sig_df.itertuples():
        df = _fetch_entry_bars(cur, row.symbol, row.interval, row.opened_at)
        if df is None:
            continue
        comp = _components(df, direction)
        if comp is None:
            continue
        records.append({"symbol": row.symbol, "opened_at": row.opened_at, "pnl": float(row.realized_pnl), **comp})
    return pd.DataFrame(records)


def _add_all_up(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        df[f"d_{col}"] = df.groupby("symbol")[col].diff()
    df = df.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    df["all_up"] = (df["d_vol"] > 0) & (df["d_mom"] > 0) & (df["d_volat"] > 0) & (df["d_price"] > 0)
    return df


def _econ(sub: pd.DataFrame, days: float) -> dict:
    s = _stats(sub["pnl"].to_numpy() / 100)
    n = s.get("n", 0)
    per_month = n / days * 30 if days > 0 else 0
    usd_per_trade = 100.0 * s.get("ort_%", 0) / 100
    return {**s, "per_month": round(per_month), "usd_month": round(per_month * usd_per_trade)}


def _report(name: str, sub: pd.DataFrame, days: float) -> None:
    e = _econ(sub, days)
    print(
        f"  {name:30} n={e.get('n',0):>5}  WR%={e.get('wr',0):>6}  "
        f"PF={e.get('pf',0):>6}  işlem/ay={e.get('per_month',0):>6}  $/ay={e.get('usd_month',0):>8}"
    )


def _run_one(conn, indicator: str, direction: str, bad_symbols: set[str]) -> None:
    raw = _collect(conn, indicator, direction, bad_symbols)
    if raw.empty:
        print(f"\n{indicator} — {direction}: veri yok.")
        return
    df = _add_all_up(raw)
    days = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"\n{'='*78}\n{indicator} — {direction} (CANLI FORMÜL)  (n={len(df):,}, {days:.1f} gün)\n{'='*78}")

    _report("baseline (tümü)", df, days)
    _report("all_up=True", df[df["all_up"]], days)
    _report("all_up=False", df[~df["all_up"]], days)

    print("\n  -- mum-şekli (tüm veri) --")
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        _report(f"kategori={kat}", df[df["kategori"] == kat], days)

    all_up_df = df[df["all_up"]]
    print("\n  -- kombinasyon: all_up=True İÇİNDE mum-şekli --")
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        sub = all_up_df[all_up_df["kategori"] == kat]
        _report(f"all_up=T + {kat}", sub, days)

    if len(all_up_df) < 60:
        return
    cat_pf = {}
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        sub = all_up_df[all_up_df["kategori"] == kat]
        if len(sub) >= 20:
            cat_pf[kat] = _stats(sub["pnl"].to_numpy() / 100).get("pf", 0) or 0
    if not cat_pf:
        return
    best_kat = max(cat_pf, key=cat_pf.get)
    best = all_up_df[all_up_df["kategori"] == best_kat]
    rest = df[~((df["all_up"]) & (df["kategori"] == best_kat))]
    print(f"\n  -- en iyi kombinasyon: all_up=T + {best_kat} vs geri kalan --")
    _report(f"all_up=T + {best_kat}", best, days)
    _report("geri kalanı (hepsi)", rest, days)

    pf_best = _stats(best["pnl"].to_numpy() / 100).get("pf", 0) or 0
    pf_rest = _stats(rest["pnl"].to_numpy() / 100).get("pf", 0) or 0
    real_gap = pf_best - pf_rest

    rng = np.random.default_rng(42)
    n_true = len(best)
    pnl = df["pnl"].to_numpy() / 100
    count_ge = 0
    for _ in range(200):
        perm = rng.permutation(len(df))
        fake = np.zeros(len(df), dtype=bool)
        fake[perm[:n_true]] = True
        pf_t = _stats(pnl[fake]).get("pf", 0) or 0
        pf_f = _stats(pnl[~fake]).get("pf", 0) or 0
        if abs(pf_t - pf_f) >= abs(real_gap):
            count_ge += 1
    print(f"    placebo: gerçek PF farkı ({real_gap:+.3f}) rastgelede %{count_ge/200*100:.1f} sıklıkta çıktı")

    if len(best) >= 40:
        t_min, t_max = best["opened_at"].min(), best["opened_at"].max()
        mid = t_min + (t_max - t_min) / 2
        first = best[best["opened_at"] < mid]
        second = best[best["opened_at"] >= mid]
        print(
            f"    split-period: ilk yarı PF={_stats(first['pnl'].to_numpy()/100).get('pf',0):>6} "
            f"(n={len(first)})  ikinci yarı PF={_stats(second['pnl'].to_numpy()/100).get('pf',0):>6} "
            f"(n={len(second)})"
        )


def main() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")
    for direction in DIRECTIONS:
        _run_one(conn, "RSI_Cross(9,24)", direction, bad)
    conn.close()


if __name__ == "__main__":
    main()
