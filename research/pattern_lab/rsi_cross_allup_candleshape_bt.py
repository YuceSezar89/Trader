"""
RSI_Cross için iki BAĞIMSIZ ve GEÇERLİ (look-ahead'siz) filtrenin birleşimi:

- all_up: Hacim+Momentum+Volatilite+Fiyat'ın hepsi bir önceki aynı sembol+yön
  RSI_Cross sinyaline göre arttı (consecutive_vpmv_increase_bt.py'de bugün
  doğrulandı — ESKİ "VPMV sıçraması" filtresinden FARKLI yöntem, o filtre
  look-ahead içerdiği için geçersiz ilan edilmişti — jump=post[+1]-pre[-1],
  sinyal SONRASI barı kullanıyordu. Bu script SADECE sinyal ANINA kadarki
  geçmiş veriyi kullanıyor).
- mum-şekli: sinyal mumunun gövde/üst-fitil/alt-fitil hangisi baskınsa
  (rsi_cross_candle_shape_bt.py'de doğrulandı, look-ahead YOK — ama TEK
  BAŞINA ekonomik olarak zayıf, ~$126/ay).

NOT: st_confirmed (Supertrend onayı) BİLİNÇLİ OLARAK bu teste DAHİL EDİLMEDİ —
v2-25'te (project_pattern_lab) bulundu ki DB'deki st_confirmed HAM/filtresiz
Supertrend yönüne bakıyor, canlı sistemin GERÇEKTEN kullandığı SignalFilter'dan
geçmiş yön değil; gerçek filtreli haliyle test edilince edge'i neredeyse
tamamen kayboldu (PF 1.58→1.34, 0.74→1.30 gibi). O yüzden güvenilmez.

Kullanım: python -m research.pattern_lab.rsi_cross_allup_candleshape_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_atr, calculate_rsi, truncate_after_gap
from research.pattern_lab.vol_exhaustion_bt import _stats
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)

DIRECTIONS = ["Long", "Short"]
_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_BARS_NEEDED = 220
_GAP_HOURS_THRESHOLD = 200
_POSITION_USD = 100.0


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


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS open_time, open, high, low, close, volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
            "ORDER BY timestamp DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket AS open_time, open, high, low, close, volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
        )
    rows = cur.fetchall()
    if not rows or len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)
    df = truncate_after_gap(df)
    if len(df) < 60:
        return None
    return df


def _classify_candle(last: pd.Series) -> str:
    """rsi_cross_candle_shape_bt.py::_classify ile birebir aynı mantık."""
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


def _vpmv_components(df: pd.DataFrame, direction: str) -> dict | None:
    try:
        last = df.iloc[-1]
        rsi_series = calculate_rsi(df, period=14)
        rsi_centered = rsi_series - 50
        atr_series = calculate_atr(df, period=14)
        price_pct = df["close"].pct_change().fillna(0.0) * 100.0

        vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])
        mom_score = float(normalize_momentum_0_100(rsi_centered).iloc[-1])
        volat_score = float(normalize_volatility_0_100(atr_series).iloc[-1])
        price_score = float(normalize_price_0_100(price_pct).iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    vals = (vol_score, mom_score, volat_score, price_score)
    if any(pd.isna(v) for v in vals):
        return None
    kategori = _classify_candle(last)
    if direction == "Short":
        mom_score = 100.0 - mom_score
        price_score = 100.0 - price_score
    return {
        "vol": vol_score,
        "mom": mom_score,
        "volat": volat_score,
        "price": price_score,
        "kategori": kategori,
    }


def _collect(indicator: str, direction: str, exclude_symbols: set[str]) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, indicator, direction, exclude_symbols)
    if sig_df.empty:
        conn.close()
        return sig_df

    records = []
    for row in sig_df.itertuples():
        df = _fetch_bars(cur, row.symbol, row.interval, row.opened_at)
        if df is None:
            continue
        comp = _vpmv_components(df, direction)
        if comp is None:
            continue
        records.append(
            {
                "symbol": row.symbol,
                "opened_at": row.opened_at,
                "pnl": float(row.realized_pnl),
                **comp,
            }
        )
    conn.close()
    return pd.DataFrame(records)


def _add_all_up(df: pd.DataFrame) -> pd.DataFrame:
    # Sembol bazinda, zaman sirali, bir onceki AYNI SEMBOL sinyaline gore delta
    df = df.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        df[f"d_{col}"] = df.groupby("symbol")[col].diff()
    df = df.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    df["all_up"] = (
        (df["d_vol"] > 0) & (df["d_mom"] > 0) & (df["d_volat"] > 0) & (df["d_price"] > 0)
    )
    return df


def _econ(sub: pd.DataFrame, days: float) -> dict:
    s = _stats(sub["pnl"].to_numpy() / 100)
    n = s.get("n", 0)
    per_month = n / days * 30 if days > 0 else 0
    usd_per_trade = _POSITION_USD * s.get("ort_%", 0) / 100
    return {**s, "per_month": round(per_month), "usd_month": round(per_month * usd_per_trade)}


def _report(name: str, sub: pd.DataFrame, days: float) -> None:
    e = _econ(sub, days)
    print(
        f"  {name:30} n={e.get('n',0):>5}  WR%={e.get('wr',0):>6}  "
        f"PF={e.get('pf',0):>6}  işlem/ay={e.get('per_month',0):>6}  $/ay={e.get('usd_month',0):>8}"
    )


def _run_one(indicator: str, direction: str, bad_symbols: set[str]) -> None:
    raw = _collect(indicator, direction, bad_symbols)
    if raw.empty:
        print(f"\n{indicator} — {direction}: veri yok.")
        return
    df = _add_all_up(raw)

    days = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"\n{'='*78}\n{indicator} — {direction}  (n={len(df):,}, {days:.1f} gün)\n{'='*78}")

    _report("baseline (tümü)", df, days)
    _report("all_up=True", df[df["all_up"]], days)
    _report("all_up=False", df[~df["all_up"]], days)

    print("\n  -- mum-şekli (tüm veri) --")
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        _report(f"kategori={kat}", df[df["kategori"] == kat], days)

    print("\n  -- kombinasyon: all_up=True İÇİNDE mum-şekli --")
    all_up_df = df[df["all_up"]]
    for kat in ("govde", "ust_fitil", "alt_fitil"):
        sub = all_up_df[all_up_df["kategori"] == kat]
        _report(f"all_up=T + {kat}", sub, days)

    # En iyi tekil kategoriyi bul, split-period + placebo uygula
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


def run(indicators: list[str] | None = None) -> None:
    indicators = indicators or ["RSI_Cross(9,24)"]
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    bad_symbols = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad_symbols)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor\n")

    for indicator in indicators:
        for direction in DIRECTIONS:
            _run_one(indicator, direction, bad_symbols)


if __name__ == "__main__":
    import sys

    run([sys.argv[1]] if len(sys.argv) > 1 else None)
