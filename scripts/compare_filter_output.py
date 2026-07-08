"""
Supertrend + SignalFilter karşılaştırma scripti.

PineScript çıktısıyla karşılaştırmak için:
  python scripts/compare_filter_output.py --symbol BTCUSDT --interval 1h --limit 500

Çıktı: CSV + konsol tablosu
  timestamp | signal | high | low | filter_ref | valid

PineScript parametreleri (değiştirilebilir):
  ATR Length : 10
  Factor     : 3.0
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from signals.signal_filter import SignalFilter


# ── Supertrend (PineScript ta.supertrend ile birebir) ────────────────────────

def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder Moving Average — PineScript ta.rma ile aynı."""
    alpha = 1.0 / length
    result = series.copy().astype(float)
    result.iloc[:length] = np.nan
    first_valid = series.iloc[:length].mean()
    result.iloc[length - 1] = first_valid
    for i in range(length, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result


def supertrend(df: pd.DataFrame, atr_length: int = 10, factor: float = 3.0) -> pd.DataFrame:
    """
    PineScript ta.supertrend() ile birebir hesaplama.
    Döndürür: df + ['st_line', 'st_dir', 'long_signal', 'short_signal']
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, atr_length)
    hl2 = (high + low) / 2.0

    upper_basic = hl2 + factor * atr
    lower_basic = hl2 - factor * atr

    upper = pd.Series(np.nan, index=df.index, dtype=float)
    lower = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(np.nan, index=df.index, dtype=float)

    start = atr_length - 1
    upper.iloc[start] = upper_basic.iloc[start]
    lower.iloc[start] = lower_basic.iloc[start]
    direction.iloc[start] = 1.0

    for i in range(start + 1, len(df)):
        ub = upper_basic.iloc[i]
        lb = lower_basic.iloc[i]
        prev_upper = upper.iloc[i - 1]
        prev_lower = lower.iloc[i - 1]
        prev_close = close.iloc[i - 1]

        upper.iloc[i] = ub if ub < prev_upper or prev_close > prev_upper else prev_upper
        lower.iloc[i] = lb if lb > prev_lower or prev_close < prev_lower else prev_lower

        prev_dir = direction.iloc[i - 1]
        if prev_dir == -1:
            direction.iloc[i] = 1.0 if close.iloc[i] < lower.iloc[i] else -1.0
        else:
            direction.iloc[i] = -1.0 if close.iloc[i] > upper.iloc[i] else 1.0

    st_line = pd.Series(np.where(direction == -1, lower, upper), index=df.index)

    df["st_line"] = st_line
    df["st_dir"] = direction
    df["is_long"] = direction < 0
    df["is_short"] = direction > 0
    df["long_signal"] = df["is_long"] & ~df["is_long"].shift(1).fillna(False)
    df["short_signal"] = df["is_short"] & ~df["is_short"].shift(1).fillna(False)

    return df


# ── Veri çekme ────────────────────────────────────────────────────────────────

_RESAMPLE_MAP = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


def fetch_ohlcv(
    symbol: str, interval: str, limit: int,
    from_date: str | None = None, to_date: str | None = None,
) -> pd.DataFrame:
    import psycopg2

    resample_rule = _RESAMPLE_MAP.get(interval)
    multipliers = {"5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "1D": 1440}

    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT,
        dbname=Config.DB_NAME, user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM price_data WHERE symbol=%s AND interval=%s LIMIT 1", (symbol, interval))
    native = cur.fetchone()

    db_interval = interval if native else "1m"
    do_resample = not native and resample_rule and interval != "1m"
    fetch_limit = limit * multipliers.get(resample_rule, 1) + 200 if do_resample else limit

    if do_resample:
        print(f"  {interval} DB'de yok, 1m'den resample edilecek.")

    if from_date and to_date:
        cur.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol=%s AND interval=%s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """,
            (symbol, db_interval, from_date, to_date),
        )
    else:
        cur.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol=%s AND interval=%s
            ORDER BY timestamp DESC LIMIT %s
            """,
            (symbol, db_interval, fetch_limit),
        )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise SystemExit(f"Veri bulunamadı: {symbol} {db_interval} {from_date or ''}–{to_date or ''}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if do_resample:
        df = df.set_index("timestamp")
        df = df.resample(resample_rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if not from_date:
            df = df.tail(limit).reset_index(drop=True)
        print(f"  Resample sonrası: {len(df)} mum ({interval})")

    return df



# ── Ana akış ─────────────────────────────────────────────────────────────────

async def run(
    symbol: str, interval: str, limit: int, atr_length: int, factor: float,
    out_csv: str | None, from_date: str | None, to_date: str | None,
) -> None:
    date_info = f"{from_date}–{to_date}" if from_date else f"limit={limit}"
    print(f"Veri çekiliyor: {symbol} {interval} {date_info}")
    df = fetch_ohlcv(symbol, interval, limit, from_date, to_date)
    print(f"  {len(df)} mum yüklendi.")

    df = supertrend(df, atr_length=atr_length, factor=factor)

    filt = SignalFilter()
    indicator_key = f"Supertrend({atr_length},{factor})"
    # Canlı sistem aynı (symbol, interval, indicator) key'ini gerçek sinyal
    # filtrelemesi için kullanıyor — check() artık her denemeyi doğrudan
    # signal_filter_events'e yazdığı için, gerçek symbol adıyla çalıştırmak bu
    # analiz verisini canlı filtre referanslarına karıştırırdı. Binance
    # sembollerinde "_" hiç geçmediğinden bu suffix gerçek bir sembolle asla
    # çakışmaz.
    filter_symbol = f"{symbol}__ANALYSIS"

    rows = []
    for _, row in df.iterrows():
        if not row["long_signal"] and not row["short_signal"]:
            continue

        sig = "Long" if row["long_signal"] else "Short"
        opposite = "Short" if sig == "Long" else "Long"

        # Referansı check()'ten ÖNCE al — check() denemeyi kaydettikten sonra
        # bu satırın kendi high/low'u referansa karışmasın.
        ref_pair = await filt.last_reference(filter_symbol, interval, indicator_key, opposite)
        ref_label = "prevShortHigh" if sig == "Long" else "prevLongLow"
        ref_val = (ref_pair[0] if sig == "Long" else ref_pair[1]) if ref_pair else None

        bar_time = row["timestamp"].tz_localize(None) if row["timestamp"].tzinfo else row["timestamp"]
        valid = await filt.check(
            sig,
            high=float(row["high"]),
            low=float(row["low"]),
            symbol=filter_symbol,
            interval=interval,
            indicator=indicator_key,
            bar_time=bar_time,
        )

        ts_str = row["timestamp"].tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M")
        rows.append({
            "timestamp": ts_str,
            "signal": sig,
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "filter_ref_name": ref_label,
            "filter_ref_val": round(ref_val, 4) if ref_val is not None else "—",
            "valid": "✓" if valid else "✗",
        })

    await filt.cleanup(filter_symbol, interval, indicator_key)

    result = pd.DataFrame(rows)

    if result.empty:
        print("Sinyal üretilmedi.")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(f"\n=== {symbol} {interval} — Supertrend({atr_length},{factor}) ===")
    print(result.to_string(index=False))
    print(f"\nToplam: {len(result)} sinyal  |  "
          f"Geçerli: {(result['valid']=='✓').sum()}  |  "
          f"Geçersiz: {(result['valid']=='✗').sum()}")

    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"CSV kaydedildi: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supertrend + SignalFilter karşılaştırma")
    parser.add_argument("--symbol",     default="BTCUSDT")
    parser.add_argument("--interval",   default="1h")
    parser.add_argument("--limit",      type=int, default=500)
    parser.add_argument("--atr-length", type=int, default=10)
    parser.add_argument("--factor",     type=float, default=3.0)
    parser.add_argument("--csv",   default=None, help="CSV çıktı dosyası")
    parser.add_argument("--from",  default=None, dest="from_date", help="Başlangıç tarihi (2025-09-26)")
    parser.add_argument("--to",    default=None, dest="to_date",   help="Bitiş tarihi   (2025-10-02)")
    args = parser.parse_args()

    asyncio.run(run(args.symbol, args.interval, args.limit, args.atr_length, args.factor,
        args.csv, args.from_date, args.to_date))
