"""
do_kirilimi ham setup frekans taraması.

check() sadece pencerenin SON barını değerlendirir; bu script aynı 6-kapı +
ADX + ST state machine'ini geniş bir sembol+zaman penceresinde TÜM barlar
için replay eder ve her "entry" adayını (BTC filtresi öncesi/sonrası) sayar.
Amaç: canlıda 3 Tem'den beri hiç pozisyon açılmamasının nedeni — mantık hiç
ateşlemiyor mu, yoksa ateşliyor da BTC rejim/ayrışma filtresinde mi eleniyor?

Kullanım: python -m research.pattern_lab.scan_do_kirilimi_frequency [gun] [sembol_limit]
"""
import sys
import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from signals.do_kirilimi import (
    _crossover,
    _daily_open,
    _htf_forming_bull,
    ADX_LEN,
    ADX_MIN,
    DO_HOUR,
    HTF_MINUTES,
    MARKOV_ADD,
    MARKOV_KEEP,
    MARU_BODY,
    MARU_WICK,
    MIN_ONAY,
    MIN_BARS,
    SESSION_HOURS,
    ST_LEN,
    ST_MULT,
)

import pandas_ta_classic as pta  # pylint: disable=wrong-import-position


def _db_conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _scan_symbol(symbol: str, df: pd.DataFrame, btc_day_ret_by_ts: dict) -> list[dict]:
    """df: bucket(=lokal zaman)+OHLCV. Pencerenin TÜM barları için entry adaylarını döner."""
    d = df.reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        d[col] = d[col].astype(float)
    ts = pd.Series(pd.to_datetime(d["bucket"]))

    o = d["open"].to_numpy(); h = d["high"].to_numpy()
    l = d["low"].to_numpy();  c = d["close"].to_numpy()
    n = len(d)
    if n < MIN_BARS:
        return []

    st = pta.supertrend(d["high"], d["low"], d["close"], length=ST_LEN, multiplier=ST_MULT)
    if st is None:
        return []
    st_dir = st[f"SUPERTd_{ST_LEN}_{ST_MULT}"].to_numpy()
    adx = pta.adx(d["high"], d["low"], d["close"], length=ADX_LEN)[f"ADX_{ADX_LEN}"].to_numpy()
    atr = pta.atr(d["high"], d["low"], d["close"], length=14).to_numpy()

    rsi = pta.rsi(np.log(d["close"]), length=14)
    d_rsi = rsi.diff()
    c10 = _crossover(d_rsi, 10.0)
    c20 = _crossover(d_rsi, 20.0)

    dailyOpen, new_day = _daily_open(ts, o)
    mins_local = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    mins_from_do = (mins_local - DO_HOUR * 60) % 1440
    session_ok = mins_from_do < SESSION_HOURS * 60

    do_touch = (l <= dailyOpen) & (h >= dailyOpen)
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    do_break = (c > dailyOpen) & (prev_c <= dailyOpen)
    do_lift = (l <= dailyOpen) & (c > dailyOpen)
    c10c20_do = (c10 | c20) & (do_touch | do_break | do_lift) & np.isfinite(dailyOpen)

    rng = h - l
    body = np.abs(c - o)
    with np.errstate(divide="ignore", invalid="ignore"):
        marubozu = (c > o) & (rng > 0) & (body / rng >= MARU_BODY) & \
                   ((h - c) / rng <= MARU_WICK) & ((o - l) / rng <= MARU_WICK)
    prev_o = np.roll(o, 1); prev_o[0] = np.nan
    prev_h = np.roll(h, 1); prev_l = np.roll(l, 1)
    engulf = (c > o) & (prev_c < prev_o) & (l < prev_l) & (h > prev_h)

    bull = c > o
    bull1 = np.roll(bull, 2); bull2 = np.roll(bull, 1)
    bull1[:2] = False; bull2[:1] = False
    h2 = np.roll(h, 2); h2[:2] = np.nan
    fvg3 = bull1 & bull2 & bull & (l > h2)
    fvg_above_do = (h2 > dailyOpen) & (l > dailyOpen)
    setup_above_do = (o > dailyOpen) & (c > dailyOpen)

    bull_count = np.zeros(n, dtype=np.int8)
    for tfm in HTF_MINUTES:
        bull_count += _htf_forming_bull(ts, d, tfm).astype(np.int8)
    add_flags = bull_count >= MIN_ONAY
    s = 0.5
    t_ok = np.empty(n, dtype=bool)
    for i in range(n):
        s = s * MARKOV_KEEP + (MARKOV_ADD if add_flags[i] else 0.0)
        t_ok[i] = s > 0.5

    sdf_last_short_high = np.nan
    sdf_last_long_low = np.nan
    sdf_dir = 0
    do_setup = False
    fvg_mem = False
    fvg_lower = np.nan
    fvg_bar = -1

    events = []
    for i in range(1, n):
        long_flip = st_dir[i] == 1 and st_dir[i - 1] == -1
        short_flip = st_dir[i] == -1 and st_dir[i - 1] == 1
        if long_flip and not np.isnan(sdf_last_short_high) and l[i] > sdf_last_short_high:
            sdf_dir = 1
        if short_flip and not np.isnan(sdf_last_long_low) and h[i] < sdf_last_long_low:
            sdf_dir = -1
        if short_flip:
            sdf_last_short_high = h[i]
        if long_flip:
            sdf_last_long_low = l[i]
        sdf_long = sdf_dir == 1

        if new_day[i]:
            do_setup = False
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        sess = session_ok[i]
        if sess and sdf_long and t_ok[i] and c10c20_do[i]:
            do_setup = True
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        if do_setup and sess and sdf_long and t_ok[i] and fvg3[i] and fvg_above_do[i]:
            fvg_mem = True
            fvg_lower = h2[i]
            fvg_bar = i

        if fvg_mem and not np.isnan(fvg_lower) and i > fvg_bar and l[i] < fvg_lower:
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        setup_candle = (fvg_mem and sess and sdf_long and t_ok[i]
                         and setup_above_do[i] and (marubozu[i] or engulf[i]))
        if setup_candle:
            do_setup = False
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1
            if st_dir[i] == 1 and adx[i] >= ADX_MIN and np.isfinite(atr[i]) and atr[i] > 0:
                bar_ts = ts.iloc[i]
                do_i = dailyOpen[i]
                coin_day_ret = (c[i] / do_i - 1) * 100 if np.isfinite(do_i) and do_i > 0 else np.nan
                btc_ret = btc_day_ret_by_ts.get(bar_ts)
                btc_up = (btc_ret is not None) and (btc_ret > 0)
                ayrisma = (coin_day_ret - btc_ret) if (btc_ret is not None and np.isfinite(coin_day_ret)) else np.nan
                passes = bool(btc_up and ayrisma is not None and ayrisma > 0)
                events.append({
                    "symbol": symbol, "ts": bar_ts,
                    "coin_day_ret": round(coin_day_ret, 2) if np.isfinite(coin_day_ret) else None,
                    "btc_day_ret": round(btc_ret, 2) if btc_ret is not None else None,
                    "ayrisma": round(ayrisma, 2) if ayrisma is not None and np.isfinite(ayrisma) else None,
                    "btc_up": btc_up, "passes_filter": passes,
                })
    return events


def main(days: int = 14, symbol_limit: int = 100):
    conn = _db_conn()
    start = pd.Timestamp.now() - pd.Timedelta(days=days + 2)  # MIN_BARS buffer için +2 gün

    print(f"[scan] son {days} gün, en likit {symbol_limit} sembol taranıyor...")

    top_q = """
        SELECT symbol, sum(volume) v FROM cagg_5m
        WHERE bucket >= %s GROUP BY symbol ORDER BY v DESC LIMIT %s
    """
    top_symbols = pd.read_sql(top_q, conn, params=(start, symbol_limit))["symbol"].tolist()
    if "BTCUSDT" not in top_symbols:
        top_symbols.append("BTCUSDT")

    btc_df = pd.read_sql(
        "SELECT bucket, open, high, low, close, volume FROM cagg_5m WHERE symbol=%s AND bucket>=%s ORDER BY bucket",
        conn, params=("BTCUSDT", start),
    )
    btc_ts = pd.Series(pd.to_datetime(btc_df["bucket"]))
    btc_o = btc_df["open"].astype(float).to_numpy()
    btc_c = btc_df["close"].astype(float).to_numpy()
    btc_do, _ = _daily_open(btc_ts, btc_o)
    btc_day_ret = np.where(np.isfinite(btc_do) & (btc_do > 0), (btc_c / btc_do - 1) * 100, np.nan)
    btc_day_ret_by_ts = dict(zip(btc_ts, btc_day_ret))

    all_events = []
    raw_count = 0
    for idx, symbol in enumerate(top_symbols, 1):
        df = pd.read_sql(
            "SELECT bucket, open, high, low, close, volume FROM cagg_5m WHERE symbol=%s AND bucket>=%s ORDER BY bucket",
            conn, params=(symbol, start),
        )
        if df.empty:
            continue
        try:
            events = _scan_symbol(symbol, df, btc_day_ret_by_ts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  ! {symbol}: hata {exc}")
            continue
        raw_count += len(events)
        all_events.extend(events)
        if idx % 20 == 0:
            print(f"  ... {idx}/{len(top_symbols)} sembol tarandı, şu ana kadar {raw_count} ham entry")

    conn.close()

    passed = [e for e in all_events if e["passes_filter"]]
    print()
    print(f"=== SONUÇ ({days} gün, {len(top_symbols)} sembol) ===")
    print(f"Ham entry (6 kapı+ADX+ST, BTC filtresi öncesi): {len(all_events)}")
    print(f"BTC rejim+ayrışma filtresini geçen:              {len(passed)}")
    print()

    if all_events:
        df_ev = pd.DataFrame(all_events)
        print("--- Ham entry sembol dağılımı (ilk 15) ---")
        print(df_ev["symbol"].value_counts().head(15).to_string())
        print()
        print("--- Filtreyi geçenler ---")
        if passed:
            print(pd.DataFrame(passed).sort_values("ts").to_string(index=False))
        else:
            print("(yok)")
        print()
        print("--- BTC günü negatifken elenenler: %d ---" % sum(1 for e in all_events if not e["btc_up"]))
        print("--- Ayrışma <= 0 nedeniyle elenenler: %d ---" %
              sum(1 for e in all_events if e["btc_up"] and (e["ayrisma"] is None or e["ayrisma"] <= 0)))
    else:
        print("Hiç ham entry bulunamadı — sorun BTC filtresinde değil, 6 kapı+ADX+ST setup'ının kendisinde.")


if __name__ == "__main__":
    _days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    _limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(_days, _limit)
