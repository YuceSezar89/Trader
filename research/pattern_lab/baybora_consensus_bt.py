"""
Baybora "Consensus Master v2.4" (13-state consensus, VPMV=%70 fiyat ağırlıklı)
sistemi — kendi 1h verimizle (cagg_1h) backtest.

Baybora'nın `compute_consensus_full` fonksiyonu (ve yardımcıları) DOKUNULMADAN,
doğrudan kendi dosyasından import edilir (saf numpy/numba, DB/Redis bağımlılığı
yok). Onların kendi varsayılan parametreleri kullanılır (config_vpmv.py +
consensus_calculator.py sınıf sabitleri): RSI(14), ATR(14)/lookback(100),
SuperTrend(10,3.0), ağırlık V=10/M=10/P=70/VLT=10, LVMACD(12,21,5,35),
XMACD(12,26,9), eşik 11/13, pencere (MAX_BARS) 500 bar.

Günlük/haftalık/aylık açılış: onların CANLI kodu bunu "şu anki tek skaler
değer" olarak tutuyor (backtest için look-ahead riski taşır) — burada
bar-bazlı, look-ahead'siz doğru seri hesaplanıyor (UTC takvim sınırları).

Walk-forward: her 1h barda son WINDOW=500 barlık pencereyle full-recalc
(onların kendi "bar_changed=True" full recalc deseniyle birebir), state
(ribbon/c12/c13/pozisyon/signal_bars) adım adım taşınıyor.

Giriş: pozisyon flip'i (eşik 11/13 geçildiğinde). Çıkış: kendi paper_trade.py
konvansiyonları — TP=%3, SL=%2, ters sinyalde kapat (hangisi önce gelirse).

Kullanım: python -m research.pattern_lab.baybora_consensus_bt [sembol_sayısı] [gün]
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config  # pylint: disable=wrong-import-position

BAYBORA_PATH = "/Users/yusuf/Downloads/baybora/Binance_customerv3/stratejiler/vpmv_cons"
sys.path.insert(0, BAYBORA_PATH)
from consensus_calculator import compute_consensus_full  # pylint: disable=import-error,wrong-import-position

# ── Baybora'nın kendi varsayılan parametreleri (DEĞİŞTİRİLMEDİ) ──────────────
RSI_LEN, VOL_LEN, ATR_LEN, ATR_LOOKBACK = 14, 14, 14, 100
ST_PERIOD, ST_MULT = 10, 3.0
VOL_W, MOM_W, PRC_W, VLT_W = 10.0, 10.0, 70.0, 10.0
LV_FAST, LV_SLOW, LV_SIG, LV_LR = 12, 21, 5, 35
XM_SHORT, XM_LONG, XM_TRIG = 12, 26, 9
AL_ESIGI, SAT_ESIGI = 11, 11
WINDOW = 500  # MAX_BARS

# ── Bizim paper_trade.py konvansiyonlarıyla eşleşen TP/SL ────────────────────
TP_PCT = 0.03
SL_PCT = 0.02

# ── Ekonomik etki konvansiyonu (do_break_gauss_economic_bt.py ile aynı) ─────
POSITION_USD = 100.0
FEE_RATE = 0.0005
ROUND_TRIP_FEE = POSITION_USD * FEE_RATE * 2

DEFAULT_NUM_SYMBOLS = 100
DEFAULT_DAYS = 120


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _get_symbols(limit: int) -> list[str]:
    conn = _conn()
    q = "SELECT symbol, COUNT(*) c FROM cagg_1h GROUP BY symbol ORDER BY c DESC LIMIT %s"
    df = pd.read_sql(q, conn, params=(limit,))
    conn.close()
    return df["symbol"].tolist()


def _fetch(symbol: str, days: int) -> pd.DataFrame:
    conn = _conn()
    q = """SELECT bucket, open, high, low, close, volume FROM cagg_1h
           WHERE symbol=%s AND bucket >= now() - interval '%s days'
           ORDER BY bucket"""
    df = pd.read_sql(q, conn, params=(symbol, days))
    conn.close()
    return df


def _period_opens(ts: pd.Series, o: np.ndarray):
    """Bar-bazlı, look-ahead'siz günlük/haftalık/aylık açılış serisi (UTC)."""
    day_key = ts.dt.date
    iso = ts.dt.isocalendar()
    week_key = iso.year.astype(str) + "-W" + iso.week.astype(str)
    month_key = ts.dt.year.astype(str) + "-" + ts.dt.month.astype(str)

    def _open_series(keys):
        new_period = (keys != keys.shift(1)).to_numpy()
        new_period[0] = True
        return pd.Series(np.where(new_period, o, np.nan)).ffill().to_numpy()

    return _open_series(day_key), _open_series(week_key), _open_series(month_key)


def replay_symbol(df: pd.DataFrame) -> list[dict]:
    """Walk-forward consensus replay — her pozisyon flip'inde bir entry event'i."""
    n = len(df)
    if n < WINDOW + 10:
        return []

    ts = pd.to_datetime(df["bucket"], utc=True)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    daily_o, weekly_o, monthly_o = _period_opens(ts, o)

    prev_ribbon, prev_c12, prev_c13, prev_poz, prev_sb = 0, 0, 0, 0, 0
    events: list[dict] = []

    for i in range(WINDOW, n):
        lo = i - WINDOW + 1
        sl = slice(lo, i + 1)

        result = compute_consensus_full(
            c[sl], h[sl], l[sl], v[sl],
            float(daily_o[i]), float(weekly_o[i]), float(monthly_o[i]),
            RSI_LEN, VOL_LEN, ATR_LEN, ATR_LOOKBACK,
            ST_PERIOD, ST_MULT,
            VOL_W, MOM_W, PRC_W, VLT_W,
            LV_FAST, LV_SLOW, LV_SIG, LV_LR,
            XM_SHORT, XM_LONG, XM_TRIG,
            prev_ribbon, prev_c12, prev_c13, prev_poz, prev_sb,
            AL_ESIGI, SAT_ESIGI,
        )
        _states, _ye, _kr, poz, sb, rib, c12s, c13s, _ema_vals = result

        if poz != prev_poz and poz != 0:
            own_ret_24h = float(c[i] / c[i - 24] - 1) if i >= 24 else np.nan
            events.append({
                "idx": i,
                "direction": "Long" if poz == 1 else "Short",
                "entry_price": float(c[i]),
                "entry_time": ts.iloc[i],
                "own_ret_24h": own_ret_24h,
            })

        prev_ribbon, prev_c12, prev_c13, prev_poz, prev_sb = rib, c12s, c13s, poz, sb

    return events


def simulate_exits(df: pd.DataFrame, events: list[dict]) -> list[dict]:
    """TP=%3 / SL=%2 / ters sinyal — hangisi önce gelirse (paper_trade.py ile aynı)."""
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(c)

    trades = []
    for k, ev in enumerate(events):
        idx = ev["idx"]
        direction = ev["direction"]
        entry = ev["entry_price"]
        reverse_idx = events[k + 1]["idx"] if k + 1 < len(events) else n - 1

        exit_price, exit_reason = None, None
        for j in range(idx + 1, reverse_idx + 1):
            if direction == "Long":
                tp_price, sl_price = entry * (1 + TP_PCT), entry * (1 - SL_PCT)
                hit_tp, hit_sl = h[j] >= tp_price, l[j] <= sl_price
            else:
                tp_price, sl_price = entry * (1 - TP_PCT), entry * (1 + SL_PCT)
                hit_tp, hit_sl = l[j] <= tp_price, h[j] >= sl_price

            if hit_sl:  # aynı barda ikisi de tetiklenirse kötümser varsayım: SL önce
                exit_price, exit_reason = sl_price, "SL"
                break
            if hit_tp:
                exit_price, exit_reason = tp_price, "TP"
                break

        if exit_price is None:
            exit_price, exit_reason = c[reverse_idx], "reverse"

        ret = (exit_price / entry - 1) if direction == "Long" else (entry / exit_price - 1)
        trades.append({
            "entry_time": ev["entry_time"], "direction": direction,
            "entry_price": entry, "exit_price": exit_price,
            "exit_reason": exit_reason, "ret": ret,
            "own_ret_24h": ev["own_ret_24h"],
        })
    return trades


def _fetch_btc_return_series(days: int) -> pd.DataFrame:
    """BTCUSDT'nin kendi 24h getirisi — 'ayrışma' (divergence) referansı."""
    df = _fetch("BTCUSDT", days)
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    df["btc_ret_24h"] = df["close"] / df["close"].shift(24) - 1
    return df[["bucket", "btc_ret_24h"]].dropna().sort_values("bucket")


def add_divergence(tdf: pd.DataFrame, days: int) -> pd.DataFrame:
    """Her işleme, o anki BTC 24h getirisine göre 'ayrışma' skoru ekler.
    Long: kendi getiri - BTC getirisi (pozitif = BTC'den daha güçlü yükseliyor).
    Short: BTC getirisi - kendi getiri (pozitif = BTC'den daha güçlü düşüyor).
    Kullanıcının tarif ettiği 'elle en çok ayrışanı seçme' davranışının vekili."""
    btc = _fetch_btc_return_series(days)
    tdf = tdf.sort_values("entry_time")
    merged = pd.merge_asof(tdf, btc, left_on="entry_time", right_on="bucket", direction="backward")
    sign = np.where(merged["direction"] == "Long", 1.0, -1.0)
    merged["ayrisma"] = sign * (merged["own_ret_24h"] - merged["btc_ret_24h"])
    return merged


def _dollar_stats(rets: np.ndarray, days_span: float) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    pnl = rets * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    per_month = total / days_span * 30 if days_span > 0 else 0.0
    return {
        "n": len(rets), "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3), "total_usd": round(total, 1),
        "usd_per_month": round(per_month, 1),
        "pf": round(float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())), 3) if (pnl < 0).any() else float("inf"),
    }


def run(num_symbols: int, days: int):
    symbols = _get_symbols(num_symbols)
    print(f"Test evreni: {len(symbols)} sembol, son {days} gün (1h)")
    print(f"Parametreler: pencere={WINDOW} bar, eşik={AL_ESIGI}/13, TP=%{TP_PCT*100:.0f} SL=%{SL_PCT*100:.0f}\n")

    all_trades = []
    for i, symbol in enumerate(symbols):
        df = _fetch(symbol, days)
        if len(df) < WINDOW + 10:
            continue
        events = replay_symbol(df)
        if not events:
            continue
        trades = simulate_exits(df, events)
        for t in trades:
            t["symbol"] = symbol
        all_trades.extend(trades)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(symbols)} sembol işlendi, şu ana kadar {len(all_trades)} işlem")

    if not all_trades:
        print("HİÇ İŞLEM YOK — eşik hiç geçilmedi ya da veri yetersiz.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf = add_divergence(tdf, days)
    t_min, t_max = tdf["entry_time"].min(), tdf["entry_time"].max()
    span_days = (t_max - t_min).total_seconds() / 86400
    mid = t_min + (t_max - t_min) / 2

    print(f"\nToplam {len(tdf)} işlem, {t_min.date()} - {t_max.date()} ({span_days:.0f} gün)")
    print(f"Yön dağılımı: Long={sum(tdf['direction']=='Long')} Short={sum(tdf['direction']=='Short')}")
    print(f"Çıkış nedeni: {tdf['exit_reason'].value_counts().to_dict()}\n")

    print("── TÜMÜ ──")
    s = _dollar_stats(tdf["ret"].to_numpy(), span_days)
    print(f"n={s['n']} WR%={s['wr']} PF={s['pf']} ort$/işlem={s['avg_usd']} $/ay={s['usd_per_month']}\n")

    print("── Yöne göre ──")
    for direction in ("Long", "Short"):
        sub = tdf[tdf["direction"] == direction]
        s = _dollar_stats(sub["ret"].to_numpy(), span_days)
        print(f"{direction:6} n={s.get('n',0)} WR%={s.get('wr','-')} PF={s.get('pf','-')} "
              f"ort$/işlem={s.get('avg_usd','-')} $/ay={s.get('usd_per_month','-')}")

    print("\n── Split-period sağlamlık (ilk yarı vs ikinci yarı) ──")
    first_half = tdf[tdf["entry_time"] < mid]
    second_half = tdf[tdf["entry_time"] >= mid]
    for name, sub in (("İlk yarı", first_half), ("İkinci yarı", second_half)):
        half_days = span_days / 2
        s = _dollar_stats(sub["ret"].to_numpy(), half_days)
        print(f"{name:12} n={s.get('n',0)} WR%={s.get('wr','-')} PF={s.get('pf','-')} "
              f"ort$/işlem={s.get('avg_usd','-')} $/ay={s.get('usd_per_month','-')}")

    print("\n── AYRIŞMA tercili (kullanıcının 'en çok hareket edeni elle seç' davranışının vekili) ──")
    print("(ayrışma = sinyal yönünde, BTC'ye göre 24h göreli getiri fazlası)")
    valid = tdf.dropna(subset=["ayrisma"]).copy()
    valid["tercil"] = pd.qcut(valid["ayrisma"], 3, labels=["düşük", "orta", "yüksek"], duplicates="drop")
    for name in ["düşük", "orta", "yüksek"]:
        sub = valid[valid["tercil"] == name]
        s = _dollar_stats(sub["ret"].to_numpy(), span_days)
        print(f"{name:8} n={s.get('n',0)} WR%={s.get('wr','-')} PF={s.get('pf','-')} "
              f"ort$/işlem={s.get('avg_usd','-')} $/ay={s.get('usd_per_month','-')}")

    print("\n── AYRIŞMA tercili × split-period (sağlamlık) ──")
    for half_name, half_df in (("İlk yarı", first_half), ("İkinci yarı", second_half)):
        half_valid = half_df.dropna(subset=["ayrisma"]).copy() if "ayrisma" in half_df.columns else valid[valid["entry_time"].isin(half_df["entry_time"])]
        half_valid = valid[valid["entry_time"].isin(half_df["entry_time"])]
        if half_valid.empty:
            continue
        half_valid = half_valid.copy()
        try:
            half_valid["tercil2"] = pd.qcut(half_valid["ayrisma"], 3, labels=["düşük", "orta", "yüksek"], duplicates="drop")
        except ValueError:
            continue
        print(f"  {half_name}:")
        for name in ["düşük", "orta", "yüksek"]:
            sub = half_valid[half_valid["tercil2"] == name]
            half_days = span_days / 2
            s = _dollar_stats(sub["ret"].to_numpy(), half_days)
            print(f"    {name:8} n={s.get('n',0)} WR%={s.get('wr','-')} PF={s.get('pf','-')} "
                  f"ort$/işlem={s.get('avg_usd','-')} $/ay={s.get('usd_per_month','-')}")


if __name__ == "__main__":
    n_sym = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM_SYMBOLS
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DAYS
    run(n_sym, n_days)
