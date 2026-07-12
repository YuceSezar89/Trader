"""
rsi_cross_combined_mtf_bt.py'nin devamı — Long'un ilk OOS yarısında GERÇEKTEN
bozuk çıkmasının nedeni bulundu (piyasa geneli %-6 ort/%-10 medyan getiriyle
çökmüştü, 189 sembolün %81.5'i düşüyordu) — MTF onayı (sembolün KENDİ üst-TF
trendi) bunu düzeltemedi çünkü sorun sembol-özgü değil, PİYASA GENELİ.

Bu script projenin ZATEN VAR OLAN ama şu an sadece kayıt için kullanılan BTC
rejim formülünü (`signals/signal_processor.py:582-594` — btc_z = (BTC_close -
BTC_EMA200)/BTC_rolling_std200, bullish>0.5/bearish<-0.5/neutral arası) AYNEN
kullanarak, sinyal ANINDA BTC "bearish" ise Long'u (ve simetrik olarak
"bullish" ise Short'u) elemenin ilk-yarı çöküşünü düzeltip düzeltmediğini
test eder. Aynı VPMV+SL disiplini (v2-18) korunuyor.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import DAYS, HORIZON_BARS, MIN_BARS  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_sl_sweep_bt import _rsi_cross_events, _apply_signal_filter  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import _fetch_with_volume, _report  # pylint: disable=wrong-import-position

BTC_EMA_SPAN = 200
BTC_STD_WINDOW = 200
BULLISH_Z = 0.5
BEARISH_Z = -0.5


def _fetch_btc_regime() -> dict:
    """signals/signal_processor.py'deki AYNI formül — BTCUSDT'nin 15m
    serisinde. Döner: {timestamp: btc_trend ('bullish'|'bearish'|'neutral')}."""
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT bucket AS ts, close FROM cagg_15m
        WHERE symbol = 'BTCUSDT' AND bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    closes = df["close"].astype(float)
    ema = closes.ewm(span=BTC_EMA_SPAN, adjust=False).mean()
    std = closes.rolling(BTC_STD_WINDOW).std()
    z = (closes - ema) / (std + 1e-12)
    trend = np.where(z > BULLISH_Z, "bullish", np.where(z < BEARISH_Z, "bearish", "neutral"))
    return dict(zip(df["ts"], trend))


def _regime_ok(direction: str, btc_trend: str) -> bool:
    """Long: BTC 'bearish' değilse (bullish/neutral kabul); Short: BTC
    'bullish' değilse (bearish/neutral kabul) — rejime KARŞI bahis elenir."""
    if btc_trend is None:
        return False
    if direction == "Long":
        return btc_trend != "bearish"
    return btc_trend != "bullish"


def run():
    btc_regime = _fetch_btc_regime()
    print(f"BTC rejim serisi: {len(btc_regime):,} bar\n")

    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    clean_events = []  # (direction, h, l, c, atr_val, i, entry_price, ts, jump, btc_trend)

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        fast = calculate_rsi(g, period=Config.RSI_FAST_WINDOW).to_numpy()
        slow = calculate_rsi(g, period=Config.RSI_SLOW_WINDOW).to_numpy()
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        raw_events = _rsi_cross_events(fast, slow)
        filtered_events = _apply_signal_filter(raw_events, h, l)
        if not filtered_events:
            continue

        series_long = compute_series(g, "Long")
        series_short = compute_series(g, "Short")

        for i, direction in filtered_events:
            if i + HORIZON_BARS >= len(c) or i - 1 < 0 or i + 1 >= len(g) \
                    or not (np.isfinite(atr[i]) and atr[i] > 0):
                continue
            series = series_long if direction == "Long" else series_short
            pre_v, post_v = series.iloc[i - 1], series.iloc[i + 1]
            if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                continue

            btc_trend = btc_regime.get(ts.iloc[i])
            clean_events.append((
                direction, h, l, c, atr[i], i, c[i], ts.iloc[i], post_v - pre_v, btc_trend,
            ))

    t_min = min(e[7] for e in clean_events)
    t_max = max(e[7] for e in clean_events)
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max}")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    is_events = [e for e in clean_events if e[7] < mid]
    oos_events = [e for e in clean_events if e[7] >= mid]
    jump_threshold = float(np.percentile([e[8] for e in is_events], 66.7))
    print(f"in-sample olay: {len(is_events)} | SABİT VPMV eşiği: {jump_threshold:.2f}\n")

    vpmv_only = [e for e in oos_events if e[8] >= jump_threshold]
    vpmv_regime = [e for e in vpmv_only if _regime_ok(e[0], e[9])]

    def _by_dir(events: list, direction: str) -> list:
        return [e for e in events if e[0] == direction]

    oos_mid = mid + (t_max - mid) / 2
    half_days = oos_days / 2

    for dir_label, direction in (("LONG+SHORT (birlikte)", None), ("SADECE LONG", "Long"), ("SADECE SHORT", "Short")):
        vo = vpmv_only if direction is None else _by_dir(vpmv_only, direction)
        vr = vpmv_regime if direction is None else _by_dir(vpmv_regime, direction)

        print(f"\n\n╔══════════ {dir_label} ══════════╗")
        print("\n════ TÜM OOS ════")
        _report("VPMV sıçraması (rejimsiz, v2-18)", [e[:7] for e in vo], oos_days)
        _report("VPMV + BTC rejim filtresi", [e[:7] for e in vr], oos_days)

        print(f"\n════ OOS — İLK YARI (biten: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] < oos_mid], half_days)
        _report("VPMV + BTC rejim filtresi", [e[:7] for e in vr if e[7] < oos_mid], half_days)

        print(f"\n════ OOS — İKİNCİ YARI (başlayan: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] >= oos_mid], half_days)
        _report("VPMV + BTC rejim filtresi", [e[:7] for e in vr if e[7] >= oos_mid], half_days)


if __name__ == "__main__":
    run()
