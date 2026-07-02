"""
DO Kırılımı dedektörü — "Güçlü DO Kırılımı + FVG + Marubozu" (Pine v6 portu).

6 kapı: T (HTF HA onayı, Markov) · M (SuperTrend SDF güçlü flip) · D (Daily Open
temas) · C (C10/C20 log-RSI delta cross) · F (3 yeşil mum FVG, DO üstü) ·
B (Marubozu/Engulfing) + ADX(14) >= 25. Entry: setup ile aynı bar.

Paper-trade filtreleri (backtest 2 Tem 2026, PF 1.75 hücresi):
  BTC günü pozitif (BTC close > BTC DO) ve ayrışma > 0
  (coin gün-içi getirisi − BTC gün-içi getirisi, DO'ya göre).

Her çağrıda pencerenin tamamı replay edilir — kalıcı state yok
(restart/gap-proof). 300 barlık 5m penceresinde HTF 540/720 HA recursion'ı
sığ kalır; sapma geometrik küçülür (backtest tam geçmişle koşuldu, kabul
edilen yaklaşıklık).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

DO_HOUR       = 3
SESSION_HOURS = 12
HTF_MINUTES   = [90, 180, 360, 540, 720]
MIN_ONAY      = 3
MARKOV_KEEP, MARKOV_ADD = 0.7, 0.3
ST_LEN, ST_MULT   = 10, 3.0
ADX_LEN, ADX_MIN  = 14, 25.0
MARU_BODY, MARU_WICK = 0.70, 0.20
MIN_BARS = 220


def _heikin_ashi(o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, ha_c


def _htf_forming_bull(ts: pd.Series, df: pd.DataFrame, tf_min: int) -> np.ndarray:
    bucket = (ts - pd.Timedelta(hours=3)).dt.floor(f"{tf_min}min") + pd.Timedelta(hours=3)
    g = df.groupby(bucket.values)
    o_w = g["open"].transform("first").to_numpy()
    h_w = g["high"].cummax().to_numpy()
    l_w = g["low"].cummin().to_numpy()
    c_w = df["close"].to_numpy()
    ha_close_f = (o_w + h_w + l_w + c_w) / 4.0

    agg = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    ha_o_full, ha_c_full = _heikin_ashi(
        agg["o"].to_numpy(), agg["h"].to_numpy(), agg["l"].to_numpy(), agg["c"].to_numpy()
    )
    prev_ha_o = pd.Series(np.roll(ha_o_full, 1), index=agg.index)
    prev_ha_c = pd.Series(np.roll(ha_c_full, 1), index=agg.index)
    prev_ha_o.iloc[0] = np.nan
    prev_ha_c.iloc[0] = np.nan
    ha_open_f = ((prev_ha_o + prev_ha_c) / 2.0).reindex(bucket.values).to_numpy()
    return (ha_close_f > ha_open_f) & ~np.isnan(ha_open_f)


def _crossover(s: pd.Series, level: float) -> np.ndarray:
    return ((s > level) & (s.shift(1) <= level)).fillna(False).to_numpy()


def _daily_open(ts: pd.Series, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day_key = (ts - pd.Timedelta(hours=DO_HOUR)).dt.date
    new_day = (day_key != day_key.shift(1)).to_numpy()
    new_day[0] = True
    do = pd.Series(np.where(new_day, o, np.nan)).ffill().to_numpy()
    return do, new_day


def btc_day_context(btc_df: pd.DataFrame) -> Optional[dict]:
    """BTC'nin gün-içi durumu: DO'ya göre getiri ve pozitiflik."""
    if btc_df is None or len(btc_df) < 10 or "open_time" not in btc_df.columns:
        return None
    # open_time epoch ms (UTC) → lokal naive (İstanbul)
    ts = pd.to_datetime(btc_df["open_time"], unit="ms") + pd.Timedelta(hours=3)
    o = btc_df["open"].astype(float).to_numpy()
    c = float(btc_df["close"].astype(float).iloc[-1])
    do, _ = _daily_open(pd.Series(ts), o)
    b_do = do[-1]
    if not np.isfinite(b_do) or b_do <= 0:
        return None
    day_ret = (c / b_do - 1) * 100
    return {"day_ret": day_ret, "day_up": day_ret > 0}


class DOKirilimiDetector:
    """Sembolün 5m penceresinde son kapanan barda entry olup olmadığını söyler."""

    def check(self, symbol: str, df: pd.DataFrame, btc_ctx: Optional[dict]) -> Optional[dict]:
        try:
            import pandas_ta_classic as pta  # pylint: disable=import-outside-toplevel

            if df is None or len(df) < MIN_BARS:
                return None
            d = df.tail(320).reset_index(drop=True).copy()
            for col in ("open", "high", "low", "close"):
                d[col] = d[col].astype(float)

            if "open_time" in d.columns:
                ts = pd.to_datetime(d["open_time"], unit="ms") + pd.Timedelta(hours=3)
            else:
                ts = pd.to_datetime(d.index)
            ts = pd.Series(ts)

            o = d["open"].to_numpy(); h = d["high"].to_numpy()
            l = d["low"].to_numpy();  c = d["close"].to_numpy()
            n = len(d)

            st = pta.supertrend(d["high"], d["low"], d["close"], length=ST_LEN, multiplier=ST_MULT)
            if st is None:
                return None
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

            # ── State machine replay (Pine §16-21 sırası) ──
            sdf_last_short_high = np.nan
            sdf_last_long_low = np.nan
            sdf_dir = 0
            do_setup = False
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1
            entry_at_last = False

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
                    if i == n - 1 and st_dir[i] == 1 and adx[i] >= ADX_MIN:
                        entry_at_last = True

            if not entry_at_last:
                return None

            # ── Paper filtreleri: BTC rejimi + ayrışma ──
            cfg = Config.PAPER["DO_KIRILIMI"]
            if btc_ctx is None:
                logger.debug("[DOKirilimi] %s: BTC bağlamı yok, atlandı", symbol)
                return None
            if cfg.get("REQUIRE_BTC_DAY_UP", True) and not btc_ctx["day_up"]:
                logger.info("[DOKirilimi] %s: setup var ama BTC günü negatif — atlandı", symbol)
                return None
            coin_day_ret = (c[-1] / dailyOpen[-1] - 1) * 100
            ayrisma = coin_day_ret - btc_ctx["day_ret"]
            if ayrisma <= cfg.get("AYRISMA_MIN", 0.0):
                logger.info("[DOKirilimi] %s: setup var ama ayrışma %.2f%% <= eşik — atlandı",
                            symbol, ayrisma)
                return None

            atr_val = float(atr[-1])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None
            price = float(c[-1])
            return {
                "price": price,
                "atr": atr_val,
                "sl_price": price - cfg["SL_ATR"] * atr_val,
                "tp_price": price + cfg["TP_ATR"] * atr_val,
                "ayrisma": round(ayrisma, 2),
                "pattern": "MARU" if marubozu[-1] else "ENGULF",
                "z_hint": round(coin_day_ret, 2),
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[DOKirilimi] %s dedektör hatası: %s", symbol, exc)
            return None


do_kirilimi_detector = DOKirilimiDetector()
