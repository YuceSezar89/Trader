"""
st_confirmed DÜZELTMESİ — kullanıcının uyarısı (11 Tem 2026): `st_confirmed`
alanı Supertrend'in HAM `st_direction`'ına (calculate_supertrend'in ATR
bant kesişimi) bakıyor, ama canlı sistemin GERÇEKTEN ürettiği Supertrend
sinyali `signal_filter.py::SignalFilter` şartından geçmiş olanı — yani
"bu long'un high'ı, EN SON short sinyalinin high'ından büyük mü" (fiyat
hareketindeki break-of-structure/önceki high-low kırılımı fikri, yatayda
Supertrend'in gereksiz flip atmasını önlemek için) + aynı yönde art arda
tekrar sinyal üretmeme (trend devam) kuralı.

Bu script, HAM Supertrend flip'lerini (`indicators/core.py::calculate_supertrend`)
`_apply_signal_filter` (signal_filter.py ile birebir, bugün RSI_Cross için
zaten doğrulanmış) + trend-devam bastırmasıyla FİLTRELENMİŞ ST yönüne
çevirip, RSI_Cross/HA_Cross/MA200_Cross sinyalleriyle bu FİLTRELİ yönü
karşılaştırır (ham st_direction yerine).

Look-ahead kontrolü: aynı sembol+interval'in KENDİ ham verisinden hesaplanan
filtreli yön, sadece sinyal barından ÖNCE (veya aynı anda) kapanmış barlara
bakılarak forward-fill edilip kullanılıyor — 11 Tem'deki iki hatadan
(bucket/opened_at karışıklığı — `_signal_bar_ts` ile düzeltildi; farklı
TF'ler arası look-ahead — burada TEK bir TF/sembol içinde kalındığı için
uygulanmıyor) etkilenmiyor.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_supertrend  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import _fetch_symbol_history, _signal_bar_ts  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_sl_sweep_bt import _apply_signal_filter  # pylint: disable=wrong-import-position

INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross"]
DIRECTIONS = ["Long", "Short"]
INTERVALS = ["5m", "15m"]


def _suppress_duplicate_direction(filtered_events: list) -> list:
    """signal_engine.py::supertrend_signal'daki _st_last_valid mantığı —
    art arda aynı yönde filtreli sinyal üretilmez (trend devam)."""
    result = []
    last_valid = None
    for i, direction in filtered_events:
        if last_valid == direction:
            continue
        last_valid = direction
        result.append((i, direction))
    return result


def _filtered_st_direction_series(symbol: str, interval: str) -> pd.DataFrame:
    """Sembolün KENDİ ham verisinden (aynı TF) FİLTRELİ Supertrend yönünü
    (bullish/bearish) her bar için forward-fill edilmiş olarak döner."""
    hist = _fetch_symbol_history(symbol, interval)
    if len(hist) < 20:
        return pd.DataFrame(columns=["ts", "filtered_bullish"])
    hist = hist.sort_values("ts").reset_index(drop=True)
    high = hist["high"].to_numpy(float)
    low = hist["low"].to_numpy(float)

    _st_line, _direction, long_signal, short_signal = calculate_supertrend(hist)
    raw_events = []
    for i in range(len(hist)):
        if bool(long_signal.iloc[i]):
            raw_events.append((i, "Long"))
        elif bool(short_signal.iloc[i]):
            raw_events.append((i, "Short"))

    filtered = _apply_signal_filter(raw_events, high, low)
    filtered = _suppress_duplicate_direction(filtered)

    filtered_dir = pd.Series(np.nan, index=hist.index)
    for i, direction_label in filtered:
        filtered_dir.iloc[i] = 1.0 if direction_label == "Long" else 0.0
    filtered_dir = filtered_dir.ffill()

    out = hist[["ts"]].copy()
    out["filtered_bullish"] = filtered_dir
    return out


def _fetch_signals(indicator: str, direction: str, interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, realized_pnl, opened_at
        FROM signals
        WHERE indicators = %s AND signal_type = %s AND interval = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction, interval))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


def run():
    for indicator in INDICATORS:
        for direction in DIRECTIONS:
            rows = []
            for interval in INTERVALS:
                sigs = _fetch_signals(indicator, direction, interval)
                for symbol, sub in sigs.groupby("symbol"):
                    st_df = _filtered_st_direction_series(symbol, interval)
                    if st_df.empty:
                        continue
                    ts_to_idx = {t: i for i, t in enumerate(st_df["ts"])}
                    bullish_arr = st_df["filtered_bullish"].to_numpy()

                    for _, row in sub.iterrows():
                        i = ts_to_idx.get(_signal_bar_ts(row["opened_at"], interval))
                        if i is None or not np.isfinite(bullish_arr[i]):
                            continue
                        st_bullish = bullish_arr[i] == 1.0
                        st_confirmed_new = (direction == "Long" and st_bullish) or \
                                            (direction == "Short" and not st_bullish)
                        rows.append({
                            "st_confirmed_new": st_confirmed_new,
                            "realized_pnl": row["realized_pnl"],
                            "opened_at": row["opened_at"],
                        })

            df = pd.DataFrame(rows)
            print(f"\n{'='*70}\n{indicator} — {direction}  (n={len(df):,})\n{'='*70}")
            if len(df) < 60:
                print("Örneklem çok küçük.")
                continue

            baseline = _pf(df)
            print(f"{'grup':22} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
            print(f"{'baseline (tümü)':22} {baseline.get('n',0):>7} {baseline.get('wr',0):>6} "
                  f"{baseline.get('ort_%',0):>8} {baseline.get('pf',0):>7}")

            confirmed = df[df["st_confirmed_new"]]
            unconfirmed = df[~df["st_confirmed_new"]]
            s_c = _pf(confirmed)
            print(f"{'st_confirmed_YENİ=True':22} {s_c.get('n',0):>7} {s_c.get('wr',0):>6} "
                  f"{s_c.get('ort_%',0):>8} {s_c.get('pf',0):>7}")
            s_u = _pf(unconfirmed)
            print(f"{'st_confirmed_YENİ=False':22} {s_u.get('n',0):>7} {s_u.get('wr',0):>6} "
                  f"{s_u.get('ort_%',0):>8} {s_u.get('pf',0):>7}")

            t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
            mid = t_min + (t_max - t_min) / 2
            first_c = confirmed[confirmed["opened_at"] < mid]
            second_c = confirmed[confirmed["opened_at"] >= mid]
            first_u = unconfirmed[unconfirmed["opened_at"] < mid]
            second_u = unconfirmed[unconfirmed["opened_at"] >= mid]
            print("-- split-period (True) --")
            print(f"  ilk yarı: PF={_pf(first_c).get('pf',0):>6} (n={_pf(first_c).get('n',0)}) | "
                  f"ikinci yarı: PF={_pf(second_c).get('pf',0):>6} (n={_pf(second_c).get('n',0)})")
            print("-- split-period (False) --")
            print(f"  ilk yarı: PF={_pf(first_u).get('pf',0):>6} (n={_pf(first_u).get('n',0)}) | "
                  f"ikinci yarı: PF={_pf(second_u).get('pf',0):>6} (n={_pf(second_u).get('n',0)})")


if __name__ == "__main__":
    run()
