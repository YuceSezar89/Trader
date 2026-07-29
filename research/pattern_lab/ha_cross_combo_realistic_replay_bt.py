"""
HA_Cross için gerçekçi SL/TP/trailing replay — rsi_cross_combo_realistic_replay_bt.py'nin
HA_Cross replikasyonu (24 Tem 2026, kullanıcı isteği, "test edilecekler"
listesinin son maddesi).

Risk kuralları RSI_Cross testiyle BİREBİR aynı (signals/risk_policy.py +
signals/trailing.py): SL=giriş ATR(14)×1.5, TP=ATR×3.0, TP'de trailing
aktive olur/kapatmaz, aynı barda SL+TP ikisi menzildeyse SL önce, 3x
kaldıraç, $25 pozisyon, backtest-özel 500-bar zaman aşımı sınırı.

Popülasyonlar:
  - Long ÜÇLÜ: all_up + TA-base(pct>=55) + kovalama(pct>=90&tırmanıyor) + HA-hizalı
    (eşik 24 Tem'de 80'den 90'a güncellendi, [[project_ha_cross_kovalama_threshold_sweep_24tem]])
  - Short İKİLİ: all_up + TA-base(pct<=45) + kovalama(pct<=20&düşüyor), HA'SIZ
    ([[project_ha_cross_short_24tem]]'de HA'nın katkı sağlamadığı bulundu)

Kullanım: python -m research.pattern_lab.ha_cross_combo_realistic_replay_bt
(önce ha_cross_ta_triple_combo_bt.py VE ha_cross_ta_triple_combo_short_bt.py
çalıştırılmış olmalı — cache'lerini kullanır)
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import _deep_validate_usd, _fetch_5m_full, _simulate, _pnl_usd
from research.pattern_lab.ha_cross_ta_triple_combo_bt import _HA_CACHE_PATH as _LONG_CACHE_PATH
from research.pattern_lab.ha_cross_ta_triple_combo_short_bt import _CACHE_PATH as _SHORT_CACHE_PATH
from research.pattern_lab.do_open_streak_full_clean_bt import _conn

_BASE_TH_LONG = 55
_EXTREME_TH_LONG = 90  # 24 Tem: [[project_ha_cross_kovalama_threshold_sweep_24tem]] ile 80'den güncellendi
_BASE_TH_SHORT = 45
_EXTREME_TH_SHORT = 20


def _fetch_signal_open_prices(cur, direction: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'HA_Cross' AND signal_type = %s AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        """,
        (direction,),
    )
    return pd.DataFrame(cur.fetchall(), columns=["symbol", "opened_at", "open_price"])


def _run_replay(label: str, qualifying: pd.DataFrame, direction: str) -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, direction)
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"[{label}] açılış fiyatı eşleşen n={len(merged)} (teorik popülasyon n={len(qualifying)})")

    results = []
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_full(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < 14 or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            entry_price = float(row["open_price"])
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, direction, entry_price, float(atr[idx]))
            pnl_usd = _pnl_usd(direction, entry_price, exit_price)
            results.append({
                "symbol": sym, "opened_at": row["opened_at"], "pnl_usd": pnl_usd,
                "reason": reason, "bars_held": bars_held,
            })
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    return pd.DataFrame(results)


def main() -> None:
    print("=" * 78)
    print("HA_Cross LONG ÜÇLÜ (all_up + TA-base>=55 + kovalama>=80 + HA-hizalı) — gerçekçi replay")
    print("=" * 78)
    long_df = pd.read_parquet(_LONG_CACHE_PATH)
    long_df = long_df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"])
    ta_base = (long_df["pct_1h"] >= _BASE_TH_LONG) & (long_df["pct_4h"] >= _BASE_TH_LONG)
    kovalama = ((long_df["pct_1h"] >= _EXTREME_TH_LONG) & (long_df["slope_1h"] > 0)) | \
               ((long_df["pct_4h"] >= _EXTREME_TH_LONG) & (long_df["slope_4h"] > 0))
    ha = (long_df["ha_bull_1h"] > 0.5) & (long_df["ha_bull_4h"] > 0.5)
    long_qualifying = long_df[ta_base & kovalama & ha][["symbol", "opened_at"]].reset_index(drop=True)

    long_result = _run_replay("HA_Cross Long üçlü", long_qualifying, "Long")
    if not long_result.empty:
        long_result.to_parquet(os.path.join(os.path.dirname(__file__), "_cache_replay_ha_long.parquet"))
        _deep_validate_usd("HA_Cross Long üçlü — gerçekçi $ PnL", long_result)

    print("\n" + "=" * 78)
    print("HA_Cross SHORT İKİLİ (all_up + TA-base<=45 + kovalama<=20, HA'sız) — gerçekçi replay")
    print("=" * 78)
    short_df = pd.read_parquet(_SHORT_CACHE_PATH)
    short_df = _add_all_up(short_df)
    short_df = short_df[short_df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"])
    ta_base_s = (short_df["pct_1h"] <= _BASE_TH_SHORT) & (short_df["pct_4h"] <= _BASE_TH_SHORT)
    kovalama_s = ((short_df["pct_1h"] <= _EXTREME_TH_SHORT) & (short_df["slope_1h"] < 0)) | \
                 ((short_df["pct_4h"] <= _EXTREME_TH_SHORT) & (short_df["slope_4h"] < 0))
    short_qualifying = short_df[ta_base_s & kovalama_s][["symbol", "opened_at"]].reset_index(drop=True)

    short_result = _run_replay("HA_Cross Short ikili", short_qualifying, "Short")
    if not short_result.empty:
        short_result.to_parquet(os.path.join(os.path.dirname(__file__), "_cache_replay_ha_short.parquet"))
        _deep_validate_usd("HA_Cross Short ikili — gerçekçi $ PnL", short_result)


if __name__ == "__main__":
    main()
