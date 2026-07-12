"""
V100_kırmızı (rekor 50-bar hacimli kırmızı mum, [[project_pattern_lab]] — bugün
bulunan, her ufukta [15dk-4s] iki yarıda da tutarlı edge gösteren tek yeni
bulgu) RSI_Cross'a rejim/filtre olarak uygular. rsi_cross_volbreakout_regime_bt.py
ile aynı desen: threshold_optimizer'ın 3-kapılı disiplini.

Durum: son `LOOKBACK_BARS` bar içinde (mevcut bar dahil) bir V100_kırmızı olayı
oldu mu (1/0) — vpmv_v100_bt.py'nin bulduğu edge'in ufku (~2-4 saat, 8-16 bar
@15m) ile eşleşecek şekilde LOOKBACK_BARS=8 (2 saat) seçildi.

Look-ahead yok: normalize_volume_0_100 SADECE geçmiş 50 barla (rolling),
lookback penceresi de geçmişe bakıyor. merge_regime'in mevcut (opened_at-15dk)
kesme mantığıyla bar-kapanış güvenliği korunuyor.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    INDICATOR, _fetch_regime, _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position
from utils.preprocessing import normalize_volume_0_100  # pylint: disable=wrong-import-position

BAR_DURATION = pd.Timedelta(minutes=15)
LOOKBACK_BARS = 8  # ~2 saat @ 15m — vpmv_v100_bt.py'de edge'in en güçlü olduğu ufuk


def _v100_kirmizi_recent_series(g: pd.DataFrame) -> pd.Series:
    v_score = normalize_volume_0_100(g["volume"], window=50)
    is_v100_kirmizi = (v_score >= 99.99) & (g["close"] < g["open"])
    recent = is_v100_kirmizi.rolling(LOOKBACK_BARS, min_periods=1).max()
    return recent.fillna(0.0)


def _v100_yesil_recent_series(g: pd.DataFrame) -> pd.Series:
    v_score = normalize_volume_0_100(g["volume"], window=50)
    is_v100_yesil = (v_score >= 99.99) & (g["close"] > g["open"])
    recent = is_v100_yesil.rolling(LOOKBACK_BARS, min_periods=1).max()
    return recent.fillna(0.0)


STATE_FNS = {
    "v100_kirmizi_recent": _v100_kirmizi_recent_series,
    "v100_yesil_recent": _v100_yesil_recent_series,
}


def run(indicator: str = INDICATOR) -> None:
    for col_name, state_fn in STATE_FNS.items():
        for direction in ("Long", "Short"):
            sig_df = _fetch_signals(indicator, direction)
            if len(sig_df) < 50:
                print(f"{indicator} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
                continue

            regime_df = _fetch_regime(sig_df["symbol"].unique().tolist(), "15m", state_fn, col_name)
            merged = _merge_regime(sig_df, regime_df, col_name, BAR_DURATION)
            print(f"{indicator} — {direction}: {len(sig_df):,} sinyal, {len(merged):,} durumla eşleşti "
                  f"(son {LOOKBACK_BARS} barda {col_name} oranı={merged[col_name].mean():.2%})")

            label = f"{indicator} — {direction} — {col_name} (son {LOOKBACK_BARS} bar)"
            _run_single_var_on_df(label, merged, col_name)


if __name__ == "__main__":
    run()
