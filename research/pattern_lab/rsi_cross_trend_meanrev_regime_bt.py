"""
rsi_cross_volbreakout_regime_bt.py'nin AYNI deseniyle, DevisSoTrader'ın kalan
iki ajanının (trend_follower, mean_reversion) durumunu TETİKLEYİCİ değil
REJİM/FİLTRE olarak RSI_Cross sinyallerine uygular — hocanın MultiAgentSystem'i
bu ajanları zaten tek başına tetiklemiyor, oy/onay olarak kullanıyor
([[project-turtle-traders]]).

trend_follower: coin bazlı SMA20/50 uptrend/downtrend durumu (4h — ajanın
tercih ettiği TF'ye en yakını, do_kirilimi'nin de kullandığı ufuk).
mean_reversion: RSI+Bollinger aşırı-satım/alım durumu (15m).

Look-ahead: cagg_{interval}.bucket bar AÇILIŞ zamanı (bkz. mtf_helpers.py).
merge_asof'a opened_at yerine (opened_at - bar_süresi) veriliyor — bar
GERÇEKTEN kapanmadan durumu kullanılmıyor.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.devissotrader_agents_bt import (  # pylint: disable=wrong-import-position
    _mean_reversion_state_series, _trend_state_series,
)
from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    INDICATOR, _fetch_regime, _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position

TESTS = [
    ("trend_state", "4h", _trend_state_series, pd.Timedelta(hours=4)),
    ("meanrev_state", "15m", _mean_reversion_state_series, pd.Timedelta(minutes=15)),
]


def run() -> None:
    for col_name, interval, state_fn, bar_duration in TESTS:
        for direction in ("Long", "Short"):
            sig_df = _fetch_signals(INDICATOR, direction)
            if len(sig_df) < 50:
                print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
                continue

            regime_df = _fetch_regime(sig_df["symbol"].unique().tolist(), interval, state_fn, col_name)
            merged = _merge_regime(sig_df, regime_df, col_name, bar_duration)
            print(f"{INDICATOR} — {direction} — {col_name} ({interval}): {len(sig_df):,} sinyal, "
                  f"{len(merged):,} rejim durumuyla eşleşti (ort={merged[col_name].mean():.2f})")

            label = f"{INDICATOR} — {direction} — {col_name} ({interval} rejimi)"
            _run_single_var_on_df(label, merged, col_name)


if __name__ == "__main__":
    run()
