"""
Jenerik multi-timeframe HA-yön yardımcıları — "Ultra hizalama" (v2-14/v2-15)
testlerinin ortak çekirdeği. Herhangi bir sinyal ailesine özgü değil, sadece
"bir sembolün bir TF'deki HA yönü, verilen andan ÖNCE ne durumdaydı" sorusunu
cevaplar (look-ahead yok — sadece geçmiş barlar kullanılır).

Bağımlılık döngüsünü önlemek için ayrı bir modülde tutuluyor: bu fonksiyonları
kullanan test dosyaları (ha_cross_mtf_alignment_bt.py, do_open_streak_bt.py)
farklı importlar üzerinden birbirine dolaylı bağımlı olabiliyor; bu dosya
sadece rsi_cross_vpmv_jump_bt.py'ye (MIN_HISTORY, _fetch_symbol_history) ve
indicators/core.py'ye bağımlı, hiçbir pattern_lab modülünü geri import etmiyor.

DÜZELTME (11 Tem 2026, bkz. project_pattern_lab v2-24): `_last_direction_before`
cagg bucket'ları (bar AÇILIŞ zamanı) doğrudan `opened_at`/`ts` ile karşılaştırıyordu
— bu, henüz KAPANMAMIŞ bir üst-TF barını (ör. 1h barı saat başından 59 dakika
sonrasına kadar) "kapanmış" sayan gerçek bir LOOK-AHEAD hatasıydı (5m sinyalin
saatin ilk 55 dakikasında gelmesi ~%92 olasılıkla bu hataya düşürüyordu). Artık
her TF'nin kendi süresi hesaba katılıyor: bir bucket'ın GERÇEKTEN kapanmış
sayılması için bucket+süre <= verilen an olmalı.
"""
import numpy as np
import pandas as pd

from indicators.core import calculate_ha
from research.pattern_lab.rsi_cross_vpmv_jump_bt import MIN_HISTORY, _fetch_symbol_history

TF_DURATION = {
    "5m": np.timedelta64(5, "m"),
    "15m": np.timedelta64(15, "m"),
    "1h": np.timedelta64(1, "h"),
    "4h": np.timedelta64(4, "h"),
}


def _ha_direction_series(interval: str, symbol: str) -> pd.DataFrame | None:
    hist = _fetch_symbol_history(symbol, interval)
    if len(hist) < MIN_HISTORY:
        return None
    hist = hist.sort_values("ts").reset_index(drop=True)
    ha = calculate_ha(hist)
    ha["bullish"] = ha["ha_close"] > ha["ha_open"]
    return ha[["ts", "bullish"]]


def _last_direction_before(dir_df: pd.DataFrame, ts_arr: np.ndarray, opened_at, tf: str) -> bool | None:
    """tf: bu ts_arr'ın ait olduğu zaman dilimi ("5m"/"15m"/"1h"/"4h") — bir
    bucket'ın GERÇEKTEN kapanmış sayılması için bucket + TF_DURATION[tf] <=
    opened_at şartı aranır (sadece bucket <= opened_at YETERSİZ, bkz. modül
    docstring'i)."""
    cutoff = np.datetime64(opened_at) - TF_DURATION[tf]
    idx = np.searchsorted(ts_arr, cutoff, side="right") - 1
    if idx < 0:
        return None
    return bool(dir_df["bullish"].iloc[idx])


def _fetch_dir_data(symbol: str, tfs: list[str]) -> dict | None:
    """Verilen TF listesi için (yön_df, ts_array) çiftlerini toplar; herhangi
    biri için yeterli geçmiş yoksa None döner (çağıran sembolü atlamalı)."""
    dir_data = {}
    for tf in tfs:
        d = _ha_direction_series(tf, symbol)
        if d is None:
            return None
        dir_data[tf] = (d, d["ts"].to_numpy())
    return dir_data


def _confirm_count(dir_data: dict, tfs: list[str], ts, want_bullish: bool) -> int | None:
    """Verilen andan (ts) ÖNCE kapanmış barlara göre, TF'lerden kaçının
    want_bullish yönüyle uyuştuğunu sayar. Herhangi bir TF için veri
    yetersizse None (çağıran olayı atlamalı)."""
    count = 0
    for tf in tfs:
        d, ts_arr = dir_data[tf]
        bull = _last_direction_before(d, ts_arr, ts, tf)
        if bull is None:
            return None
        if bull == want_bullish:
            count += 1
    return count
