"""
DO Kırılımı + Ardışık Yeşil Mum + Gauss Büyüklük dedektörü (Long-only).

Kaynak: Pattern Lab araştırması (research/pattern_lab/do_break_gauss_*.py,
9-10 Tem 2026) — kullanıcının canlı gözlemi (en çok yükselen coinlerde D-open'dan
tepki alıp art arda yeşil mum yakma → büyük hareket) + arşivinden bulduğu bir
Pine script'in ("Ardışık Sistemler.txt") ardışık mum + Gauss toplamı (n(n+1)/2)
fikirlerinin birleşimi.

Sinyal: DO'nun kapanışla yukarı kesilmesi (do_break) → TAM 3 ardışık yeşil mum →
o 3 mumda kat edilen mesafenin (streak başlangıcının low'undan şu anki high'a)
Gauss-ağırlıklı büyüklüğü eşik üzerinde. 390 sembol/45 gün/15m/24h ufkunda
split-period + OOS ekonomik test ile doğrulandı (+$367-1175/ay, yönteme göre).

SADECE LONG — Short tarafı simetrik çıkmadı (Gauss refinement işe yaramadı,
split-period tutarsız), bilinçli olarak dahil edilmedi.

Çıkış: SL=3×ATR TEK BAŞINA (TP yok, breakeven yok — backtestte en iyi çıkan
yöntem: TP/breakeven eklemek asıl kazandıran büyük hareketi erken kesiyordu),
24h/96-bar (15m) timeout — bkz. paper_trade_manager.py::max_hold_hours.

Pozisyon boyutlandırma: volatilite-ayarlı (hocanın DevisSoTrader
risk_management.py::position_size_volatility_adjusted fikri) — sabit $ risk
hedefi, ATR'ye göre pozisyon büyüklüğü otomatik ayarlanır (bkz. live_data_manager.py
_check_do_open_streak).

Her çağrıda pencerenin tamamı replay edilir — kalıcı state yok (do_kirilimi.py
ile aynı felsefe, restart/gap-proof).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from indicators.core import calculate_atr
from signals.do_kirilimi import _daily_open

logger = logging.getLogger(__name__)

STREAK_THRESHOLD = 3
GAUSS_THRESHOLD = 4.5     # research/pattern_lab/do_break_gauss_economic_bt.py OOS eşiği (~4.5)
SL_ATR_MULT = 3.0
MAX_HOLD_HOURS = 24.0
MIN_BARS = 100


def _gauss_sum(x: float) -> float:
    return x * (x + 1) / 2.0


class DoOpenStreakDetector:
    """Sembolün 15m penceresinde son kapanan barda entry olup olmadığını söyler."""

    def check(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        try:
            if df is None or len(df) < MIN_BARS:
                return None
            d = df.tail(200).reset_index(drop=True).copy()
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

            daily_open, _ = _daily_open(ts, o)

            # do_break gate: DO'nun kapanışla yukarı kesilmesinden itibaren,
            # ardışık yeşil mum bozulana (kırmızı mum) kadar aktif kalır.
            # research/pattern_lab/do_open_streak_bt.py::_do_break_gate ile birebir.
            prev_c = np.roll(c, 1)
            prev_c[0] = np.nan
            do_break = (c > daily_open) & (prev_c <= daily_open) & np.isfinite(daily_open)
            is_long = c > o

            gate_active = False
            count_long = 0
            start_low = np.nan
            for i in range(n):
                if do_break[i]:
                    gate_active = True
                elif not is_long[i]:
                    gate_active = False

                if is_long[i]:
                    count_long += 1
                    if count_long == 1 or np.isnan(start_low):
                        start_low = l[i]
                else:
                    count_long = 0
                    start_low = np.nan

            last = n - 1
            if not gate_active or count_long != STREAK_THRESHOLD:
                return None

            long_perc = (h[last] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val < GAUSS_THRESHOLD:
                return None

            atr_series = calculate_atr(d, period=14)
            atr_val = float(atr_series.iloc[last])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None

            entry_price = c[last]
            sl_price = entry_price - SL_ATR_MULT * atr_val

            return {
                "price": entry_price,
                "atr": atr_val,
                "sl_price": sl_price,
                "tp_price": None,
                "gauss_val": gauss_val,
                "long_perc": long_perc,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("[DoOpenStreak] %s dedektör hatası: %s", symbol, exc, exc_info=True)
            return None


do_open_streak_detector = DoOpenStreakDetector()
