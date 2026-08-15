"""
Totalamount Rank-1 dedektörü — Devisso Döngüsü (13 Ağu 2026).

Hoca Telegram külliyatındaki (S10_SIRALAMA, "Devis'So döngüsü") doktrin:
"Sinyal oluştuğunda ilk muma göre Totalamount'a göre long/short sıralaması
yaptır" (faraday, 21.03.2025) — aktif Supertrend Long sinyali olan
semboller arasında, Totalamount'ta (sinyalin açılış fiyatına göre kümülatif
% fiyat hareketi) 1. sıraya çıkan sembole Long aç.

Sıfırlama noktası SİNYAL BAŞLANGICI'dır (per-signal reset), sabit bir UTC
döngüsü değil — "Sinyal Mumu Ratioları" Pine referansındaki entryPrice/pct
mantığıyla birebir: `entryPrice := close` sinyal anında, `pct = (close -
entryPrice) / entryPrice * 100` sonrasında her barda. İlk sürüm (13 Ağu,
aynı gün içinde) `signals/ta_kovalama_gate.py::_net_ta_series`'ten 12 saatlik
global UTC-döngü sıfırlamasını kopyalamıştı — o formül farklı bir özellik
(RSI_Cross percentile filtresi) için tasarlanmıştı ve doktrinle uyuşmuyordu,
kullanıcı denetiminde düzeltildi.

Çıkış: sabit SL/TP değil — Supertrend o sembolde Short'a döndüğünde
(ters sinyal) kapanır (bkz. live_data_manager.py::_totalamount_rank1_loop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_ta_series(df: pd.DataFrame, open_price: float, bars_since_signal: int) -> np.ndarray:
    """Sinyalin açılış fiyatına (open_price) göre basit % değişim serisi —
    SADECE sinyal açıldığından bu yana olan barlar için (bars_since_signal
    kadar geriye, elde olan df bu kadar geriye gitmiyorsa mevcut kadarı).
    Tamamen vektörize — Python döngüsü yok."""
    if df is None or open_price in (None, 0):
        return np.zeros(0)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n == 0:
        return np.zeros(0)
    start_idx = max(0, n - 1 - max(bars_since_signal, 0))
    close_slice = close[start_idx:]
    return (close_slice - float(open_price)) / float(open_price) * 100.0
