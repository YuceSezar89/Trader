"""
Kanonik kline şeması — buffer'ı dolduran her yol aynı kolon setini taşımalı.

Dört yükleme yolu var: WS tick/kapanan mum, REST fetch_klines,
DB get_recent_klines, CA get_cagg_klines. 3 Tem 2026'da görüldü ki her yol
kendi elle yazılmış kolon listesini kullanıyor ve yönlü hacim (buy/sell)
yolda sessizce düşüyor (234 sembolün 0'ı tam kapsamlıydı). Bu modül tek
sözleşme tanımlar; her yol dönüşte check_kline_schema'dan geçer.

Eksik çekirdek kolon → ERROR (veri kullanılamaz durumda demektir).
Eksik yönlü kolon   → WARNING (metrikler toplam hacme düşer — görünür olsun).
Aynı kaynak+eksik kombinasyonu bir kez loglanır (log seli olmaz).
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CORE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
DIRECTIONAL_COLUMNS = ["buy_volume", "sell_volume"]

_warned: set = set()


def check_kline_schema(
    df: Optional[pd.DataFrame],
    source: str,
    expect_directional: bool = True,
) -> Optional[pd.DataFrame]:
    """Şema kontrolü yapar, df'yi olduğu gibi döndürür (akışı bozmaz)."""
    if df is None or df.empty:
        return df

    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        key = (source, "core", tuple(missing_core))
        if key not in _warned:
            _warned.add(key)
            logger.error("[KlineSchema] %s: çekirdek kolon eksik: %s", source, missing_core)

    if expect_directional:
        missing_dir = [c for c in DIRECTIONAL_COLUMNS if c not in df.columns]
        if missing_dir:
            key = (source, "dir", tuple(missing_dir))
            if key not in _warned:
                _warned.add(key)
                logger.warning("[KlineSchema] %s: yönlü hacim kolonu eksik: %s — metrikler toplam hacme düşecek", source, missing_dir)

    return df
