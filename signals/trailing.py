"""
Ortak trailing-stop algoritması — Signal ve PaperTrade nesneleri için.

Akış:
  1. Trailing aktif değilken fiyat SL'e değerse → "stop_loss".
  2. Fiyat TP'ye ulaşırsa trailing_stop_price aktive edilir (kapatma yok).
  3. Trailing aktifken fiyat lehte gittikçe seviye güncellenir.
  4. Fiyat trailing seviyesine dönerse → "trailing_stop".
"""

from typing import Optional


def update_trailing(pos, price: float, dist: float) -> Optional[str]:
    """pos: signal_type, stop_loss_price, take_profit_price,
    trailing_stop_price alanları olan ORM nesnesi (Signal | PaperTrade).
    trailing_stop_price yerinde güncellenebilir.
    """
    sl    = pos.stop_loss_price
    tp    = pos.take_profit_price
    trail = pos.trailing_stop_price

    if pos.signal_type == "Long":
        if trail is None:
            if sl is not None and price <= float(sl):
                return "stop_loss"
            if tp is not None and price >= float(tp):
                pos.trailing_stop_price = price - dist
                return None
        else:
            new_trail = price - dist
            if new_trail > float(trail):
                pos.trailing_stop_price = new_trail
            if price <= float(pos.trailing_stop_price):
                return "trailing_stop"

    else:  # Short
        if trail is None:
            if sl is not None and price >= float(sl):
                return "stop_loss"
            if tp is not None and price <= float(tp):
                pos.trailing_stop_price = price + dist
                return None
        else:
            new_trail = price + dist
            if new_trail < float(trail):
                pos.trailing_stop_price = new_trail
            if price >= float(pos.trailing_stop_price):
                return "trailing_stop"

    return None
