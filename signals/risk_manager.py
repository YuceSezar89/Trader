"""
RiskManager — aktif sinyallerin SL / trailing-stop seviyelerini izler.

Akış:
  1. SL tetiklenirse → stop_loss ile kapat.
  2. Fiyat TP seviyesine ulaşırsa → TP'de kapatma, trailing_stop_price'ı aktif et.
  3. Trailing aktifken fiyat lehte gitmeye devam ederse trailing_stop_price güncelle.
  4. Fiyat trailing_stop_price'a dönerse → trailing_stop ile kapat.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session
from database.models import Signal
from signals.signal_lifecycle_manager import _calc_pnl

logger = logging.getLogger(__name__)


class RiskManager:

    async def check_price(self, symbol: str, current_price: float) -> list[int]:
        triggered = []
        async with get_session() as session:
            try:
                result = await session.execute(
                    select(Signal).where(
                        Signal.symbol == symbol,
                        Signal.status == "active",
                        Signal.stop_loss_price.isnot(None),
                    )
                )
                actives = result.scalars().all()

                changed = False
                for sig in actives:
                    reason = self._update_trailing(sig, current_price)
                    if reason:
                        await self._close(session, sig, current_price, reason)
                        triggered.append(sig.id)
                        logger.info(
                            "[%s] %s id=%d %s @ %.6f (trail=%.6f)",
                            symbol, sig.signal_type, sig.id, reason,
                            current_price,
                            sig.trailing_stop_price or sig.stop_loss_price or 0,
                        )
                        changed = True
                    elif sig.trailing_stop_price is not None:
                        changed = True  # trailing_stop_price güncellendi

                if changed:
                    await session.commit()

            except Exception as exc:
                await session.rollback()
                logger.error("RiskManager hatası [%s]: %s", symbol, exc, exc_info=True)

        return triggered

    @staticmethod
    def _trail_distance(sig: Signal) -> float:
        if sig.atr and sig.sl_multiplier:
            return float(sig.atr) * float(sig.sl_multiplier)
        if sig.stop_loss_price and sig.open_price:
            return abs(float(sig.open_price) - float(sig.stop_loss_price))
        return float(sig.open_price) * 0.005

    @staticmethod
    def _update_trailing(sig: Signal, price: float) -> Optional[str]:
        sl    = sig.stop_loss_price
        tp    = sig.take_profit_price
        trail = sig.trailing_stop_price
        dist  = RiskManager._trail_distance(sig)

        if sig.signal_type == "Long":
            # SL kontrolü (trailing aktif değilken)
            if trail is None:
                if sl is not None and price <= float(sl):
                    return "stop_loss"
                if tp is not None and price >= float(tp):
                    # TP'ye ulaştı → trailing başlat
                    sig.trailing_stop_price = price - dist
                    return None
            else:
                # Trailing aktif: fiyat yükselince trail'i yukarı çek
                new_trail = price - dist
                if new_trail > float(trail):
                    sig.trailing_stop_price = new_trail
                # Trailing tetiklendi mi?
                if price <= float(sig.trailing_stop_price):
                    return "trailing_stop"

        else:  # Short
            if trail is None:
                if sl is not None and price >= float(sl):
                    return "stop_loss"
                if tp is not None and price <= float(tp):
                    sig.trailing_stop_price = price + dist
                    return None
            else:
                new_trail = price + dist
                if new_trail < float(trail):
                    sig.trailing_stop_price = new_trail
                if price >= float(sig.trailing_stop_price):
                    return "trailing_stop"

        return None

    @staticmethod
    async def _close(
        session: AsyncSession,
        sig: Signal,
        close_price: float,
        reason: str,
    ) -> None:
        sig.status       = "closed"
        sig.closed_at    = datetime.now()
        sig.close_price  = close_price
        sig.close_reason = reason
        sig.realized_pnl = _calc_pnl(sig.signal_type, float(sig.open_price), close_price)
        session.add(sig)


risk_manager = RiskManager()
