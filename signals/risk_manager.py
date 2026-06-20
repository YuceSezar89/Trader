"""
RiskManager — aktif sinyallerin SL/TP seviyelerini izler.

Her fiyat güncellemesinde check_price() çağrılır.
SL veya TP tetiklenince signal_lifecycle_manager üzerinden sinyali kapatır.
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session
from database.models import Signal
from signals.signal_lifecycle_manager import _calc_pnl

logger = logging.getLogger(__name__)


class RiskManager:

    async def check_price(self, symbol: str, current_price: float) -> list[int]:
        """
        Sembolün aktif sinyallerini fiyata karşı kontrol eder.
        Tetiklenen sinyallerin id listesini döner.
        """
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

                for sig in actives:
                    reason = self._check_levels(sig, current_price)
                    if reason:
                        await self._close(session, sig, current_price, reason)
                        triggered.append(sig.id)
                        logger.info(
                            "[%s] %s id=%d %s tetiklendi @ %.6f (SL=%.6f TP=%.6f)",
                            symbol, sig.signal_type, sig.id, reason,
                            current_price, sig.stop_loss_price, sig.take_profit_price,
                        )

                if triggered:
                    await session.commit()

            except Exception as exc:
                await session.rollback()
                logger.error("RiskManager hatası [%s]: %s", symbol, exc, exc_info=True)

        return triggered

    @staticmethod
    def _check_levels(sig: Signal, price: float) -> Optional[str]:
        sl = sig.stop_loss_price
        tp = sig.take_profit_price

        if sig.signal_type == "Long":
            if sl is not None and price <= sl:
                return "stop_loss"
            if tp is not None and price >= tp:
                return "take_profit"
        else:
            if sl is not None and price >= sl:
                return "stop_loss"
            if tp is not None and price <= tp:
                return "take_profit"
        return None

    @staticmethod
    async def _close(
        session: AsyncSession,
        sig: Signal,
        close_price: float,
        reason: str,
    ) -> None:
        from datetime import datetime  # pylint: disable=import-outside-toplevel
        sig.status       = "closed"
        sig.closed_at    = datetime.now()
        sig.close_price  = close_price
        sig.close_reason = reason
        sig.realized_pnl = _calc_pnl(sig.signal_type, float(sig.open_price), close_price)
        session.add(sig)


risk_manager = RiskManager()
