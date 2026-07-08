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

from config import Config
from database.engine import get_session, run_with_db_timeout
from database.models import Signal
from signals.signal_lifecycle_manager import _calc_pnl
from signals.trailing import update_trailing

logger = logging.getLogger(__name__)


class RiskManager:

    def __init__(self) -> None:
        self._active_symbols: set[str] = set()

    async def load_active_symbols(self) -> None:
        """Startup'ta aktif sinyallerin sembollerini bellekte önbelleğe al."""
        async def _do_load() -> set[str]:
            async with get_session() as session:
                result = await session.execute(
                    select(Signal.symbol).where(
                        Signal.status == "active",
                        Signal.stop_loss_price.isnot(None),
                    )
                )
                return {row[0] for row in result.all()}

        try:
            self._active_symbols = await run_with_db_timeout(_do_load())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[RiskManager] aktif sembol yükleme zaman aşımı: %s", exc)
            self._active_symbols = set()
        logger.info("[RiskManager] %d aktif sembol yüklendi", len(self._active_symbols))

    def register(self, symbol: str) -> None:
        """Yeni sinyal açıldığında sembolü set'e ekle."""
        self._active_symbols.add(symbol)

    async def check_all_prices(self, prices: dict[str, float]) -> None:
        """Tüm aktif sinyalleri tek DB sorgusunda kontrol et."""
        if not self._active_symbols:
            return
        symbols_to_check = [s for s in self._active_symbols if s in prices]
        if not symbols_to_check:
            return

        async def _do_check() -> None:
            async with get_session() as session:
                try:
                    result = await session.execute(
                        select(Signal).where(
                            Signal.status == "active",
                            Signal.stop_loss_price.isnot(None),
                            Signal.symbol.in_(symbols_to_check),
                        )
                    )
                    actives = result.scalars().all()
                    if not actives:
                        return

                    changed = False
                    closed_symbols: set[str] = set()
                    for sig in actives:
                        price = prices.get(sig.symbol)
                        if price is None:
                            continue
                        old_trail = sig.trailing_stop_price
                        reason = self._update_trailing(sig, price)
                        if reason:
                            await self._close(session, sig, price, reason)
                            closed_symbols.add(sig.symbol)
                            logger.info(
                                "[%s] %s id=%d %s @ %.6f",
                                sig.symbol, sig.signal_type, sig.id, reason, price,
                            )
                            changed = True
                        elif sig.trailing_stop_price != old_trail:
                            changed = True

                    if changed:
                        await session.commit()

                    active_syms = {s.symbol for s in actives if s.status == "active"}
                    for sym in closed_symbols:
                        if sym not in active_syms:
                            self._active_symbols.discard(sym)

                except Exception as exc:
                    await session.rollback()
                    logger.error("RiskManager batch hatası: %s", exc, exc_info=True)

        try:
            await run_with_db_timeout(_do_check())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("RiskManager batch zaman aşımı: %s", exc)

    @staticmethod
    def _trail_distance(sig: Signal) -> float:
        if sig.atr and sig.sl_multiplier:
            return float(sig.atr) * float(sig.sl_multiplier)
        if sig.stop_loss_price and sig.open_price:
            return abs(float(sig.open_price) - float(sig.stop_loss_price))
        return float(sig.open_price) * 0.005

    @staticmethod
    def _early_exit_check(sig: Signal, price: float) -> bool:
        cfg = Config.VPM
        if not cfg.get("EARLY_EXIT_ENABLED", False):
            return False
        if sig.interval not in ("5m", "15m"):
            return False
        if not sig.atr or float(sig.atr) <= 0:
            return False
        iv_min = Config.INTERVAL_MINUTES.get(sig.interval, 5)
        elapsed_bars = (datetime.now() - sig.opened_at).total_seconds() / 60 / iv_min
        if elapsed_bars > cfg.get("EARLY_EXIT_MAX_BARS", 10):
            return False
        mae_thr = float(cfg.get("EARLY_EXIT_MAE_ATR", -1.5))
        atr = float(sig.atr)
        entry = float(sig.open_price)
        adverse = (price - entry) / atr if sig.signal_type == "Long" else (entry - price) / atr
        return adverse <= mae_thr

    @staticmethod
    def _update_trailing(sig: Signal, price: float) -> Optional[str]:
        if RiskManager._early_exit_check(sig, price):
            return "stop_loss"
        return update_trailing(sig, price, RiskManager._trail_distance(sig))

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
