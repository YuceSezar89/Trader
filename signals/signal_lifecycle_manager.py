"""
Sinyal yaşam döngüsü — temiz state machine.

active → closed  (reversal | timeout | manual)

Geçiş kuralları:
  - Aynı key, ters yön   → aktifi kapat (reversal), yenisini aç
  - Aynı key, aynı yön   → sadece skorları güncelle
  - Aktif sinyal yok      → aç
  - Sweeper               → timeout eşiği geçmişse kapat
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Europe/Istanbul")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session
from database.models import Signal
from signals.risk_policy import default_policy

logger = logging.getLogger(__name__)

TIMEOUT_HOURS: dict[str, int] = {
    "1m":  4,
    "5m":  24,
    "15m": 48,
    "1h":  7 * 24,
    "4h":  21 * 24,
    "1d":  60 * 24,
}
_DEFAULT_TIMEOUT = 24


def _calc_pnl(signal_type: str, open_price: float, close_price: float) -> float:
    if open_price == 0:
        return 0.0
    if signal_type == "Long":
        return (close_price - open_price) / open_price * 100
    return (open_price - close_price) / open_price * 100


class SignalLifecycleManager:

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, symbol: str, interval: str) -> asyncio.Lock:
        key = f"{symbol}:{interval}"
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def process(
        self,
        signal_data: Dict[str, Any],
        current_price: Optional[float] = None,
    ) -> Optional[int]:
        """
        Yeni sinyal işler.
        Returns: yeni sinyalin id'si, sadece güncelleme yapıldıysa None.
        """
        symbol     = signal_data["symbol"]
        interval   = signal_data["interval"]
        indicators = signal_data["indicators"]
        sig_type   = signal_data["signal_type"]
        open_price = float(signal_data["open_price"])

        async with self._get_lock(symbol, interval):
            async with get_session() as session:
                try:
                    active = await self._get_active(session, symbol, interval)

                    if active:
                        if active.signal_type == sig_type:
                            await self._update_scores(session, active, signal_data)
                            logger.debug(
                                "[%s] %s %s skor güncellendi (%s)", symbol, interval, sig_type, indicators
                            )
                            return None

                        close_px = current_price or open_price
                        await self._close(session, active, close_px, "reversal")
                        logger.info(
                            "[%s] %s %s kapatıldı → %s açılıyor",
                            symbol, interval, active.signal_type, sig_type,
                        )

                    atr_val = signal_data.get("atr") or 0.0
                    if atr_val > 0:
                        features = {
                            "vpms_score":  signal_data.get("vpms_score"),
                            "mtf_score":   signal_data.get("mtf_score"),
                            "interval":    interval,
                        }
                        levels = default_policy.calculate_levels(
                            sig_type, open_price, float(atr_val), features
                        )
                        sl_price   = levels.sl_price
                        tp_price   = levels.tp_price
                        sl_mult    = levels.sl_multiplier
                        tp_mult    = levels.tp_multiplier
                    else:
                        sl_price = tp_price = sl_mult = tp_mult = None

                    new_sig = Signal(
                        symbol            = symbol,
                        interval          = interval,
                        indicators        = indicators,
                        signal_type       = sig_type,
                        opened_at         = signal_data.get("opened_at", datetime.now(_IST)),
                        open_price        = open_price,
                        status            = "active",
                        vpms_score        = signal_data.get("vpms_score"),
                        mtf_score         = signal_data.get("mtf_score"),
                        st_confirmed      = signal_data.get("st_confirmed"),
                        rsi               = signal_data.get("rsi"),
                        strength          = signal_data.get("strength"),
                        atr               = signal_data.get("atr"),
                        alpha             = signal_data.get("alpha"),
                        beta              = signal_data.get("beta"),
                        sharpe_ratio      = signal_data.get("sharpe_ratio"),
                        sortino_ratio     = signal_data.get("sortino_ratio"),
                        calmar_ratio      = signal_data.get("calmar_ratio"),
                        information_ratio = signal_data.get("information_ratio"),
                        oi_data           = signal_data.get("oi_data"),
                        stop_loss_price   = sl_price,
                        take_profit_price = tp_price,
                        sl_multiplier     = sl_mult,
                        tp_multiplier     = tp_mult,
                        z_score_entry     = signal_data.get("z_score_entry"),
                        is_confluence     = signal_data.get("is_confluence", False),
                        vpmv_pre_avg      = signal_data.get("vpmv_pre_avg"),
                        vpmv_slope        = signal_data.get("vpmv_slope"),
                        vpmv_ratio        = signal_data.get("vpmv_ratio"),
                    )
                    session.add(new_sig)
                    await session.flush()

                    if active:
                        active.closed_by = new_sig.id

                    await session.commit()
                    logger.info(
                        "[%s] %s %s sinyal açıldı (id=%s, %s)",
                        symbol, interval, sig_type, new_sig.id, indicators,
                    )
                    return new_sig.id

                except Exception as exc:
                    await session.rollback()
                    logger.error("[%s] sinyal işleme hatası: %s", symbol, exc, exc_info=True)
                    raise

    async def sweep_timeouts(self) -> int:
        """Timeout eşiğini geçmiş aktif sinyalleri kapatır."""
        closed = 0
        async with get_session() as session:
            try:
                result = await session.execute(
                    select(Signal).where(Signal.status == "active")
                )
                actives = result.scalars().all()

                now = datetime.now(_IST)
                for sig in actives:
                    hours = TIMEOUT_HOURS.get(sig.interval, _DEFAULT_TIMEOUT)
                    if sig.opened_at < now - timedelta(hours=hours):
                        await self._close(session, sig, float(sig.open_price), "timeout")
                        closed += 1

                if closed:
                    await session.commit()
                    logger.info("Timeout sweep: %d sinyal kapatıldı", closed)

            except Exception as exc:
                await session.rollback()
                logger.error("Sweep hatası: %s", exc, exc_info=True)

        return closed

    async def close_stale(self, signal_id: int, close_price: float, reason: str = "reconciliation") -> bool:
        """Startup reconciliation veya harici tetikleyiciler için sinyal kapatır."""
        async with get_session() as session:
            try:
                result = await session.execute(
                    select(Signal).where(
                        Signal.id == signal_id,
                        Signal.status == "active",
                    )
                )
                sig = result.scalar_one_or_none()
                if not sig:
                    return False
                await self._close(session, sig, close_price, reason)
                await session.commit()
                return True
            except Exception as exc:
                await session.rollback()
                logger.error("Stale kapatma hatası (id=%s): %s", signal_id, exc, exc_info=True)
                return False

    async def manual_close(self, signal_id: int, close_price: float) -> bool:
        async with get_session() as session:
            try:
                result = await session.execute(
                    select(Signal).where(
                        Signal.id == signal_id,
                        Signal.status == "active",
                    )
                )
                sig = result.scalar_one_or_none()
                if not sig:
                    return False
                await self._close(session, sig, close_price, "manual")
                await session.commit()
                return True
            except Exception as exc:
                await session.rollback()
                logger.error("Manuel kapatma hatası: %s", exc, exc_info=True)
                return False

    async def _get_active(
        self, session: AsyncSession, symbol: str, interval: str
    ) -> Optional[Signal]:
        result = await session.execute(
            select(Signal).where(
                Signal.symbol   == symbol,
                Signal.interval == interval,
                Signal.status   == "active",
            ).order_by(Signal.id.desc()).with_for_update()
        )
        actives = result.scalars().all()
        if not actives:
            return None
        if len(actives) > 1:
            for stale in actives[1:]:
                await self._close(session, stale, float(stale.open_price), "reconciliation")
            logger.warning(
                "[%s] %s: %d duplikat aktif sinyal temizlendi",
                symbol, interval, len(actives) - 1,
            )
        return actives[0]

    async def _close(
        self,
        session: AsyncSession,
        sig: Signal,
        close_price: float,
        reason: str,
    ) -> None:
        sig.status       = "closed"
        sig.closed_at    = datetime.now(_IST)
        sig.close_price  = close_price
        sig.close_reason = reason
        sig.realized_pnl = _calc_pnl(sig.signal_type, float(sig.open_price), close_price)
        session.add(sig)

    async def _update_scores(
        self, session: AsyncSession, sig: Signal, data: Dict[str, Any]
    ) -> None:
        if "vpms_score" in data:
            sig.vpms_score = data["vpms_score"]
        if "mtf_score" in data:
            sig.mtf_score = data["mtf_score"]
        if "st_confirmed" in data:
            sig.st_confirmed = data["st_confirmed"]
        session.add(sig)
        await session.commit()


signal_lifecycle_manager = SignalLifecycleManager()
