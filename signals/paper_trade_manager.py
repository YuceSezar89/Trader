"""
PaperTradeManager — konfluans sinyalleri için simüle edilmiş pozisyon yönetimi.

Her konfluans sinyalinde $100 sanal pozisyon açar.
RiskManager ile aynı SL/trailing mantığını kullanır.
ML için açılış anında market snapshot'ı kaydeder.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session
from database.models import PaperTrade, PaperPortfolio, Signal
from signals.signal_lifecycle_manager import _calc_pnl

logger = logging.getLogger(__name__)

STRATEGY       = "conf_100"
POSITION_USD   = 100.0
FEE_RATE       = 0.0005   # %0.05 taker fee
MAX_OPEN       = 10       # aynı anda max açık pozisyon


class PaperTradeManager:

    async def on_new_signal(
        self,
        signal_data: dict,
        signal_id: Optional[int],
        current_price: float,
        btc_z_score: Optional[float] = None,
        btc_trend: Optional[str] = None,
        funding_rate: Optional[float] = None,
    ) -> None:
        if not signal_data.get("is_confluence") or signal_id is None:
            return

        async with get_session() as session:
            try:
                # Max açık pozisyon kontrolü
                count_result = await session.execute(
                    select(func.count()).select_from(PaperTrade).where(
                        PaperTrade.strategy == STRATEGY,
                        PaperTrade.status == "open",
                    )
                )
                open_count = count_result.scalar() or 0
                if open_count >= MAX_OPEN:
                    logger.info(
                        "[PaperTrade] Max açık pozisyon (%d) doldu, %s atlandı",
                        MAX_OPEN, signal_data.get("symbol"),
                    )
                    return

                # Aynı sembol + yön için zaten açık pozisyon varsa atla
                existing = await session.execute(
                    select(PaperTrade).where(
                        PaperTrade.strategy == STRATEGY,
                        PaperTrade.status == "open",
                        PaperTrade.symbol == signal_data.get("symbol"),
                        PaperTrade.signal_type == signal_data.get("signal_type"),
                    )
                )
                if existing.scalars().first():
                    return

                # SL/TP'yi DB'den çek (lifecycle manager kaydetmiş olur)
                sig_result = await session.execute(
                    select(Signal).where(Signal.id == signal_id)
                )
                sig = sig_result.scalars().first()
                sl_price = sig.stop_loss_price if sig else None
                tp_price = sig.take_profit_price if sig else None

                # Recent win rate hesapla
                recent_win_rate = await self._recent_win_rate(
                    session, signal_data.get("symbol", "")
                )

                opened_at = signal_data.get("opened_at") or datetime.now(timezone.utc)
                if isinstance(opened_at, str):
                    opened_at = datetime.fromisoformat(opened_at)

                trade = PaperTrade(
                    signal_id=signal_id,
                    strategy=STRATEGY,
                    symbol=signal_data.get("symbol", ""),
                    signal_type=signal_data.get("signal_type", ""),
                    interval=signal_data.get("interval", ""),
                    position_usd=POSITION_USD,
                    entry_price=current_price,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    status="open",
                    opened_at=opened_at,
                    # ML snapshot
                    btc_z_score=btc_z_score,
                    btc_trend=btc_trend,
                    hour_utc=opened_at.hour if opened_at else None,
                    day_of_week=opened_at.weekday() if opened_at else None,
                    funding_rate=funding_rate,
                    recent_win_rate=recent_win_rate,
                    # Signal features
                    vpms_score=signal_data.get("vpms_score"),
                    z_score_entry=signal_data.get("z_score_entry"),
                    mtf_score=signal_data.get("mtf_score"),
                    atr=signal_data.get("atr"),
                )
                session.add(trade)
                await session.commit()

                logger.info(
                    "[PaperTrade] ★ AÇILDI %s %s %s @ %.6f | VPMV=%.1f Z=%+.2f",
                    trade.symbol, trade.signal_type, trade.interval,
                    current_price,
                    trade.vpms_score or 0, trade.z_score_entry or 0,
                )

            except Exception as exc:
                await session.rollback()
                logger.error("[PaperTrade] Açma hatası: %s", exc, exc_info=True)

    async def check_price(self, symbol: str, current_price: float) -> None:
        async with get_session() as session:
            try:
                result = await session.execute(
                    select(PaperTrade).where(
                        PaperTrade.symbol == symbol,
                        PaperTrade.status == "open",
                        PaperTrade.strategy == STRATEGY,
                        PaperTrade.stop_loss_price.isnot(None),
                    )
                )
                trades = result.scalars().all()
                if not trades:
                    return

                changed = False
                for trade in trades:
                    reason = self._update_trailing(trade, current_price)
                    if reason:
                        await self._close(session, trade, current_price, reason)
                        changed = True
                    elif trade.trailing_stop_price is not None:
                        changed = True

                if changed:
                    await session.commit()

            except Exception as exc:
                await session.rollback()
                logger.error("[PaperTrade] check_price hatası [%s]: %s", symbol, exc)

    @staticmethod
    def _trail_distance(trade: PaperTrade) -> float:
        if trade.atr and trade.stop_loss_price and trade.entry_price:
            return abs(float(trade.entry_price) - float(trade.stop_loss_price))
        if trade.entry_price:
            return float(trade.entry_price) * 0.005
        return 0.0

    @staticmethod
    def _update_trailing(trade: PaperTrade, price: float) -> Optional[str]:
        sl    = trade.stop_loss_price
        tp    = trade.take_profit_price
        trail = trade.trailing_stop_price
        dist  = PaperTradeManager._trail_distance(trade)

        if trade.signal_type == "Long":
            if trail is None:
                if sl is not None and price <= float(sl):
                    return "stop_loss"
                if tp is not None and price >= float(tp):
                    trade.trailing_stop_price = price - dist
                    return None
            else:
                new_trail = price - dist
                if new_trail > float(trail):
                    trade.trailing_stop_price = new_trail
                if price <= float(trade.trailing_stop_price):
                    return "trailing_stop"
        else:  # Short
            if trail is None:
                if sl is not None and price >= float(sl):
                    return "stop_loss"
                if tp is not None and price <= float(tp):
                    trade.trailing_stop_price = price + dist
                    return None
            else:
                new_trail = price + dist
                if new_trail < float(trail):
                    trade.trailing_stop_price = new_trail
                if price >= float(trade.trailing_stop_price):
                    return "trailing_stop"

        return None

    @staticmethod
    async def _close(
        session: AsyncSession,
        trade: PaperTrade,
        exit_price: float,
        reason: str,
    ) -> None:
        pnl_pct = _calc_pnl(trade.signal_type, float(trade.entry_price), exit_price)
        fee_usd = POSITION_USD * FEE_RATE * 2  # giriş + çıkış
        pnl_usd = (pnl_pct / 100) * POSITION_USD - fee_usd

        trade.status      = "closed"
        trade.closed_at   = datetime.now(timezone.utc)
        trade.exit_price  = exit_price
        trade.close_reason = reason
        trade.pnl_pct     = pnl_pct
        trade.fee_usd     = fee_usd
        trade.pnl_usd     = pnl_usd
        session.add(trade)

        # Portföy güncelle
        portfolio_result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.strategy == STRATEGY)
        )
        portfolio = portfolio_result.scalars().first()
        if portfolio:
            portfolio.balance       += pnl_usd
            portfolio.total_pnl_usd += pnl_usd
            portfolio.total_trades  += 1
            if pnl_usd > 0:
                portfolio.winning_trades += 1
            if portfolio.balance > portfolio.peak_balance:
                portfolio.peak_balance = portfolio.balance
            drawdown = (portfolio.peak_balance - portfolio.balance) / portfolio.peak_balance * 100
            if drawdown > portfolio.max_drawdown_pct:
                portfolio.max_drawdown_pct = drawdown
            portfolio.updated_at = datetime.now(timezone.utc)
            trade.balance_after  = portfolio.balance
            session.add(portfolio)

        logger.info(
            "[PaperTrade] KAPANDI %s %s @ %.6f | %s | PnL: %+.2f$ (%.2f%%) | Bakiye: %.2f$",
            trade.symbol, trade.signal_type, exit_price, reason,
            pnl_usd, pnl_pct, trade.balance_after or 0,
        )

    @staticmethod
    async def _recent_win_rate(session: AsyncSession, symbol: str) -> Optional[float]:
        try:
            result = await session.execute(
                select(PaperTrade.pnl_usd)
                .where(
                    PaperTrade.symbol == symbol,
                    PaperTrade.strategy == STRATEGY,
                    PaperTrade.status == "closed",
                    PaperTrade.pnl_usd.isnot(None),
                )
                .order_by(PaperTrade.closed_at.desc())
                .limit(20)
            )
            pnls = [r for r in result.scalars().all()]
            if not pnls:
                return None
            return round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)
        except Exception:
            return None


paper_trade_manager = PaperTradeManager()
