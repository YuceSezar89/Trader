"""
PaperTradeManager — simüle edilmiş pozisyon yönetimi.

Her strateji için ayrı instance kullanılır:
  conf_100  — konfluans sinyalleri ($100/pozisyon)
  ha_cross  — tüm HA_Cross sinyalleri (5m ve 15m)
  rsi_15m   — 15m RSI_Cross sinyalleri
"""

import json as _json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session
from database.models import PaperTrade, PaperPortfolio, Signal
from signals.risk_policy import default_policy
from signals.signal_lifecycle_manager import _calc_pnl
from utils.redis_client import RedisClient

logger = logging.getLogger(__name__)

POSITION_USD = 100.0
FEE_RATE     = 0.0005
MAX_OPEN     = 10

_STRATEGY_TRIGGERS: dict[str, Callable[[dict], bool]] = {
    "conf_100": lambda sd: False,
    "ha_cross": lambda sd: sd.get("indicators") == "HA_Cross",
    "rsi_15m":  lambda sd: "RSI_Cross" in (sd.get("indicators") or "") and sd.get("interval") == "15m",
}


class PaperTradeManager:

    def __init__(self, strategy: str = "conf_100") -> None:
        self.strategy = strategy
        self._open_symbols: set[str] = set()

    async def load_open_symbols(self) -> None:
        async with get_session() as session:
            result = await session.execute(
                select(PaperTrade.symbol).where(
                    PaperTrade.strategy == self.strategy,
                    PaperTrade.status == "open",
                )
            )
            self._open_symbols = {row[0] for row in result.all()}
        logger.info("[PaperTrade][%s] %d açık sembol yüklendi", self.strategy, len(self._open_symbols))

    async def on_new_signal(
        self,
        signal_data: dict,
        signal_id: Optional[int],
        current_price: float,
        btc_z_score: Optional[float] = None,
        btc_trend: Optional[str] = None,
        funding_rate: Optional[float] = None,
        regime_trend: Optional[str] = None,
        volatility_regime: Optional[str] = None,
    ) -> None:
        trigger = _STRATEGY_TRIGGERS.get(self.strategy, lambda _: False)
        if not trigger(signal_data) or signal_id is None:
            return

        async with get_session() as session:
            try:
                sig_result = await session.execute(
                    select(Signal).where(Signal.id == signal_id)
                )
                sig = sig_result.scalars().first()
                atr_val = float(sig.atr) if sig and sig.atr else 0.0
                if atr_val > 0:
                    levels = default_policy.calculate_levels(
                        signal_data.get("signal_type", ""),
                        current_price,
                        atr_val,
                        {
                            "vpms_score": signal_data.get("vpms_score"),
                            "mtf_score":  signal_data.get("mtf_score"),
                            "interval":   signal_data.get("interval", ""),
                        },
                    )
                    sl_price = levels.sl_price
                    tp_price = levels.tp_price
                else:
                    sl_price = None
                    tp_price = None

                rank_at_entry: Optional[int] = None
                try:
                    raw = await RedisClient.get_client().get("ranking:snapshot")
                    if raw:
                        snap = _json.loads(raw)
                        rank_map = {item["symbol"]: item["rank"] for item in snap}
                        rank_at_entry = rank_map.get(signal_data.get("symbol", ""))
                except Exception:
                    pass

                recent_win_rate = await self._recent_win_rate(
                    session, signal_data.get("symbol", ""), self.strategy
                )

                opened_at = signal_data.get("opened_at") or datetime.utcnow()
                if isinstance(opened_at, str):
                    opened_at = datetime.fromisoformat(opened_at)
                if isinstance(opened_at, datetime) and opened_at.tzinfo is not None:
                    opened_at = opened_at.astimezone(timezone.utc).replace(tzinfo=None)

                trade = PaperTrade(
                    signal_id=signal_id,
                    strategy=self.strategy,
                    symbol=signal_data.get("symbol", ""),
                    signal_type=signal_data.get("signal_type", ""),
                    interval=signal_data.get("interval", ""),
                    position_usd=POSITION_USD,
                    entry_price=current_price,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    status="open",
                    opened_at=opened_at,
                    btc_z_score=btc_z_score,
                    btc_trend=btc_trend,
                    hour_utc=opened_at.hour if opened_at else None,
                    day_of_week=opened_at.weekday() if opened_at else None,
                    funding_rate=funding_rate,
                    recent_win_rate=recent_win_rate,
                    vpms_score=signal_data.get("vpms_score"),
                    z_score_entry=signal_data.get("z_score_entry"),
                    mtf_score=signal_data.get("mtf_score"),
                    atr=signal_data.get("atr"),
                    rank_at_entry=rank_at_entry,
                    regime_trend=regime_trend,
                    volatility_regime=volatility_regime,
                )
                session.add(trade)
                await session.commit()
                self._open_symbols.add(trade.symbol)

                logger.info(
                    "[PaperTrade][%s] ★ AÇILDI %s %s %s @ %.6f | VPMV=%.1f Z=%+.2f",
                    self.strategy, trade.symbol, trade.signal_type, trade.interval,
                    current_price,
                    trade.vpms_score or 0, trade.z_score_entry or 0,
                )

            except Exception as exc:
                await session.rollback()
                logger.error("[PaperTrade][%s] Açma hatası: %s", self.strategy, exc, exc_info=True)

    async def check_all_prices(self, prices: dict[str, float]) -> None:
        if not self._open_symbols:
            return
        symbols_to_check = [s for s in self._open_symbols if s in prices]
        if not symbols_to_check:
            return
        async with get_session() as session:
            try:
                trades_result = await session.execute(
                    select(PaperTrade).where(
                        PaperTrade.strategy == self.strategy,
                        PaperTrade.status == "open",
                        PaperTrade.stop_loss_price.isnot(None),
                        PaperTrade.symbol.in_(symbols_to_check),
                    )
                )
                trades = trades_result.scalars().all()
                if not trades:
                    return

                pf_result = await session.execute(
                    select(PaperPortfolio).where(PaperPortfolio.strategy == self.strategy)
                )
                portfolio = pf_result.scalars().first()

                changed = False
                closed_symbols: set[str] = set()
                for trade in trades:
                    price = prices.get(trade.symbol)
                    if price is None:
                        continue
                    old_trail = trade.trailing_stop_price
                    reason = self._update_trailing(trade, price)
                    if reason:
                        self._apply_close(trade, price, reason, portfolio)
                        closed_symbols.add(trade.symbol)
                        changed = True
                    elif trade.trailing_stop_price != old_trail:
                        changed = True

                if changed:
                    if portfolio:
                        session.add(portfolio)
                    await session.commit()

                open_syms = {t.symbol for t in trades if t.status == "open"}
                for sym in closed_symbols:
                    if sym not in open_syms:
                        self._open_symbols.discard(sym)

            except Exception as exc:
                await session.rollback()
                logger.error("[PaperTrade][%s] batch check hatası: %s", self.strategy, exc, exc_info=True)

    async def check_price(self, symbol: str, current_price: float) -> None:
        if symbol not in self._open_symbols:
            return

        async with get_session() as session:
            try:
                result = await session.execute(
                    select(PaperTrade).where(
                        PaperTrade.symbol == symbol,
                        PaperTrade.status == "open",
                        PaperTrade.strategy == self.strategy,
                        PaperTrade.stop_loss_price.isnot(None),
                    )
                )
                trades = result.scalars().all()
                if not trades:
                    self._open_symbols.discard(symbol)
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

                if all(t.status == "closed" for t in trades):
                    self._open_symbols.discard(symbol)

            except Exception as exc:
                await session.rollback()
                logger.error("[PaperTrade][%s] check_price hatası [%s]: %s", self.strategy, symbol, exc)

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
        else:
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
    def _apply_close(
        trade: PaperTrade,
        exit_price: float,
        reason: str,
        portfolio: Optional["PaperPortfolio"],
    ) -> None:
        pnl_pct = _calc_pnl(trade.signal_type, float(trade.entry_price), exit_price)
        fee_usd = POSITION_USD * FEE_RATE * 2
        pnl_usd = (pnl_pct / 100) * POSITION_USD - fee_usd

        trade.status       = "closed"
        trade.closed_at    = datetime.utcnow()
        trade.exit_price   = exit_price
        trade.close_reason = reason
        trade.pnl_pct      = pnl_pct
        trade.fee_usd      = fee_usd
        trade.pnl_usd      = pnl_usd

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
            portfolio.updated_at = datetime.utcnow()
            trade.balance_after  = portfolio.balance

        logger.info(
            "[PaperTrade] KAPANDI %s %s @ %.6f | %s | PnL: %+.2f$ (%.2f%%) | Bakiye: %.2f$",
            trade.symbol, trade.signal_type, exit_price, reason,
            pnl_usd, pnl_pct, trade.balance_after or 0,
        )

    async def _close(
        self,
        session: AsyncSession,
        trade: PaperTrade,
        exit_price: float,
        reason: str,
    ) -> None:
        pnl_pct = _calc_pnl(trade.signal_type, float(trade.entry_price), exit_price)
        fee_usd = POSITION_USD * FEE_RATE * 2
        pnl_usd = (pnl_pct / 100) * POSITION_USD - fee_usd

        trade.status       = "closed"
        trade.closed_at    = datetime.utcnow()
        trade.exit_price   = exit_price
        trade.close_reason = reason
        trade.pnl_pct      = pnl_pct
        trade.fee_usd      = fee_usd
        trade.pnl_usd      = pnl_usd
        session.add(trade)

        portfolio_result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.strategy == self.strategy)
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
            portfolio.updated_at = datetime.utcnow()
            trade.balance_after  = portfolio.balance
            session.add(portfolio)

        logger.info(
            "[PaperTrade] KAPANDI %s %s @ %.6f | %s | PnL: %+.2f$ (%.2f%%) | Bakiye: %.2f$",
            trade.symbol, trade.signal_type, exit_price, reason,
            pnl_usd, pnl_pct, trade.balance_after or 0,
        )

        try:
            from utils.vpmv import compute_post, POST_BARS
            import pandas as _pd
            df = await RedisClient.get_mtf_klines(trade.symbol, trade.interval)
            if df is not None and not df.empty and trade.opened_at is not None:
                raw_times = _pd.to_datetime(df["open_time"]) if "open_time" in df.columns else df.index
                diffs = (raw_times - _pd.Timestamp(trade.opened_at)).abs()
                bar_idx = int(diffs.argmin())
                post_avg, post_delta = compute_post(df, trade.signal_type, bar_idx, POST_BARS)
                if post_avg is not None:
                    trade.vpmv_post_avg   = round(post_avg, 2)
                    trade.vpmv_post_delta = round(post_delta, 2)
                    session.add(trade)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    @staticmethod
    async def _recent_win_rate(session: AsyncSession, symbol: str, strategy: str) -> Optional[float]:
        try:
            result = await session.execute(
                select(PaperTrade.pnl_usd)
                .where(
                    PaperTrade.symbol == symbol,
                    PaperTrade.strategy == strategy,
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


paper_trade_manager = PaperTradeManager("conf_100")
ha_cross_manager    = PaperTradeManager("ha_cross")
rsi_15m_manager     = PaperTradeManager("rsi_15m")
