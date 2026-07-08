"""
PaperTradeManager — simüle edilmiş pozisyon yönetimi.

Her strateji için ayrı instance kullanılır:
  conf_100  — konfluans sinyalleri ($100/pozisyon)
  ha_cross  — tüm HA_Cross sinyalleri (5m ve 15m)
  rsi_15m   — 15m RSI_Cross sinyalleri
"""

import asyncio
import json as _json
import logging
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import Config
from database.engine import get_session, run_with_db_timeout
from database.models import PaperTrade, PaperPortfolio, Signal
from signals.risk_policy import default_policy
from signals.signal_lifecycle_manager import _calc_pnl
from signals.trailing import update_trailing
from utils.redis_client import RedisClient, SAFE_EXTERNAL_TIMEOUT

logger = logging.getLogger(__name__)

POSITION_USD = 100.0
FEE_RATE     = 0.0005
MAX_OPEN     = 10

_STRATEGY_TRIGGERS: dict[str, Callable[[dict], bool]] = {
    "conf_100": lambda sd: True,
    "ha_cross": lambda sd: sd.get("indicators") == "HA_Cross",
    "rsi_15m":  lambda sd: "RSI_Cross" in (sd.get("indicators") or "") and sd.get("interval") == "15m",
}


class PaperTradeManager:

    def __init__(self, strategy: str = "conf_100") -> None:
        self.strategy = strategy
        self._open_symbols: set[str] = set()

    async def load_open_symbols(self) -> None:
        async def _do_load() -> set[str]:
            async with get_session() as session:
                result = await session.execute(
                    select(PaperTrade.symbol).where(
                        PaperTrade.strategy == self.strategy,
                        PaperTrade.status == "open",
                    )
                )
                return {row[0] for row in result.all()}

        try:
            self._open_symbols = await run_with_db_timeout(_do_load())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] açık sembol yükleme zaman aşımı: %s", self.strategy, exc)
            self._open_symbols = set()
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
        if self.strategy not in Config.PAPER.get("ENABLED_STRATEGIES", []):
            return
        trigger = _STRATEGY_TRIGGERS.get(self.strategy, lambda _: False)
        if not trigger(signal_data) or signal_id is None:
            return

        symbol = signal_data.get("symbol", "")

        async def _do_open() -> None:
            async with get_session() as session:
                try:
                    existing = await session.execute(
                        select(PaperTrade.id).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.symbol == symbol,
                            PaperTrade.status == "open",
                        )
                    )
                    if existing.scalars().first() is not None:
                        logger.info(
                            "[PaperTrade][%s] %s için zaten açık pozisyon var, atlandı",
                            self.strategy, symbol,
                        )
                        return

                    open_count_result = await session.execute(
                        select(func.count()).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.status == "open",
                        )
                    )
                    if open_count_result.scalar() >= MAX_OPEN:
                        logger.debug(
                            "[PaperTrade][%s] MAX_OPEN=%d dolu, atlandı (%s)",
                            self.strategy, MAX_OPEN, symbol,
                        )
                        return

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
                        raw = await asyncio.wait_for(
                            RedisClient.get_client().get("ranking:snapshot"), timeout=SAFE_EXTERNAL_TIMEOUT
                        )
                        if raw:
                            snap = _json.loads(raw)
                            rank_map = {item["symbol"]: item["rank"] for item in snap}
                            rank_at_entry = rank_map.get(symbol)
                    except Exception as exc:
                        logger.debug("[PaperTrade] ranking snapshot okunamadı: %s", exc)

                    recent_win_rate = await self._recent_win_rate(
                        session, symbol, self.strategy
                    )

                    opened_at = signal_data.get("opened_at") or datetime.now()
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at)
                    if isinstance(opened_at, datetime) and opened_at.tzinfo is not None:
                        opened_at = opened_at.replace(tzinfo=None)

                    trade = PaperTrade(
                        signal_id=signal_id,
                        strategy=self.strategy,
                        symbol=symbol,
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
                        vpmv_pre_avg=signal_data.get("vpmv_pre_avg"),
                        vpmv_slope=signal_data.get("vpmv_slope"),
                        vpmv_ratio=signal_data.get("vpmv_ratio"),
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

        try:
            await run_with_db_timeout(_do_open())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] Açma zaman aşımı (%s): %s", self.strategy, symbol, exc)

    async def open_direct(
        self,
        symbol: str,
        signal_type: str,
        interval: str,
        price: float,
        atr: float,
        sl_price: float,
        tp_price: float,
        note: str = "",
    ) -> bool:
        """Sinyal tablosundan bağımsız pozisyon açar (dedektör tabanlı stratejiler)."""
        if self.strategy not in Config.PAPER.get("ENABLED_STRATEGIES", []):
            return False

        async def _do_open_direct() -> bool:
            async with get_session() as session:
                try:
                    existing = await session.execute(
                        select(PaperTrade.id).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.symbol == symbol,
                            PaperTrade.status == "open",
                        )
                    )
                    if existing.scalars().first() is not None:
                        return False

                    open_count = await session.execute(
                        select(func.count()).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.status == "open",
                        )
                    )
                    if open_count.scalar() >= MAX_OPEN:
                        logger.debug("[PaperTrade][%s] MAX_OPEN dolu, %s atlandı", self.strategy, symbol)
                        return False

                    trade = PaperTrade(
                        signal_id=None,
                        strategy=self.strategy,
                        source=note or None,
                        symbol=symbol,
                        signal_type=signal_type,
                        interval=interval,
                        position_usd=POSITION_USD,
                        entry_price=price,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        status="open",
                        opened_at=datetime.now(),
                        atr=atr,
                    )
                    session.add(trade)
                    await session.commit()
                    self._open_symbols.add(symbol)
                    logger.info(
                        "[PaperTrade][%s] ★ AÇILDI %s %s %s @ %.6f | SL=%.6f TP=%.6f %s",
                        self.strategy, symbol, signal_type, interval, price,
                        sl_price, tp_price, note,
                    )
                    return True
                except Exception as exc:
                    await session.rollback()
                    logger.error("[PaperTrade][%s] open_direct hatası: %s", self.strategy, exc, exc_info=True)
                    return False

        try:
            return await run_with_db_timeout(_do_open_direct())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] open_direct zaman aşımı (%s): %s", self.strategy, symbol, exc)
            return False

    async def check_all_prices(self, prices: dict[str, float]) -> None:
        if not self._open_symbols:
            return
        symbols_to_check = [s for s in self._open_symbols if s in prices]
        if not symbols_to_check:
            return

        async def _do_check() -> None:
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

        try:
            await run_with_db_timeout(_do_check())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] batch check zaman aşımı: %s", self.strategy, exc)

    @staticmethod
    def _trail_distance(trade: PaperTrade) -> float:
        if trade.atr and trade.stop_loss_price and trade.entry_price:
            return abs(float(trade.entry_price) - float(trade.stop_loss_price))
        if trade.entry_price:
            return float(trade.entry_price) * 0.005
        return 0.0

    @staticmethod
    def _update_trailing(trade: PaperTrade, price: float) -> Optional[str]:
        return update_trailing(trade, price, PaperTradeManager._trail_distance(trade))

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
        trade.closed_at    = datetime.now()
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
            portfolio.updated_at = datetime.now()
            trade.balance_after  = portfolio.balance

        logger.info(
            "[PaperTrade] KAPANDI %s %s @ %.6f | %s | PnL: %+.2f$ (%.2f%%) | Bakiye: %.2f$",
            trade.symbol, trade.signal_type, exit_price, reason,
            pnl_usd, pnl_pct, trade.balance_after or 0,
        )

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
manual_manager      = PaperTradeManager("manual")
do_kirilimi_manager = PaperTradeManager("do_kirilimi")
