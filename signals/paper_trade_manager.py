"""
PaperTradeManager — simüle edilmiş pozisyon yönetimi.

Her strateji için ayrı instance kullanılır:
  conf_100  — konfluans sinyalleri ($100/pozisyon)
  ha_cross  — tüm HA_Cross sinyalleri (5m ve 15m)
  rsi_15m   — 15m RSI_Cross sinyalleri
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import Config
from database.engine import get_session, run_with_db_timeout
from database.models import PaperPortfolio, PaperTrade, Signal
from database.signal_repository import build_paper_trade
from indicators.core import calculate_evol
from signals.risk_policy import default_policy
from signals.signal_lifecycle_manager import _calc_pnl
from signals.trailing import update_trailing
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient

logger = logging.getLogger(__name__)

POSITION_USD = 100.0
FEE_RATE = 0.0005
MAX_OPEN = 10

# Strateji-özel kaldıraç çarpanı (PnL'e uygulanır) — tanımlı olmayan stratejiler 1.0 (kaldıraçsız).
LEVERAGE_BY_STRATEGY: dict[str, float] = {"rsi_cross_live": 3.0, "ta_kovalama_live": 3.0}

# EVOL-disiplinli erken çıkış (HA_Cross Long'a özel — [[project_devisso_ersi]],
# ha_cross_evol_exit_sweep_bt.py'de eşik=25/min_hold=8 "güvenilir bölge" olarak
# doğrulandı: PF 1.585→1.821, split-period'ın iki yarısında da tutarlı).
EVOL_EXIT_THRESHOLD = 25.0
EVOL_MIN_HOLD_BARS = 8
_INTERVAL_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

_STRATEGY_TRIGGERS: dict[str, Callable[[dict], bool]] = {
    "conf_100": lambda sd: True,
    "ha_cross": lambda sd: sd.get("indicators") == "HA_Cross",
    "rsi_15m": lambda sd: "RSI_Cross" in (sd.get("indicators") or "")
    and sd.get("interval") == "15m",
    # all_up+gövde canlı formülle doğrulandı (14 Tem, memory: rsi_ha_cross_live_readiness).
    # Short filtresiz bırakılmıştı ("baseline zaten sağlam" varsayımı) ama canlı veri
    # (258 işlem, %8.8 win rate, -$115.50) bunu çürüttü — Long'daki filtre Short'a da
    # uygulandı (14 Tem gece, aynı gün içinde revize).
    "rsi_cross_live": lambda sd: sd.get("indicators") == "RSI_Cross(9,24)"
    and sd.get("all_up")
    and sd.get("candle_kategori") == "govde",
    # Gating (HA hizalanma + kohort sıralaması) signals/tf_alignment_gate.py'de
    # on_new_signal çağrılmadan ÖNCE bitiyor — burada ek şart yok.
    "tf_alignment_live": lambda sd: True,
    # Gating (all_up + TA-percentile/kovalama + HA) signals/ta_kovalama_gate.py'de
    # on_new_signal çağrılmadan ÖNCE bitiyor — burada ek şart yok (24 Tem 2026,
    # [[project_rsi_cross_ta_triple_combo_24tem]]).
    "ta_kovalama_live": lambda sd: True,
}


class PaperTradeManager:

    def __init__(
        self,
        strategy: str = "conf_100",
        max_hold_hours: Optional[float] = None,
        max_open: Optional[int] = None,
        position_usd: Optional[float] = None,
        allow_scale_in: bool = False,
    ) -> None:
        self.strategy = strategy
        self._open_symbols: set[str] = set()
        # False (varsayılan, eski davranış): sembolde HERHANGİ bir açık pozisyon
        # varsa yeni sinyal reddedilir. True: SADECE ters yönde açık pozisyon
        # varsa reddedilir — aynı yönde ikinci bir bacak (kendi SL/TP'siyle)
        # açılabilir (24 Tem 2026, [[project_combo_scale_in]] — küçük ama gerçek
        # kazanç, +$77/116 işlem, risk artışı ihmal edilebilir).
        self.allow_scale_in = allow_scale_in
        # None: eski davranış (fiyat kapanışı yok, sadece ters sinyal/manuel/paper
        # SL-TP-trailing). Verilirse: bu süreyi aşan açık pozisyonlar check_all_prices'ta
        # "timeout" ile kapatılır (do_open_streak — SL=3xATR TEK BAŞINA, TP/breakeven yok,
        # backtestte kazananın büyük hareketi 24h'e kadar sınırsız bırakması gerektiği
        # bulundu — bkz. research/pattern_lab/do_break_gauss_sltp_bt.py).
        self.max_hold_hours = max_hold_hours
        # Strateji-özel kapasite/boyut — verilmezse global MAX_OPEN/POSITION_USD kullanılır.
        self.max_open = max_open if max_open is not None else MAX_OPEN
        self.position_usd = position_usd if position_usd is not None else POSITION_USD

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
        logger.info(
            "[PaperTrade][%s] %d açık sembol yüklendi", self.strategy, len(self._open_symbols)
        )

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
                    if self.allow_scale_in:
                        existing = await session.execute(
                            select(PaperTrade.signal_type).where(
                                PaperTrade.strategy == self.strategy,
                                PaperTrade.symbol == symbol,
                                PaperTrade.status == "open",
                            )
                        )
                        existing_dirs = set(existing.scalars().all())
                        new_dir = signal_data.get("signal_type", "")
                        if existing_dirs and existing_dirs - {new_dir}:
                            logger.info(
                                "[PaperTrade][%s] %s için TERS yönde açık pozisyon var, atlandı",
                                self.strategy,
                                symbol,
                            )
                            return
                        # aynı yönde açıksa (veya hiç açık yoksa) devam — scale-in (2. bacak)
                    else:
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
                                self.strategy,
                                symbol,
                            )
                            return

                    open_count_result = await session.execute(
                        select(func.count()).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.status == "open",
                        )
                    )
                    if open_count_result.scalar() >= self.max_open:
                        logger.debug(
                            "[PaperTrade][%s] MAX_OPEN=%d dolu, atlandı (%s)",
                            self.strategy,
                            self.max_open,
                            symbol,
                        )
                        return

                    sig_result = await session.execute(select(Signal).where(Signal.id == signal_id))
                    sig = sig_result.scalars().first()
                    atr_val = float(sig.atr) if sig and sig.atr else 0.0
                    if atr_val > 0:
                        levels = default_policy.calculate_levels(
                            signal_data.get("signal_type", ""),
                            current_price,
                            atr_val,
                            {
                                "vpms_score": signal_data.get("vpms_score"),
                                "mtf_score": signal_data.get("mtf_score"),
                                "interval": signal_data.get("interval", ""),
                            },
                        )
                        sl_price = levels.sl_price
                        tp_price = levels.tp_price
                    else:
                        sl_price = None
                        tp_price = None

                    # 2 Ağu 2026 (Fable 5 performans denetimi): ranking:snapshot
                    # artık SADECE signal_processor.py'de (30sn cache'li) okunup
                    # enriched_signal'e (=signal_data) gömülüyor — burada ayrıca
                    # okunmuyor. rank_score/vs_btc/rank_* ailesi build_paper_trade
                    # içinde doğrudan signal_data'dan çekiliyor, rank_at_entry
                    # (pozisyon) tek istisna: DB'ye özel kolon ismi farklı olduğu
                    # için burada ayrı bir parametre olarak kalıyor.
                    rank_at_entry = signal_data.get("rank_at_entry")

                    recent_win_rate = await self._recent_win_rate(session, symbol, self.strategy)

                    opened_at = signal_data.get("opened_at") or datetime.now()
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at)
                    if isinstance(opened_at, datetime) and opened_at.tzinfo is not None:
                        opened_at = opened_at.replace(tzinfo=None)

                    # 2 Ağu 2026 (Fable 5 mimari denetimi, Kademe 2): PaperTrade
                    # nesnesinin KURULUŞU artık database/signal_repository.py'de
                    # (build_paper_trade) — build_signal ile aynı desen. Davranış
                    # birebir aynı, sadece bu alan-eşleme bloğunun adresi değişti.
                    trade = build_paper_trade(
                        signal_data,
                        signal_id=signal_id,
                        strategy=self.strategy,
                        position_usd=self.position_usd,
                        entry_price=current_price,
                        opened_at=opened_at,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        btc_z_score=btc_z_score,
                        btc_trend=btc_trend,
                        funding_rate=funding_rate,
                        recent_win_rate=recent_win_rate,
                        regime_trend=regime_trend,
                        volatility_regime=volatility_regime,
                        rank_at_entry=rank_at_entry,
                        devisso_score=sig.devisso_score if sig else signal_data.get("devisso_score"),
                        devisso_delta=sig.devisso_delta if sig else None,
                        devisso_ratio=sig.devisso_ratio if sig else None,
                        sl_multiplier=sig.sl_multiplier if sig else None,
                        tp_multiplier=sig.tp_multiplier if sig else None,
                    )
                    session.add(trade)
                    await session.commit()
                    self._open_symbols.add(trade.symbol)

                    logger.info(
                        "[PaperTrade][%s] ★ AÇILDI %s %s %s @ %.6f | VPMV=%.1f Z=%+.2f",
                        self.strategy,
                        trade.symbol,
                        trade.signal_type,
                        trade.interval,
                        current_price,
                        trade.vpms_score or 0,
                        trade.z_score_entry or 0,
                    )

                except Exception as exc:
                    await session.rollback()
                    logger.error(
                        "[PaperTrade][%s] Açma hatası: %s", self.strategy, exc, exc_info=True
                    )

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
        tp_price: Optional[float],
        note: str = "",
        position_usd: Optional[float] = None,
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
                    if open_count.scalar() >= self.max_open:
                        logger.debug(
                            "[PaperTrade][%s] MAX_OPEN dolu, %s atlandı", self.strategy, symbol
                        )
                        return False

                    trade = PaperTrade(
                        signal_id=None,
                        strategy=self.strategy,
                        source=note or None,
                        symbol=symbol,
                        signal_type=signal_type,
                        interval=interval,
                        position_usd=(
                            position_usd if position_usd is not None else self.position_usd
                        ),
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
                    tp_str = f"{tp_price:.6f}" if tp_price is not None else "yok"
                    logger.info(
                        "[PaperTrade][%s] ★ AÇILDI %s %s %s @ %.6f | pozisyon=$%.2f SL=%.6f TP=%s %s",
                        self.strategy,
                        symbol,
                        signal_type,
                        interval,
                        price,
                        trade.position_usd,
                        sl_price,
                        tp_str,
                        note,
                    )
                    return True
                except Exception as exc:
                    await session.rollback()
                    logger.error(
                        "[PaperTrade][%s] open_direct hatası: %s", self.strategy, exc, exc_info=True
                    )
                    return False

        # Kayıp-kritik nokta: bu, do_kirilimi/do_open_streak gibi nadir ateşlenen
        # dedektör-tabanlı sinyallerin YEGANE giriş noktası — burada timeout olup
        # kaybedilen bir sinyal bir sonraki bar'a kadar (ya da hiç) geri gelmez
        # (bkz. 10 Tem 2026 ARXUSDT/POLUSDT vakaları, memory: project_data_layer_debt.md).
        # 1 ek deneme: havuz burst anında birkaç saniye içinde açılıyor, retry çoğu
        # zaman yeterli. _do_open_direct() kendi "existing" kontrolüyle idempotent —
        # ilk deneme aslında commit'i başarıp sadece bize timeout dönmüşse bile
        # retry çift pozisyon AÇMAZ (zaten açık bulur, False döner).
        for attempt in range(2):
            try:
                return await run_with_db_timeout(_do_open_direct())
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if attempt == 0:
                    logger.warning(
                        "[PaperTrade][%s] open_direct zaman aşımı (%s), tekrar deneniyor: %s",
                        self.strategy,
                        symbol,
                        exc,
                    )
                    await asyncio.sleep(0.5)
                else:
                    logger.error(
                        "[PaperTrade][%s] open_direct zaman aşımı (%s), retry de başarısız: %s",
                        self.strategy,
                        symbol,
                        exc,
                    )
                    return False
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

                        if self.max_hold_hours is not None:
                            age_hours = (datetime.now() - trade.opened_at).total_seconds() / 3600
                            if age_hours >= self.max_hold_hours:
                                self._apply_close(trade, price, "timeout", portfolio)
                                closed_symbols.add(trade.symbol)
                                changed = True
                                continue

                        old_trail = trade.trailing_stop_price
                        reason = await self._update_trailing(trade, price)
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
                    logger.error(
                        "[PaperTrade][%s] batch check hatası: %s", self.strategy, exc, exc_info=True
                    )

        try:
            # 10s: periyodik/kritik-olmayan bir kontrol — havuzun kendi pool_timeout'una
            # (30s) daha yakın bir süre, burst anındaki (ör. 500+ sembol aynı anda kapanışı)
            # normal kısa kuyruklanmaya tahammül eder (bkz. run_with_db_timeout docstring).
            await run_with_db_timeout(_do_check(), timeout=10.0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] batch check zaman aşımı: %s", self.strategy, exc)

    async def check_evol_exits(
        self,
        threshold: float = EVOL_EXIT_THRESHOLD,
        min_hold_bars: int = EVOL_MIN_HOLD_BARS,
    ) -> None:
        """
        EVOL (hacim verimliliği) eşiğin altına düşerse SL/TP'yi beklemeden erken
        çıkış — SADECE Long pozisyonlar (short hiç test edilmedi). Pozisyon en az
        `min_hold_bars` bar (kendi interval'ine göre dakikaya çevrilmiş) açık
        kalmadan kontrol edilmez — ham/tek-barlık RVOL denemesi bu şart olmadan
        neredeyse anında tetikleniyordu (artefakt), bkz. [[project_devisso_ersi]].

        check_all_prices'tan bağımsız, daha YAVAŞ bir döngüden (ör. 60s)
        çağrılmalı — EVOL geçmiş OHLCV+hacim gerektirir, anlık fiyattan ucuz
        değil, bu kadar sık değişmiyor.
        """

        async def _do_check() -> None:
            async with get_session() as session:
                try:
                    trades_result = await session.execute(
                        select(PaperTrade).where(
                            PaperTrade.strategy == self.strategy,
                            PaperTrade.status == "open",
                            PaperTrade.signal_type == "Long",
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
                        interval_min = _INTERVAL_MINUTES.get(trade.interval, 15)
                        age_minutes = (datetime.now() - trade.opened_at).total_seconds() / 60
                        if age_minutes < min_hold_bars * interval_min:
                            continue

                        df = await RedisClient.get_mtf_klines(
                            trade.symbol, trade.interval, limit=150
                        )
                        if df is None or len(df) < 30:
                            continue

                        evol = calculate_evol(df)
                        if evol is None or evol >= threshold:
                            continue

                        price = float(df["close"].iloc[-1])
                        self._apply_close(trade, price, "evol_decay", portfolio)
                        closed_symbols.add(trade.symbol)
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
                    logger.error(
                        "[PaperTrade][%s] EVOL-exit check hatası: %s",
                        self.strategy,
                        exc,
                        exc_info=True,
                    )

        try:
            await run_with_db_timeout(_do_check(), timeout=10.0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[PaperTrade][%s] EVOL-exit check zaman aşımı: %s", self.strategy, exc)

    @staticmethod
    def _static_trail_distance(trade: PaperTrade) -> float:
        """Girişteki SABİT ATR'ye dayalı mesafe — dinamik (canlı ATR) okuma
        başarısız olursa (key henüz yok/expire) düşülen fallback."""
        if trade.atr and trade.stop_loss_price and trade.entry_price:
            return abs(float(trade.entry_price) - float(trade.stop_loss_price))
        if trade.entry_price:
            return float(trade.entry_price) * 0.005
        return 0.0

    @staticmethod
    async def _trail_distance(trade: PaperTrade) -> float:
        """Trailing-stop mesafesi — mümkünse GÜNCEL ATR'den (backend'in
        `atr_live:{symbol}:{interval}` yayını, 15 Tem 2026 planı: "trailing
        dinamik ATR"), yoksa girişteki SABİT mesafeden (eski davranış,
        bozulmadı)."""
        try:
            raw = await asyncio.wait_for(
                RedisClient.get_client().get(f"atr_live:{trade.symbol}:{trade.interval}"),
                timeout=SAFE_EXTERNAL_TIMEOUT,
            )
            if raw:
                current_atr = float(raw)
                if current_atr > 0:
                    return current_atr * Config.RISK_SL_MULTIPLIER
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "[PaperTrade] %s canlı ATR okunamadı, sabit mesafeye düşülüyor: %s",
                trade.symbol,
                exc,
            )
        return PaperTradeManager._static_trail_distance(trade)

    @staticmethod
    async def _update_trailing(trade: PaperTrade, price: float) -> Optional[str]:
        dist = await PaperTradeManager._trail_distance(trade)
        return update_trailing(trade, price, dist)

    @staticmethod
    def _apply_close(
        trade: PaperTrade,
        exit_price: float,
        reason: str,
        portfolio: Optional["PaperPortfolio"],
    ) -> None:
        pnl_pct = _calc_pnl(trade.signal_type, float(trade.entry_price), exit_price)
        position_usd = float(trade.position_usd) if trade.position_usd else POSITION_USD
        leverage = LEVERAGE_BY_STRATEGY.get(trade.strategy, 1.0)
        fee_usd = position_usd * FEE_RATE * 2
        pnl_usd = (pnl_pct / 100) * position_usd * leverage - fee_usd

        trade.status = "closed"
        trade.closed_at = datetime.now()
        trade.exit_price = exit_price
        trade.close_reason = reason
        trade.pnl_pct = pnl_pct
        trade.fee_usd = fee_usd
        trade.pnl_usd = pnl_usd

        if portfolio:
            portfolio.balance += pnl_usd
            portfolio.total_pnl_usd += pnl_usd
            portfolio.total_trades += 1
            if pnl_usd > 0:
                portfolio.winning_trades += 1
            if portfolio.balance > portfolio.peak_balance:
                portfolio.peak_balance = portfolio.balance
            drawdown = (portfolio.peak_balance - portfolio.balance) / portfolio.peak_balance * 100
            if drawdown > portfolio.max_drawdown_pct:
                portfolio.max_drawdown_pct = drawdown
            portfolio.updated_at = datetime.now()
            trade.balance_after = portfolio.balance

        logger.info(
            "[PaperTrade] KAPANDI %s %s @ %.6f | %s | PnL: %+.2f$ (%.2f%%) | Bakiye: %.2f$",
            trade.symbol,
            trade.signal_type,
            exit_price,
            reason,
            pnl_usd,
            pnl_pct,
            trade.balance_after or 0,
        )

    @staticmethod
    async def _recent_win_rate(
        session: AsyncSession, symbol: str, strategy: str
    ) -> Optional[float]:
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
ha_cross_manager = PaperTradeManager("ha_cross")
rsi_15m_manager = PaperTradeManager("rsi_15m")
manual_manager = PaperTradeManager("manual")
do_kirilimi_manager = PaperTradeManager("do_kirilimi")
do_open_streak_manager = PaperTradeManager("do_open_streak", max_hold_hours=24.0)
# $2000 bakiye tavanı, işlem başına düşük risk için düşük POSITION_USD + yüksek MAX_OPEN
# (bkz. [[project_rsi_ha_cross_live_readiness_13_14tem]] — filtreli hacim zaten ~8-9 eşzamanlı).
rsi_cross_live_manager = PaperTradeManager("rsi_cross_live", max_open=80, position_usd=25.0)
# TF Hizalanma + Erken Ayrışma (18 Tem 2026, bkz. [[project_tf_alignment_early_divergence]]):
# tetikleme mantığı (1h+4h HA hizalanması + kohort içi erken ayrışma sıralaması)
# signals/tf_alignment_gate.py'de dışarıda yapılıyor — bu yüzden trigger'ı her
# zaman True (gating zaten on_new_signal çağrılmadan ÖNCE bitmiş oluyor).
# Kaldıraçsız başlatıldı (henüz gerçek $'da doğrulanmadı, LEVERAGE_BY_STRATEGY'de
# tanımsız = 1.0 varsayılan).
tf_alignment_live_manager = PaperTradeManager(
    "tf_alignment_live", max_open=80, position_usd=25.0
)
# all_up + TA-percentile/kovalama + HA-hizalanma (24 Tem 2026, [[project_rsi_cross_ta_triple_combo_24tem]]
# — "didik didik" serisinin canlıya alınması). Gating signals/ta_kovalama_gate.py'de
# on_new_signal ÇAĞRILMADAN ÖNCE bitiyor. Scale-in AÇIK (aynı yönde 2. bacağa izin
# verir, [[project_combo_scale_in]]) — kapasite analizi $500 sermayenin bile
# yetiyor göstermişti, max_open=80 rahat bir tavan.
ta_kovalama_live_manager = PaperTradeManager(
    "ta_kovalama_live", max_open=80, position_usd=25.0, allow_scale_in=True
)
