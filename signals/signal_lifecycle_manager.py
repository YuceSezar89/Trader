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
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session, run_with_db_timeout
from database.models import Signal
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient

logger = logging.getLogger(__name__)

TIMEOUT_HOURS: dict[str, int] = {
    "1m": 4,
    "5m": 24,
    "15m": 48,
    "1h": 7 * 24,
    "4h": 21 * 24,
    "1d": 60 * 24,
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
        symbol = signal_data["symbol"]
        interval = signal_data["interval"]
        indicators = signal_data["indicators"]
        sig_type = signal_data["signal_type"]
        open_price = float(signal_data["open_price"])

        async def _do_process() -> Optional[int]:
            async with get_session() as session:
                try:
                    active = await self._get_active(session, symbol, interval, indicators)

                    if active:
                        if active.signal_type == sig_type:
                            await self._update_scores(session, active, signal_data)
                            logger.debug(
                                "[%s] %s %s skor güncellendi (%s)",
                                symbol,
                                interval,
                                sig_type,
                                indicators,
                            )
                            return None

                        close_px = current_price or open_price
                        await self._close(session, active, close_px, "reversal")
                        logger.info(
                            "[%s] %s %s(%s) kapatıldı → %s açılıyor",
                            symbol,
                            interval,
                            active.signal_type,
                            indicators,
                            sig_type,
                        )

                    # 27 Tem 2026: sinyaller henüz aday aşamasında — SL/TP
                    # hesabı gerçek pozisyon açılana kadar (paper trade
                    # tarafı) gereksiz, kaldırıldı (kullanıcı kararı).
                    sl_price = tp_price = sl_mult = tp_mult = None

                    new_deviso = signal_data.get("devisso_score")
                    prev_deviso = await self._get_prev_devisso(
                        session, symbol, interval, indicators, sig_type
                    )
                    devisso_delta = (
                        round(new_deviso - prev_deviso, 2)
                        if new_deviso is not None and prev_deviso is not None
                        else None
                    )
                    devisso_ratio = (
                        round(new_deviso / prev_deviso, 3)
                        if new_deviso is not None and prev_deviso is not None and prev_deviso != 0
                        else None
                    )

                    # 2 Ağu 2026 (Fable 5 performans denetimi): bu değerler artık
                    # burada ayrıca Redis'ten okunmuyor — signal_processor.py
                    # ranking:snapshot'ı ZATEN 30sn-cache'li tek bir okumadan
                    # çıkarıp enriched_signal'e (=signal_data) gömüyor, dakikada
                    # 250+ gereksiz round-trip'in bir kısmı buradan geliyordu.
                    rank_score_val = signal_data.get("rank_score")
                    vs_btc_val = signal_data.get("vs_btc")
                    rank_combined_val = signal_data.get("rank_combined")
                    rank_rsi_cross_val = signal_data.get("rank_rsi_cross")
                    rank_z_confluence_val = signal_data.get("rank_z_confluence")
                    rank_r_score_val = signal_data.get("rank_r_score")
                    rank_aligned_val = signal_data.get("rank_aligned")
                    rank_alignment_count_val = signal_data.get("rank_alignment_count")

                    new_sig = Signal(
                        symbol=symbol,
                        interval=interval,
                        indicators=indicators,
                        signal_type=sig_type,
                        opened_at=signal_data.get("opened_at", datetime.now()),
                        open_price=open_price,
                        status="active",
                        vpms_score=signal_data.get("vpms_score"),
                        mtf_score=signal_data.get("mtf_score"),
                        st_confirmed=signal_data.get("st_confirmed"),
                        rsi=signal_data.get("rsi"),
                        strength=signal_data.get("strength"),
                        atr=signal_data.get("atr"),
                        alpha=signal_data.get("alpha"),
                        beta=signal_data.get("beta"),
                        sharpe_ratio=signal_data.get("sharpe_ratio"),
                        sortino_ratio=signal_data.get("sortino_ratio"),
                        calmar_ratio=signal_data.get("calmar_ratio"),
                        information_ratio=signal_data.get("information_ratio"),
                        oi_data=signal_data.get("oi_data"),
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        sl_multiplier=sl_mult,
                        tp_multiplier=tp_mult,
                        z_score_entry=signal_data.get("z_score_entry"),
                        is_confluence=signal_data.get("is_confluence", False),
                        ha_ultra_confirm=signal_data.get("ha_ultra_confirm"),
                        vpmv_pre_avg=signal_data.get("vpmv_pre_avg"),
                        vpmv_pre_proxy=signal_data.get("vpmv_pre_proxy"),
                        vpmv_pre_total=signal_data.get("vpmv_pre_total"),
                        vpmv_slope=signal_data.get("vpmv_slope"),
                        vpmv_ratio=signal_data.get("vpmv_ratio"),
                        cvd_slope=signal_data.get("cvd_slope"),
                        vp_buy_avg=signal_data.get("vp_buy_avg"),
                        vp_sell_avg=signal_data.get("vp_sell_avg"),
                        vp_score=signal_data.get("vp_score"),
                        vp_score_real=signal_data.get("vp_score_real"),
                        devisso_score=new_deviso,
                        devisso_delta=devisso_delta,
                        devisso_ratio=devisso_ratio,
                        pd_zone=signal_data.get("pd_zone"),
                        market_structure=signal_data.get("market_structure"),
                        fvg_tfs=signal_data.get("fvg_tfs"),
                        candle_pattern=signal_data.get("candle_pattern"),
                        rank_score=rank_score_val,
                        vs_btc=vs_btc_val,
                        rank_combined=rank_combined_val,
                        rank_rsi_cross=rank_rsi_cross_val,
                        rank_z_confluence=rank_z_confluence_val,
                        rank_r_score=rank_r_score_val,
                        rank_aligned=rank_aligned_val,
                        rank_alignment_count=rank_alignment_count_val,
                        vol_score=signal_data.get("vol_score"),
                        mom_score=signal_data.get("mom_score"),
                        volat_score=signal_data.get("volat_score"),
                        price_score=signal_data.get("price_score"),
                        candle_kategori=signal_data.get("candle_kategori"),
                        all_up=signal_data.get("all_up"),
                        regime_trend=signal_data.get("regime_trend"),
                        volatility_regime=signal_data.get("volatility_regime"),
                        btc_z_score=signal_data.get("btc_z_score"),
                        btc_trend=signal_data.get("btc_trend"),
                        funding_rate=signal_data.get("funding_rate"),
                        hour_utc=signal_data.get("hour_utc"),
                        day_of_week=signal_data.get("day_of_week"),
                    )
                    session.add(new_sig)
                    await session.flush()

                    if active:
                        active.closed_by = new_sig.id

                    await session.commit()
                    logger.info(
                        "[%s] %s %s sinyal açıldı (id=%s, %s)",
                        symbol,
                        interval,
                        sig_type,
                        new_sig.id,
                        indicators,
                    )
                    try:
                        await asyncio.wait_for(
                            RedisClient.get_client().publish(
                                "signal_opened", f"{symbol}:{interval}"
                            ),
                            timeout=SAFE_EXTERNAL_TIMEOUT,
                        )
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        logger.debug("signal_opened publish hatası: %s", exc)
                    return new_sig.id

                except Exception as exc:
                    await session.rollback()
                    logger.error("[%s] sinyal işleme hatası: %s", symbol, exc, exc_info=True)
                    raise

        async with self._get_lock(symbol, interval):
            return await run_with_db_timeout(_do_process())

    async def sweep_timeouts(self) -> int:
        """Timeout eşiğini geçmiş aktif sinyalleri kapatır."""
        closed_holder = {"n": 0}

        async def _do_sweep() -> None:
            async with get_session() as session:
                try:
                    result = await session.execute(select(Signal).where(Signal.status == "active"))
                    actives = result.scalars().all()

                    now = datetime.now()
                    for sig in actives:
                        hours = TIMEOUT_HOURS.get(sig.interval, _DEFAULT_TIMEOUT)
                        opened = sig.opened_at
                        if isinstance(opened, datetime) and opened.tzinfo is not None:
                            opened = opened.replace(tzinfo=None)
                        if now - opened > timedelta(hours=hours):
                            close_price = await self._current_price(sig.symbol, float(sig.open_price))
                            await self._close(session, sig, close_price, "timeout")
                            closed_holder["n"] += 1

                    if closed_holder["n"]:
                        await session.commit()
                        logger.info("Timeout sweep: %d sinyal kapatıldı", closed_holder["n"])

                except Exception as exc:
                    await session.rollback()
                    logger.error("Sweep hatası: %s", exc, exc_info=True)

        try:
            await run_with_db_timeout(_do_sweep())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Sweep hatası (dış): %s", exc)
        return closed_holder["n"]

    async def close_stale(
        self, signal_id: int, close_price: float, reason: str = "reconciliation"
    ) -> bool:
        """Startup reconciliation veya harici tetikleyiciler için sinyal kapatır."""

        async def _do_close() -> bool:
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

        try:
            return await run_with_db_timeout(_do_close())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Stale kapatma zaman aşımı (id=%s): %s", signal_id, exc)
            return False

    async def manual_close(self, signal_id: int, close_price: float) -> bool:
        async def _do_manual_close() -> bool:
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

        try:
            return await run_with_db_timeout(_do_manual_close())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Manuel kapatma zaman aşımı (id=%s): %s", signal_id, exc)
            return False

    async def _get_prev_devisso(
        self, session: AsyncSession, symbol: str, interval: str, indicators: str, signal_type: str
    ) -> Optional[float]:
        result = await session.execute(
            select(Signal.devisso_score)
            .where(
                Signal.symbol == symbol,
                Signal.interval == interval,
                Signal.indicators == indicators,
                Signal.signal_type == signal_type,
                Signal.devisso_score.isnot(None),
            )
            .order_by(Signal.id.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return float(row) if row is not None else None

    async def _get_active(
        self, session: AsyncSession, symbol: str, interval: str, indicators: str
    ) -> Optional[Signal]:
        """(symbol, interval, indicators) üçlüsü kendi bağımsız pozisyonuna sahip —
        farklı indicator'lar (HA_Cross/RSI_Cross/...) aynı sembol+interval'de
        eşzamanlı sinyal üretse bile birbirinin sinyalini kapatmaz/güncellemez
        (12 Tem 2026: eski davranışta reversal'ların %40'ı çapraz-indicator
        kirlenmesiydi)."""
        result = await session.execute(
            select(Signal)
            .where(
                Signal.symbol == symbol,
                Signal.interval == interval,
                Signal.indicators == indicators,
                Signal.status == "active",
            )
            .order_by(Signal.id.desc())
            .with_for_update()
        )
        actives = result.scalars().all()
        if not actives:
            return None
        if len(actives) > 1:
            for stale in actives[1:]:
                await self._close(session, stale, float(stale.open_price), "reconciliation")
            logger.warning(
                "[%s] %s (%s): %d duplikat aktif sinyal temizlendi",
                symbol,
                interval,
                indicators,
                len(actives) - 1,
            )
        return actives[0]

    @staticmethod
    async def _current_price(symbol: str, fallback: float) -> float:
        """Canlı fiyatı `ticker:{symbol}` Redis key'inden okur — timeout kapanışı
        20 Tem 2026'ya kadar sig.open_price'ı kullanıyordu, bu da HER timeout
        kapanışını sahte %0.000 PnL ile kaydediyordu (gerçek fiyat hareketi
        hiç ölçülmüyordu). Redis okunamazsa fallback (open_price) kullanılır —
        bu durumda da PnL 0 çıkar ama en azından istisnai/loglu bir durum olur."""
        try:
            raw = await asyncio.wait_for(
                RedisClient.get_client().get(f"ticker:{symbol}"),
                timeout=SAFE_EXTERNAL_TIMEOUT,
            )
            if raw:
                price = float(json.loads(raw).get("price", 0) or 0)
                if price > 0:
                    return price
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[%s] timeout kapanışı için canlı fiyat okunamadı: %s", symbol, exc)
        logger.warning("[%s] canlı fiyat bulunamadı, open_price'a düşülüyor (PnL 0 çıkacak)", symbol)
        return fallback

    async def _close(
        self,
        session: AsyncSession,
        sig: Signal,
        close_price: float,
        reason: str,
    ) -> None:
        sig.status = "closed"
        sig.closed_at = datetime.now()
        sig.close_price = close_price
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
