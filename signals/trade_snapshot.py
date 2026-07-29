"""
trade_snapshot.py — açık paper trade'ler için periyodik (5dk) CVD/VP/VPMV/SMC
"an be an" izleme (19 Tem 2026, kullanıcı isteği: "işlemleri derinlemesine
analiz edecek sistemler").

Giriş-anı özellikleri (entry_features JSONB, bkz. paper_trade_manager.py)
TEK BİR kare — bu modül ise pozisyon AÇIK KALDIĞI SÜRECE düzenli aralıklarla
tekrar örnekleyip signals/market_context.py'nin AYNI hesaplama fonksiyonlarını
kullanarak trade_snapshots (TimescaleDB hypertable) tablosuna yazar.
"""

import asyncio
import logging

import numpy as np

from database.crud import get_cagg_klines
from database.engine import get_session, run_with_db_timeout
from database.models import PaperTrade, TradeSnapshot
from signals.market_context import (
    compute_cvd_slope,
    compute_smc_market_structure,
    compute_vp_score,
)
from signals.vpm_calculator import VPMCalculator
from utils.vpmv import compute_components

logger = logging.getLogger("TradeSnapshot")

_SNAPSHOT_INTERVAL_SECONDS = 60  # 1 dakika (24 Tem 2026: 5dk çok kabaydı —
# ta_kovalama_live işlemleri ort. ~1 saat, çoğu 7-30dk içinde kapanıyor,
# 5dk'lık snapshot çoğu işlem için 0-1 nokta yakalayıp röntgen grafiğini
# anlamsız bırakıyordu. Açık pozisyon sayısı düşük (~7) olduğu için 5x
# sıklık artışı hesaplama yükü açısından sorun değil.
_KLINE_LIMIT = 500
_MIN_BARS = 60


async def _snapshot_one(
    trade_id: int, symbol: str, signal_type: str, interval: str, entry_price: float
) -> None:
    try:
        df = await get_cagg_klines(symbol, interval, _KLINE_LIMIT)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] kline çekilemedi (interval=%s): %s", symbol, interval, exc)
        return
    if df is None or df.empty or len(df) < _MIN_BARS:
        return

    try:
        price = float(df["close"].iloc[-1])
        side = 1.0 if signal_type == "Long" else -1.0
        price_since_entry_pct = (
            round((price - entry_price) / entry_price * 100.0 * side, 4)
            if entry_price
            else None
        )
        cvd = compute_cvd_slope(df)
        vp_buy, vp_sell = compute_vp_score(df)
        vpr_buy, vpr_sell = compute_vp_score(df, use_real_volume=True)
        vp_score_real = None if np.isnan(vpr_buy) else round(vpr_buy - vpr_sell, 2)
        vol_s, mom_s, vlt_s, prc_s = compute_components(df, signal_type)
        smc_struct = compute_smc_market_structure(df, signal_type)
        vpmv_combined = VPMCalculator.calculate(
            vol_score=vol_s, momentum_score=mom_s, vlt_score=vlt_s, price_score=prc_s
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] snapshot hesaplanamadı: %s", symbol, exc)
        return

    try:
        async with get_session() as session:
            session.add(
                TradeSnapshot(
                    trade_id=trade_id,
                    symbol=symbol,
                    price=price,
                    cvd_slope=cvd,
                    vp_buy=vp_buy,
                    vp_sell=vp_sell,
                    vp_score=round(vp_buy - vp_sell, 2) if vp_buy is not None else None,
                    vp_score_real=vp_score_real,
                    vol_score=round(vol_s, 2) if vol_s is not None else None,
                    mom_score=round(mom_s, 2) if mom_s is not None else None,
                    vlt_score=round(vlt_s, 2) if vlt_s is not None else None,
                    price_score=round(prc_s, 2) if prc_s is not None else None,
                    price_since_entry_pct=price_since_entry_pct,
                    vpmv_combined=round(vpmv_combined, 2) if vpmv_combined is not None else None,
                    smc_market_structure=smc_struct,
                )
            )
            await run_with_db_timeout(session.commit())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("[%s] snapshot yazılamadı: %s", symbol, exc)


async def _snapshot_all_open_trades() -> None:
    from sqlalchemy import select  # pylint: disable=import-outside-toplevel

    async with get_session() as session:
        rows = (
            await run_with_db_timeout(
                session.execute(
                    select(
                        PaperTrade.id,
                        PaperTrade.symbol,
                        PaperTrade.signal_type,
                        PaperTrade.interval,
                        PaperTrade.entry_price,
                    ).where(PaperTrade.status == "open")
                )
            )
        ).all()

    for trade_id, symbol, signal_type, interval, entry_price in rows:
        await _snapshot_one(trade_id, symbol, signal_type, interval, entry_price)


async def trade_snapshot_loop() -> None:
    """Periyodik olarak TÜM açık paper trade'ler için market_context
    anlık görüntüsü alır."""
    while True:
        await asyncio.sleep(_SNAPSHOT_INTERVAL_SECONDS)
        try:
            await _snapshot_all_open_trades()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Döngü hatası: %s", exc, exc_info=True)
