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
from database.models import PaperTrade
from database.signal_repository import build_trade_snapshot
from signals.market_context import (
    calculate_metrics_via_pool,
    classify_candle,
    compute_cvd_slope,
    compute_signal_extras,
    compute_smc_market_structure,
    compute_vp_score,
    get_ref_df_cached,
    prepare_for_metrics,
)
from signals.vpm_calculator import VPMCalculator
from utils.vpmv import compute_components, compute_raw_components

logger = logging.getLogger("TradeSnapshot")

_SNAPSHOT_INTERVAL_SECONDS = 60  # 1 dakika (24 Tem 2026: 5dk çok kabaydı —
# ta_kovalama_live işlemleri ort. ~1 saat, çoğu 7-30dk içinde kapanıyor,
# 5dk'lık snapshot çoğu işlem için 0-1 nokta yakalayıp röntgen grafiğini
# anlamsız bırakıyordu. Açık pozisyon sayısı düşük (~7) olduğu için 5x
# sıklık artışı hesaplama yükü açısından sorun değil.
_KLINE_LIMIT = 500
_MIN_BARS = 60


def _entry_change_pct(current: "float | None", entry: "float | None") -> "float | None":
    """signal_processor.py::_compute_component_change_pct ile AYNI formül —
    kıyas noktası 'bir önceki sinyal' değil 'kendi girişimiz'."""
    if current is None or entry is None or abs(entry) < 1e-8:
        return None
    return (current - entry) / abs(entry) * 100.0


async def _snapshot_one(  # pylint: disable=too-many-locals
    trade_id: int,
    symbol: str,
    signal_type: str,
    interval: str,
    entry_price: float,
    entry_vol_raw: "float | None" = None,
    entry_mom_raw: "float | None" = None,
    entry_volat_raw: "float | None" = None,
    entry_price_raw: "float | None" = None,
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
            round((price - entry_price) / entry_price * 100.0 * side, 4) if entry_price else None
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
        _, body_pct, wick_pct = classify_candle(df.iloc[-1])  # kategori değil, sadece oranlar
        extras = compute_signal_extras(df) or {}

        # 20 Ağu 2026 (migration 037): VPMV ham bileşenlerinin GİRİŞE göre
        # %değişimi — sadece giriş anındaki ham değerler kaydedilmişse
        # (totalamount_rank1 gibi Signal'siz açılan stratejiler) hesaplanabilir.
        raw_now = compute_raw_components(df, signal_type)
        vol_change_pct = mom_change_pct = volat_change_pct = price_change_pct = None
        if raw_now is not None:
            vol_change_pct = _entry_change_pct(raw_now.get("vol"), entry_vol_raw)
            mom_change_pct = _entry_change_pct(raw_now.get("mom"), entry_mom_raw)
            volat_change_pct = _entry_change_pct(raw_now.get("vlt"), entry_volat_raw)
            price_change_pct = _entry_change_pct(raw_now.get("prc"), entry_price_raw)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] snapshot hesaplanamadı: %s", symbol, exc)
        return

    # 17 Ağu 2026 (migration 036): alpha/beta + finansal oranlar — BTC referansı
    # gerektiriyor, ayrı bir try bloğunda (başarısız olursa snapshot'ın geri
    # kalanı YİNE de yazılsın, bu aile opsiyonel/best-effort).
    alpha = beta = sharpe = sortino = calmar = treynor = information = None
    try:
        ref_df = await get_ref_df_cached(interval)
        if ref_df is not None and not ref_df.empty and "open_time" in df.columns:
            df_prepared, ref_df_prepared = prepare_for_metrics(df, ref_df)
            df_with_metrics = await calculate_metrics_via_pool(
                df_prepared, ref_df_prepared, interval
            )
            last = df_with_metrics.iloc[-1]
            alpha = last.get("alpha")
            beta = last.get("beta")
            sharpe = last.get("sharpe_ratio")
            sortino = last.get("sortino_ratio")
            calmar = last.get("calmar_ratio")
            treynor = last.get("treynor_ratio")
            information = last.get("information_ratio")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[%s] alpha/beta/finansal oran hesaplanamadı: %s", symbol, exc)

    try:
        # 2 Ağu 2026 (Fable 5 mimari denetimi, Kademe 3): kuruluş build_trade_
        # snapshot()'a taşındı (tutarlılık — bu tek yazma noktasıydı, kopya
        # riski yoktu). Davranış birebir aynı.
        snapshot = build_trade_snapshot(
            trade_id=trade_id,
            symbol=symbol,
            price=price,
            cvd_slope=cvd,
            vp_buy=vp_buy,
            vp_sell=vp_sell,
            vp_score_real=vp_score_real,
            vol_score=vol_s,
            mom_score=mom_s,
            volat_score=vlt_s,
            price_score=prc_s,
            price_since_entry_pct=price_since_entry_pct,
            vpmv_combined=vpmv_combined,
            smc_market_structure=smc_struct,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            treynor_ratio=treynor,
            information_ratio=information,
            signal_rsi_change=extras.get("rsi_change"),
            signal_mfi_change=extras.get("mfi_change"),
            signal_macd_change=extras.get("macd_change"),
            signal_obv=extras.get("obv"),
            body_pct=body_pct,
            wick_pct=wick_pct,
            vol_change_pct=vol_change_pct,
            mom_change_pct=mom_change_pct,
            volat_change_pct=volat_change_pct,
            price_change_pct=price_change_pct,
        )
        async with get_session() as session:
            session.add(snapshot)
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
                        PaperTrade.vol_raw,
                        PaperTrade.mom_raw,
                        PaperTrade.volat_raw,
                        PaperTrade.price_raw,
                    ).where(PaperTrade.status == "open")
                )
            )
        ).all()

    for (
        trade_id,
        symbol,
        signal_type,
        interval,
        entry_price,
        vol_raw,
        mom_raw,
        volat_raw,
        price_raw,
    ) in rows:
        await _snapshot_one(
            trade_id,
            symbol,
            signal_type,
            interval,
            entry_price,
            entry_vol_raw=vol_raw,
            entry_mom_raw=mom_raw,
            entry_volat_raw=volat_raw,
            entry_price_raw=price_raw,
        )


async def trade_snapshot_loop() -> None:
    """Periyodik olarak TÜM açık paper trade'ler için market_context
    anlık görüntüsü alır."""
    while True:
        await asyncio.sleep(_SNAPSHOT_INTERVAL_SECONDS)
        try:
            await _snapshot_all_open_trades()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Döngü hatası: %s", exc, exc_info=True)
