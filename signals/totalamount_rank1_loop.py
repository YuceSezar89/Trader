"""Totalamount Rank-1 (Devisso Döngüsü) döngüsü — bkz. signals/totalamount_rank1.py.

16 Ağu 2026: live_data_manager.py'den buraya taşındı. Bu döngü GERÇEKTEN paper
trade açıp kapatıyor (kritik iş) ama önceden run_services.py'de 12 tane
best-effort görüntüleme döngüsüyle (ranking/VPMV/ticker/vb.) aynı event
loop'u paylaşıyordu — biri tıkanırsa açma/kapama kararı da gecikebiliyordu.

Artık signal_service.py'de (kritik-iş process'i) çalışıyor. Veri kaynağı da
live_data_manager'ın process-içi belleği (self.mtf_buffers — restart'ta
sıfırlanan, sadece o process'te var olan bir state) yerine doğrudan
Postgres/CA (get_cagg_klines) — hiçbir process'in iç durumuna veya Redis
önbelleğine bağımlı değil, her zaman kaynağın kendisinden okur.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import select as _sel

from config import Config
from database.crud import get_cagg_klines
from database.engine import get_session, run_with_db_timeout
from database.models import PaperPortfolio as _PP
from database.models import PaperTrade as _PT
from database.models import Signal as _Sig
from indicators.core import calculate_atr
from signals.paper_trade_manager import totalamount_rank1_manager
from signals.totalamount_rank1 import net_ta_series
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient
from utils.vpmv import compute_raw_components

logger = logging.getLogger("TotalamountRank1")

_INTERVAL = Config.PAPER["TOTALAMOUNT_RANK1"]["LOOP_INTERVAL_SEC"]
_SL_ATR = Config.PAPER["TOTALAMOUNT_RANK1"]["SL_ATR"]
_INDICATOR = "Supertrend(10,3.0)"
_WARMUP_MIN_COVERAGE = 0.7  # adayların en az %70'i için CA verisi hesaplanabilmeli
_SIGNAL_INTERVAL = "5m"
_LIVE_TTL = 300  # döngü ~90sn'de bir yayınlıyor, 3x pay
_SERIES_LEN = 100
_CA_LIMIT = 150  # net_ta_series + ATR için yeterli tarihsel pencere (~12.5 saat)


async def _publish_totalamount_live(symbol: str, df, net) -> None:
    """(symbol) için Totalamount serisini Redis'e yazar — grafik/panel
    görüntülemesi için, karar mantığı buna bağımlı değil."""
    redis = RedisClient.get_client()
    try:
        net_tail = net[-_SERIES_LEN:]
        ts_tail = (
            (df["open_time"].tail(len(net_tail)) / 1000.0).tolist()
            if "open_time" in df.columns
            else []
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TotalamountLive] %s hesaplanamadı: %s", symbol, exc)
        return

    payload = {
        "value_now": round(float(net_tail[-1]), 4),
        "recent": [round(float(v), 4) for v in net_tail.tolist()],
        "ts_recent": ts_tail,
        "ts": int(time.time()),
    }
    try:
        await asyncio.wait_for(
            redis.set(f"totalamount_live:{symbol}:5m", json.dumps(payload), ex=_LIVE_TTL),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TotalamountLive] %s Redis yazımı başarısız: %s", symbol, exc)


async def _publish_totalamount_snapshot(ta_by_symbol: Dict[str, float]) -> None:
    """Tüm adayların güncel Totalamount değerini + sıralarını tek bir Redis
    key'ine yazar (Rank panelinin tablo ihtiyacı için)."""
    if not ta_by_symbol:
        return
    ordered = sorted(ta_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    result = [
        {"symbol": sym, "value": round(val, 4), "rank": idx + 1}
        for idx, (sym, val) in enumerate(ordered)
    ]
    redis = RedisClient.get_client()
    try:
        await asyncio.wait_for(
            redis.set("totalamount_rank1:snapshot", json.dumps(result), ex=_LIVE_TTL),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TotalamountLive] snapshot yazılamadı: %s", exc)


async def _close_trade(trade_id: int, exit_price: float) -> None:
    """Bir totalamount_rank1 PaperTrade'ini ters sinyal sebebiyle kapatır —
    PaperTradeManager._apply_close ile AYNI mantık (fee/PnL/portfolio)."""

    async def _do_close() -> None:
        async with get_session() as session:
            result = await session.execute(_sel(_PT).where(_PT.id == trade_id))
            trade = result.scalars().first()
            if trade is None or trade.status != "open":
                return
            pf_result = await session.execute(_sel(_PP).where(_PP.strategy == trade.strategy))
            portfolio = pf_result.scalars().first()
            totalamount_rank1_manager._apply_close(  # pylint: disable=protected-access
                trade, exit_price, "reversal", portfolio
            )
            if portfolio:
                session.add(portfolio)
            await session.commit()

    await run_with_db_timeout(_do_close())


async def totalamount_rank1_loop() -> None:
    """Devisso Döngüsü / Totalamount Rank-1: aktif Supertrend Long sinyali
    olan semboller arasında Totalamount'ta (sinyalin AÇILIŞ FİYATINA göre
    basit % değişim, per-signal reset) 1. sıraya çıkan sembole Long paper
    trade açar — open_direct zaten açıksa kendi atlıyor. Ayrı bir fazda,
    açık pozisyonları o sembolde Long sinyali artık aktif değilse kapatır
    (ters sinyal, reconciliation, timeout, manuel kapatma — hepsi)."""
    while True:
        try:

            async def _fetch_candidates() -> list:
                async with get_session() as session:
                    result = await session.execute(
                        _sel(_Sig.symbol, _Sig.open_price, _Sig.opened_at).where(
                            _Sig.indicators == _INDICATOR,
                            _Sig.interval == _SIGNAL_INTERVAL,
                            _Sig.signal_type == "Long",
                            _Sig.status == "active",
                        )
                    )
                    return result.all()

            candidates = await run_with_db_timeout(_fetch_candidates())

            best_symbol: Optional[str] = None
            best_ta: Optional[float] = None
            best_df = None
            ta_by_symbol: Dict[str, float] = {}
            now = datetime.now()
            for sym, open_price, opened_at in candidates:
                try:
                    df = await get_cagg_klines(sym, _SIGNAL_INTERVAL, _CA_LIMIT)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug("[%s] CA sorgusu başarısız: %s", sym, exc)
                    continue
                if df is None or df.empty or "open_time" not in df.columns:
                    continue
                if not open_price or opened_at is None:
                    continue
                elapsed_sec = (now - opened_at).total_seconds()
                bars_since_signal = max(0, int(elapsed_sec // 300))  # 5m bar
                try:
                    net = net_ta_series(df, open_price, bars_since_signal)
                    if len(net) == 0:
                        continue
                    ta = float(net[-1])
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                ta_by_symbol[sym] = ta
                await _publish_totalamount_live(sym, df, net)
                if best_ta is None or ta > best_ta:
                    best_ta = ta
                    best_symbol = sym
                    best_df = df

            await _publish_totalamount_snapshot(ta_by_symbol)

            # Isınma koruması: CA'dan hesaplanabilen aday oranı eşiği geçmeden
            # açma fazı atlanır (snapshot yine de yayınlanır, tablo dolar).
            coverage = (len(ta_by_symbol) / len(candidates)) if candidates else 0.0
            if best_symbol is not None and best_df is not None and coverage >= _WARMUP_MIN_COVERAGE:
                try:
                    price = float(best_df["close"].iloc[-1])
                    atr_series = calculate_atr(best_df, period=Config.ATR_PERIOD)
                    atr_val = float(atr_series.iloc[-1])
                    if price > 0 and atr_val > 0:
                        sl_price = price - _SL_ATR * atr_val
                        # 20 Ağu 2026: giriş anındaki VPMV ham bileşenleri — trade_
                        # snapshot.py'nin sonradan bunlara göre GİRİŞE göre %değişim
                        # hesaplayabilmesi için (bu strateji hiç Signal oluşturmadığından
                        # başka hiçbir yerde kaydedilmiyordu). Başarısız olursa None'lar
                        # geçilir, açma engellenmez (best-effort).
                        raw_components = None
                        try:
                            raw_components = compute_raw_components(best_df, "Long")
                        except Exception as _rce:  # pylint: disable=broad-exception-caught
                            logger.debug(
                                "[%s] VPMV ham bileşen hesaplanamadı: %s", best_symbol, _rce
                            )
                        opened = await totalamount_rank1_manager.open_direct(
                            symbol=best_symbol,
                            signal_type="Long",
                            interval="5m",
                            price=price,
                            atr=atr_val,
                            sl_price=sl_price,
                            tp_price=None,
                            note=f"TA={best_ta:.2f} r1/{len(ta_by_symbol)}"[:20],
                            vol_raw=raw_components.get("vol") if raw_components else None,
                            mom_raw=raw_components.get("mom") if raw_components else None,
                            volat_raw=raw_components.get("vlt") if raw_components else None,
                            price_raw=raw_components.get("prc") if raw_components else None,
                        )
                        if opened:
                            logger.info(
                                "★ AÇILDI %s TA=%.2f (hesaplanan=%d/DB=%d)",
                                best_symbol,
                                best_ta,
                                len(ta_by_symbol),
                                len(candidates),
                            )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("[%s] açma hatası: %s", best_symbol, exc)
            elif best_symbol is not None and candidates:
                logger.info(
                    "Isınma bekleniyor: %d/%d aday hazır (%.0f%%), açma atlandı",
                    len(ta_by_symbol),
                    len(candidates),
                    coverage * 100,
                )

            # ── Faz 2: pozisyonun DAYANDIĞI Long sinyali artık aktif değilse kapat ──
            async def _fetch_reversed_open() -> list:
                async with get_session() as session:
                    open_result = await session.execute(
                        _sel(_PT.id, _PT.symbol).where(
                            _PT.strategy == "totalamount_rank1",
                            _PT.status == "open",
                        )
                    )
                    open_trades = open_result.all()
                    if not open_trades:
                        return []
                    symbols = [row.symbol for row in open_trades]
                    still_valid_result = await session.execute(
                        _sel(_Sig.symbol).where(
                            _Sig.indicators == _INDICATOR,
                            _Sig.interval == _SIGNAL_INTERVAL,
                            _Sig.signal_type == "Long",
                            _Sig.status == "active",
                            _Sig.symbol.in_(symbols),
                        )
                    )
                    still_valid = {row[0] for row in still_valid_result.all()}
                    return [row for row in open_trades if row.symbol not in still_valid]

            to_close = await run_with_db_timeout(_fetch_reversed_open())
            for row in to_close:
                try:
                    df = await get_cagg_klines(row.symbol, _SIGNAL_INTERVAL, 2)
                    if df is None or df.empty:
                        continue
                    exit_price = float(df["close"].iloc[-1])
                    await _close_trade(row.id, exit_price)
                    logger.info(
                        "✕ KAPANDI %s @ %.6f (sinyal artık aktif değil)", row.symbol, exit_price
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("[%s] kapatma hatası: %s", row.symbol, exc)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Döngü hatası: %s", exc, exc_info=True)
        await asyncio.sleep(_INTERVAL)
