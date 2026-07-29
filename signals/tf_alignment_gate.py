"""
TF Hizalanma + Erken Ayrışma — canlı entegrasyon (18 Tem 2026, 20 Tem 2026'da
dinamik-eşik mimarisine geçildi).

Doğrulanmış bulgu (bkz. [[project_tf_alignment_early_divergence]]): 1h+4h
Heikin Ashi'nin İKİSİ DE sinyal yönüyle uyumluysa VE aynı yöndeki diğer
adaylara göre en çok erken ayrışanlardansa, edge daha güçlü. HA_Cross ve
RSI_Cross(9,24)'te bağımsız doğrulandı.

20 Tem mimarisi (kullanıcı kararı — eski "3 bar sabit bekle + dar kohort"
mimarisi, ACE örneğinde en iyi adayı max_open doluluğu yüzünden kaybetmişti
ve backtest sabit pencerenin gerçek ayrışmayı geç/erken yakalayabildiğini
gösterdi):
  - Aday süresiz izlemeye alınır (sabit "decide_at" yok)
  - Her değerlendirme turunda (20sn), erken_pct'i AYNI YÖNDEKİ (Long
    Long'larla, Short Short'larla) diğer bekleyen adaylara göre percentile'a
    çevrilir — dar "aynı dakika kohortu" değil, geniş "şu an izlenen herkes"
  - Percentile >= %80 VE early_pct > 0 ise otomatik açılır
  - Minimum popülasyon: bir yönde en az 10 aday yoksa o yön için otomatik
    açma değerlendirilmez (güvenlik, RSI Cross gate'teki min_universe
    mantığıyla aynı)
  - Sembolde zaten açık pozisyon varsa (otomatik veya manuel) aday listeden
    düşer — manuel "Aç" butonuyla açılanlar da bu yolla temizlenir
  - Altındaki sinyal kapanırsa (reversal/timeout/manuel) hiç açılmadan düşer

Akış: register_candidate() sinyal açılışında çağrılır (signal_processor.py)
-> HA hizalıysa VE sembolde açık pozisyon/bekleyen aday yoksa listeye eklenir
-> tf_alignment_eval_loop() periyodik olarak değerlendirir.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import select

from database.crud import get_cagg_klines
from database.engine import get_session, run_with_db_timeout
from database.models import PaperTrade, Signal
from signals.paper_trade_manager import tf_alignment_live_manager
from utils.redis_client import SAFE_EXTERNAL_TIMEOUT, RedisClient

logger = logging.getLogger(__name__)

_MIN_POPULATION = 10
_PERCENTILE_THRESHOLD = 0.80
_EVAL_INTERVAL_SECONDS = 20
_STRATEGY = "tf_alignment_live"

_CANDIDATES_REDIS_KEY = "tf_alignment:candidates"
_CANDIDATES_TTL = 60

_HA_CACHE: dict[str, dict] = {}
_HA_CACHE_TTL = 300.0  # 1h/4h Heikin Ashi rengi bu sürede nadiren değişir

_pending: list[dict] = []
_pending_lock = asyncio.Lock()


def _heikin_ashi_bull(df: pd.DataFrame) -> Optional[bool]:
    """Son kapanmış barın Heikin Ashi rengi — research/pattern_lab script'lerindeki
    (ör. rsi_cross_tf_alignment_bt.py::_compute_ha) formülle BİREBİR aynı."""
    if df is None or len(df) < 2:
        return None
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = ha_close.copy()
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return bool(ha_close[-1] > ha_open[-1])


async def _get_ha_alignment(symbol: str, signal_type: str) -> bool:
    """1h+4h Heikin Ashi'nin İKİSİ DE sinyal yönüyle uyumlu mu — 5 dakika
    cache'li (her sinyalde ayrı DB sorgusu gereksiz, renk nadiren değişir)."""
    now = time.monotonic()
    cached = _HA_CACHE.get(symbol)
    if cached is not None and now - cached["ts"] < _HA_CACHE_TTL:
        bull_1h, bull_4h = cached["bull_1h"], cached["bull_4h"]
    else:
        try:
            df_1h = await get_cagg_klines(symbol, "1h", 60, closed_only=True)
            df_4h = await get_cagg_klines(symbol, "4h", 60, closed_only=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("[TFAlignment] %s HA verisi çekilemedi: %s", symbol, exc)
            return False
        bull_1h = _heikin_ashi_bull(df_1h)
        bull_4h = _heikin_ashi_bull(df_4h)
        _HA_CACHE[symbol] = {"ts": now, "bull_1h": bull_1h, "bull_4h": bull_4h}

    if bull_1h is None or bull_4h is None:
        return False
    want_bull = signal_type == "Long"
    return bull_1h == want_bull and bull_4h == want_bull


async def _has_open_position(symbol: str) -> bool:
    async with get_session() as session:
        result = await run_with_db_timeout(
            session.execute(
                select(PaperTrade.id).where(
                    PaperTrade.strategy == _STRATEGY,
                    PaperTrade.symbol == symbol,
                    PaperTrade.status == "open",
                )
            )
        )
        return result.scalars().first() is not None


async def _open_position_symbols() -> set[str]:
    async with get_session() as session:
        result = await run_with_db_timeout(
            session.execute(
                select(PaperTrade.symbol).where(
                    PaperTrade.strategy == _STRATEGY,
                    PaperTrade.status == "open",
                )
            )
        )
        return set(result.scalars().all())


async def _closed_signal_ids(signal_ids: list[int]) -> set[int]:
    if not signal_ids:
        return set()
    async with get_session() as session:
        result = await run_with_db_timeout(
            session.execute(
                select(Signal.id).where(Signal.id.in_(signal_ids), Signal.status == "closed")
            )
        )
        return set(result.scalars().all())


async def register_candidate(signal_data: dict, signal_id: Optional[int]) -> None:
    """RSI_Cross(9,24) veya HA_Cross sinyali açıldığında signal_processor.py'den
    çağrılır. 1h+4h HA hizalı değilse, sembolde zaten açık pozisyon/bekleyen
    aday varsa burada sessizce elenir."""
    if signal_id is None:
        return
    indicators = signal_data.get("indicators")
    if indicators not in ("RSI_Cross(9,24)", "HA_Cross"):
        return
    symbol = signal_data.get("symbol")
    signal_type = signal_data.get("signal_type")
    interval = signal_data.get("interval")
    opened_at = signal_data.get("opened_at")
    open_price = signal_data.get("open_price")
    if not symbol or not signal_type or not interval or opened_at is None or not open_price:
        return

    if not await _get_ha_alignment(symbol, signal_type):
        return

    lock = _pending_lock
    async with lock:
        if any(c["symbol"] == symbol for c in _pending):
            return

    if await _has_open_position(symbol):
        return

    async with lock:
        _pending.append(
            {
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "interval": interval,
                "opened_at": opened_at,
                "open_price": float(open_price),
                "signal_data": signal_data,
            }
        )
    logger.info("[TFAlignment] [%s] %s %s hizalı aday izlemeye alındı", symbol, interval, signal_type)


async def load_pending_from_db() -> None:
    """Restart sonrası bekleyen aday havuzunu yeniden kurar — `_pending`
    sadece bellekte tutuluyor, her restart'ta sıfırlanıyordu (20 Tem 2026,
    kullanıcı restart sonrası adayların kaybolduğunu fark etti). DB'deki
    hâlâ aktif RSI_Cross/HA_Cross sinyalleri taranır, her biri
    register_candidate() ile AYNI filtrelerden (HA hizalanma, açık pozisyon,
    dedup) geçirilerek _pending'e eklenir."""
    async with get_session() as session:
        result = await run_with_db_timeout(
            session.execute(
                select(Signal).where(
                    Signal.status == "active",
                    Signal.indicators.in_(("RSI_Cross(9,24)", "HA_Cross")),
                )
            )
        )
        actives = result.scalars().all()

    for sig in actives:
        signal_data = {
            "symbol": sig.symbol,
            "interval": sig.interval,
            "indicators": sig.indicators,
            "signal_type": sig.signal_type,
            "opened_at": sig.opened_at,
            "open_price": sig.open_price,
            "vpms_score": sig.vpms_score,
            "mtf_score": sig.mtf_score,
            "atr": sig.atr,
            "z_score_entry": sig.z_score_entry,
        }
        await register_candidate(signal_data, sig.id)

    logger.info(
        "[TFAlignment] restart sonrası %d aktif sinyal tarandı, %d aday geri yüklendi",
        len(actives),
        len(_pending),
    )


async def _current_price(symbol: str, interval: str) -> Optional[float]:
    try:
        df = await get_cagg_klines(symbol, interval, 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


async def _evaluate_candidates() -> None:
    lock = _pending_lock
    async with lock:
        snapshot = list(_pending)
    if not snapshot:
        return

    # 1) Zaten pozisyonu açılmış olanları (otomatik VEYA manuel) temizle
    open_symbols = await _open_position_symbols()
    if open_symbols:
        async with lock:
            _pending[:] = [c for c in _pending if c["symbol"] not in open_symbols]
        snapshot = [c for c in snapshot if c["symbol"] not in open_symbols]
    if not snapshot:
        return

    # 2) Altındaki sinyal kapanmışsa (reversal/timeout/manuel) hiç açılmadan temizle
    closed_ids = await _closed_signal_ids([c["signal_id"] for c in snapshot])
    if closed_ids:
        async with lock:
            _pending[:] = [c for c in _pending if c["signal_id"] not in closed_ids]
        snapshot = [c for c in snapshot if c["signal_id"] not in closed_ids]
    if not snapshot:
        return

    # 3) Güncel early_pct hesapla
    scored: list[tuple[dict, float]] = []
    for c in snapshot:
        price = await _current_price(c["symbol"], c["interval"])
        if price is None:
            continue
        side = 1.0 if c["signal_type"] == "Long" else -1.0
        early_pct = (price - c["open_price"]) / c["open_price"] * 100.0 * side
        scored.append((c, early_pct))

    # 4) Yöne göre ayrı popülasyonda percentile hesapla, eşiği geçenleri aç
    for direction in ("Long", "Short"):
        group = [(c, e) for c, e in scored if c["signal_type"] == direction]
        if len(group) < _MIN_POPULATION:
            continue
        group.sort(key=lambda item: item[1])
        n = len(group)
        for idx, (c, early_pct) in enumerate(group):
            rank_pct = (idx + 1) / n
            if rank_pct < _PERCENTILE_THRESHOLD or early_pct <= 0:
                continue
            price_now = await _current_price(c["symbol"], c["interval"])
            if price_now is None:
                continue
            logger.info(
                "[TFAlignment] [%s] dinamik eşik geçti (%s, rank=%.2f, early_pct=%+.3f) — açılmaya çalışılıyor",
                c["symbol"],
                direction,
                rank_pct,
                early_pct,
            )
            await tf_alignment_live_manager.on_new_signal(
                signal_data=c["signal_data"],
                signal_id=c["signal_id"],
                current_price=price_now,
            )
            # on_new_signal sessizce no-op olabilir (max_open dolu, zaten açık
            # pozisyon vb.) — gerçekten açıldı mı kontrol etmeden adayı listeden
            # düşürmek, kapasite dolu anlarda adayları boşa harcardı (20 Tem
            # bug'ı — 45 "açılıyor" logu, sadece 12 gerçek işlem).
            if await _has_open_position(c["symbol"]):
                logger.info("[TFAlignment] [%s] gerçekten açıldı, listeden düşürülüyor", c["symbol"])
                async with lock:
                    _pending[:] = [p for p in _pending if p["signal_id"] != c["signal_id"]]
            else:
                logger.debug(
                    "[TFAlignment] [%s] açılamadı (muhtemelen max_open dolu), adayda kalıyor",
                    c["symbol"],
                )


async def _publish_candidates_snapshot() -> None:
    """Bekleyen adayları desktop panelin okuyabilmesi için Redis'e yazar."""
    lock = _pending_lock
    async with lock:
        snapshot = list(_pending)

    redis = RedisClient.get_client()
    if not snapshot:
        try:
            await asyncio.wait_for(
                redis.delete(_CANDIDATES_REDIS_KEY), timeout=SAFE_EXTERNAL_TIMEOUT
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return

    rows = []
    for c in snapshot:
        price = await _current_price(c["symbol"], c["interval"])
        early_pct = None
        if price is not None:
            side = 1.0 if c["signal_type"] == "Long" else -1.0
            early_pct = round((price - c["open_price"]) / c["open_price"] * 100.0 * side, 3)
        rows.append(
            {
                "symbol": c["symbol"],
                "signal_type": c["signal_type"],
                "interval": c["interval"],
                "indicators": c["signal_data"].get("indicators"),
                "open_price": c["open_price"],
                "opened_at": c["opened_at"].isoformat()
                if isinstance(c["opened_at"], datetime)
                else str(c["opened_at"]),
                "early_pct": early_pct,
            }
        )

    try:
        await asyncio.wait_for(
            redis.set(_CANDIDATES_REDIS_KEY, json.dumps(rows), ex=_CANDIDATES_TTL),
            timeout=SAFE_EXTERNAL_TIMEOUT,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TFAlignment] Aday listesi yayınlanamadı: %s", exc)


async def tf_alignment_eval_loop() -> None:
    """Periyodik olarak adayları değerlendirir ve panele yayınlar."""
    while True:
        try:
            await _evaluate_candidates()
            await _publish_candidates_snapshot()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[TFAlignment] Değerlendirme hatası: %s", exc, exc_info=True)
        await asyncio.sleep(_EVAL_INTERVAL_SECONDS)
