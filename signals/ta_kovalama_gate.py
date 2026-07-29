"""
TA-uyumu + kovalama + HA-hizalanma canlı gate'i — 22-24 Tem 2026 "didik didik"
serisinin canlıya alınması (kullanıcı onayı: 24 Tem 2026).

Doğrulanmış bulgu (bkz. [[project_rsi_cross_ta_triple_combo_24tem]],
[[project_ha_cross_short_24tem]], [[project_rsi_cross_kovalama_threshold_sweep_24tem]],
[[project_ha_cross_kovalama_threshold_sweep_24tem]]): RSI_Cross(9,24) VE
HA_Cross sinyallerinde, all_up ZATEN geçmişse:

  Long: percentile(net_ta,1h)>=55 VE percentile(net_ta,4h)>=55 VE
        (percentile>=90 VE hâlâ tırmanıyor, 1h VEYA 4h) VE
        1h+4h Heikin-Ashi ikisi de bullish
        (gerçekçi replay PF~4.6-5.5)
  Short: percentile(net_ta,1h)<=45 VE percentile(net_ta,4h)<=45 VE
         (percentile<=20 VE hâlâ düşüyor, 1h VEYA 4h) — HA'SIZ
         (Short'ta HA katkı sağlamıyor, [[project_ha_cross_short_24tem]])
         (gerçekçi replay PF~3.3-4.5)

Sadece 5m interval'de (en çok test edilen yüzey) — 15m'de sadece RSI_Cross
Long tek yönlü doğrulandı, tam kapsam değil, bu yüzden şimdilik dışarıda.

concurrent_count (son 4 saatte kaç eşyönlü sinyal) FİLTRE OLARAK
KULLANILMIYOR — [[project_combo_clustering_threshold_24tem]]'de gerçek ama
ödünleşimli (toplam kâr azaltıyor) bulunduğu için kullanıcı kararıyla sadece
KAYDEDİLİYOR, ileride ML/eşik kararı için veri birikecek.

Akış: evaluate_and_open() sinyal açılışında signal_processor.py'den çağrılır
-> TA(1h+4h) + (Long'da) HA hesaplanır -> şart sağlanırsa
ta_kovalama_live_manager.on_new_signal() ile pozisyon açılır (scale-in
destekli — bkz. paper_trade_manager.py::allow_scale_in).

24 Tem 2026 (aynı gün, akşam) — KÖTÜ-KOMBİNASYON ELEME'si eklendi (Long'a
özel, Short'ta test edilmedi): giriş anındaki 5m Shannon Entropy (pencere=20,
kutu=5, log-return dağılımının düzensizliği) VE Volume TA percentile'ı
(net_ta ile birebir tarif — 12h döngü-sıfırlamalı kümülatif net-alıcı hacim
yüzdesi) AYNI ANDA kötü tarafta ise (entropy>=0.8 VE pct_volta<=30) işlem
AÇILMAZ. Bu iki metrik birbirinden bağımsız (korelasyon 0.019) ve "her ikisi
de kötü" alt-kümesi backtest'te WR'yi %74→%49'a (yazı-tura), medyanı
negatife düşürüyordu — bkz. konuşma geçmişi, backtest script'leri:
research/pattern_lab/rsi_cross_ta_triple_entropy_bt.py,
research/pattern_lab/rsi_cross_ta_triple_volume_ta_bt.py. "İyi" tarafı
ARAMIYORUZ (marjinal katkı azdı, sıklığı çok düşürüyordu) — sadece en kötü
alt-kümeyi eliyoruz.

25 Tem 2026 — Short'a TF-UYUM şartı eklendi (kullanıcının PRLUSDT'de canlıda
gözlemlediği somut vaka: 1h eğimi ZATEN pozitife dönmüşken sistem 4h
bacağından geçip Short açmıştı). Test: research/pattern_lab/
rsi_cross_short_tf_conflict_bt.py — 5m/15m/1h/4h'nin HER BİRİNİN eğimi
(kapanmış barla, repaint yok) Short yönüyle (negatif) uyumlu mu sayılıyor;
sadece 2/4 uyumluysa (yarı yarıya bölünmüş) grup başabaş/kayıp (PF 0.996,
n=26), >=3/4 şartı bu kötü alt-kümeyi eleyip PF'yi 4.50'den 5.41'e
çıkarıyor (placebo %0.0, split-period ikinci yarıda zayıflıyor ama
PF>1 kalıyor — n=227/253, hacmin sadece %10'u kayboluyor). 15m'nin
karıştığı çelişkiler önemsiz çıktı (placebo >%60), asıl belirleyici
1h/4h/5m'nin birbiriyle uyumu.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from database.crud import get_cagg_klines
from database.engine import get_session, run_with_db_timeout
from database.models import PaperTrade
from signals.paper_trade_manager import ta_kovalama_live_manager
from signals.tf_alignment_gate import _get_ha_alignment

logger = logging.getLogger(__name__)

_ALLOWED_INDICATORS = ("RSI_Cross(9,24)", "HA_Cross")
_ALLOWED_INTERVAL = "5m"

_CYCLE_MS = 12 * 3600 * 1000
_TA_LOOKBACK = 200
_TA_MIN_LOOKBACK = 50
_TA_FETCH_LIMIT = 220
_TA_CACHE_TTL = 300.0  # 1h/4h TA nadiren değişir, HA cache'iyle aynı disiplin

_LONG_TA_BASE_TH = 55.0
_LONG_KOVALAMA_TH = 90.0
_SHORT_TA_BASE_TH = 45.0
_SHORT_KOVALAMA_TH = 20.0
_SHORT_MIN_TF_AGREE = 3  # 4 TF'den (5m/15m/1h/4h) kaçının eğimi Short'la uyumlu olmalı

_CONCURRENT_WINDOW_HOURS = 4.0

_ENTROPY_WINDOW = 20
_ENTROPY_BINS = 5
_ENTROPY_BAD_TH = 0.8
_VOLTA_BAD_TH = 30.0
_QUALITY_CACHE_TTL = 300.0

_TA_CACHE: dict[str, dict] = {}
_QUALITY_CACHE: dict[str, dict] = {}


def _net_ta_series(df: pd.DataFrame) -> np.ndarray:
    """12 saatte bir (UTC 00:00/12:00) sıfırlanan kümülatif bar-bar % değişim
    toplamı — research/pattern_lab/rsi_cross_ta_alignment_bt.py::_total_amount_series
    ile birebir."""
    t_ms = df["open_time"].to_numpy()
    close = df["close"].to_numpy()
    cycle = (t_ms // _CYCLE_MS) * _CYCLE_MS
    n = len(close)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            p = close[i - 1]
            net[i] = net[i - 1] + ((close[i] - p) / p * 100.0 if p else 0.0)
    return net


def _percentile_now(net_arr: np.ndarray) -> float:
    """Son barın percentile'ı, kendisi HARİÇ son 200 bara göre (min 50 şartı)
    — research/pattern_lab/rsi_cross_ta_percentile_bt.py::_percentile_at ile aynı."""
    j = len(net_arr) - 1
    s = max(0, j - _TA_LOOKBACK)
    q = net_arr[s:j]
    if len(q) < _TA_MIN_LOOKBACK:
        return float("nan")
    return float((q <= net_arr[j]).mean() * 100.0)


def _slope_now(net_arr: np.ndarray) -> float:
    if len(net_arr) < 3:
        return float("nan")
    return float(net_arr[-1] - net_arr[-3])


async def _ta_bundle(symbol: str, tf: str) -> Optional[tuple[float, float]]:
    """(percentile, slope) döner, hesaplanamıyorsa None."""
    now = time.monotonic()
    cache_key = f"{symbol}:{tf}"
    cached = _TA_CACHE.get(cache_key)
    if cached is not None and now - cached["ts"] < _TA_CACHE_TTL:
        return cached["pct"], cached["slope"]

    try:
        df = await get_cagg_klines(symbol, tf, _TA_FETCH_LIMIT, closed_only=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TAKovalama] %s %s kline çekilemedi: %s", symbol, tf, exc)
        return None
    if df is None or df.empty or len(df) < _TA_MIN_LOOKBACK + 3:
        return None

    net_arr = _net_ta_series(df)
    pct = _percentile_now(net_arr)
    slope = _slope_now(net_arr)
    if pct != pct or slope != slope:  # nan kontrolü
        return None

    _TA_CACHE[cache_key] = {"ts": now, "pct": pct, "slope": slope}
    return pct, slope


def _net_volume_ta_series(df: pd.DataFrame) -> np.ndarray:
    """_net_ta_series ile BİREBİR aynı iskelet — bar-bar bileşen fiyat yerine
    net-alıcı yüzdesi (buy_volume-sell_volume)/volume*100 —
    research/pattern_lab/rsi_cross_ta_triple_volume_ta_bt.py::_net_volume_ta_series
    ile aynı."""
    t_ms = df["open_time"].to_numpy()
    volume = df["volume"].to_numpy()
    buy_v = df["buy_volume"].to_numpy()
    sell_v = df["sell_volume"].to_numpy()
    cycle = (t_ms // _CYCLE_MS) * _CYCLE_MS
    n = len(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_imbalance = np.where(volume > 0, (buy_v - sell_v) / volume * 100.0, 0.0)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            net[i] = net[i - 1] + bar_imbalance[i]
    return net


def _shannon_entropy(returns: np.ndarray, n_bins: int) -> Optional[float]:
    """research/pattern_lab/entropy_clean_forward_return_bt.py::_entropy ile
    birebir — eşit-genişlik kutulama, 0-1 normalize."""
    if len(returns) < n_bins * 3:
        return None
    lo, hi = returns.min(), returns.max()
    if hi <= lo:
        return None
    counts, _ = np.histogram(returns, bins=n_bins, range=(lo, hi))
    probs = counts[counts > 0] / counts.sum()
    h = -np.sum(probs * np.log2(probs))
    return float(h / np.log2(n_bins))


async def _quality_bundle(symbol: str) -> Optional[tuple[float, float]]:
    """(entropy, pct_volta) döner, hesaplanamıyorsa None — Long'daki KÖTÜ-
    KOMBİNASYON elemesi için (bkz. modül docstring'i, 24 Tem 2026 akşam)."""
    now = time.monotonic()
    cached = _QUALITY_CACHE.get(symbol)
    if cached is not None and now - cached["ts"] < _QUALITY_CACHE_TTL:
        return cached["entropy"], cached["pct_volta"]

    try:
        df = await get_cagg_klines(symbol, "5m", _TA_FETCH_LIMIT, closed_only=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TAKovalama] %s kalite verisi çekilemedi: %s", symbol, exc)
        return None
    if df is None or df.empty or len(df) < _TA_MIN_LOOKBACK + 3:
        return None

    close = df["close"].to_numpy()
    log_ret = np.diff(np.log(close))
    if len(log_ret) < _ENTROPY_WINDOW + 1:
        return None
    entropy = _shannon_entropy(log_ret[-(_ENTROPY_WINDOW + 1) : -1], _ENTROPY_BINS)
    if entropy is None:
        return None

    vol_ta = _net_volume_ta_series(df)
    pct_volta = _percentile_now(vol_ta)
    if pct_volta != pct_volta:  # nan kontrolü
        return None

    _QUALITY_CACHE[symbol] = {"ts": now, "entropy": entropy, "pct_volta": pct_volta}
    return entropy, pct_volta


async def _concurrent_count(signal_type: str) -> int:
    """Son _CONCURRENT_WINDOW_HOURS saatte aynı yönde KAÇ ta_kovalama_live
    İŞLEMİ açıldı — FİLTRE DEĞİL, sadece kayıt amaçlı
    ([[project_combo_clustering_threshold_24tem]]).

    DİKKAT: backtest'teki (combo_clustering_feature_bt.py) tanımla birebir
    eşleşmesi için ham `signals` tablosu DEĞİL, `paper_trades` (SADECE bizim
    filtremizi geçmiş, gerçekten açılan işlemler) sayılıyor — ham sinyal
    sayısı (1m/5m/15m/1h/4h hepsi dahil, filtresiz) yüzlerce/binlerce
    çıkıyordu, backtest'teki max=41 ile karşılaştırılamaz ölçekteydi (23
    Tem gece canlıda yakalanan bug, ilk gerçek işlemde fark edildi)."""
    cutoff = datetime.now() - timedelta(hours=_CONCURRENT_WINDOW_HOURS)

    async def _do_query() -> int:
        async with get_session() as session:
            result = await session.execute(
                select(func.count()).where(
                    PaperTrade.strategy == "ta_kovalama_live",
                    PaperTrade.signal_type == signal_type,
                    PaperTrade.opened_at >= cutoff,
                )
            )
            return result.scalar() or 0

    try:
        return await run_with_db_timeout(_do_query())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("[TAKovalama] concurrent_count sorgusu başarısız: %s", exc)
        return 0


async def evaluate_and_open(
    signal_data: dict,
    signal_id: Optional[int],
    current_price: float,
    **extra,
) -> None:
    # 27 Tem 2026 — DURDURULDU. Bu gate'i doğrulayan backtest zincirinin
    # kökü (research/pattern_lab/rsi_cross_ta_percentile_bt.py) look-ahead
    # hatası taşıyordu: 1h/4h percentile/eğim, sinyal anında henüz
    # KAPANMAMIŞ barın (searchsorted(...,'right')-1 ile bulunan, ama
    # closed_only disiplini uygulanmayan) nihai kapanışını kullanıyordu.
    # Aynı canlı fonksiyonlarla (closed-bar disiplinli) yeniden hesaplanan
    # "düzeltilmiş" backtest: WR %74→%31, PF 7.6→0.91 (22 Haz-27 Tem, n=72)
    # — canlı sonuca (WR %23, PF≈0.45) çok daha yakın. Gerçek edge yok/negatif.
    return
    if signal_data.get("indicators") not in _ALLOWED_INDICATORS:
        return
    if signal_data.get("interval") != _ALLOWED_INTERVAL:
        return
    if not signal_data.get("all_up"):
        return
    signal_type = signal_data.get("signal_type")
    if signal_type not in ("Long", "Short"):
        return
    symbol = signal_data.get("symbol")
    if not symbol:
        return

    ta_1h = await _ta_bundle(symbol, "1h")
    if ta_1h is None:
        return
    ta_4h = await _ta_bundle(symbol, "4h")
    if ta_4h is None:
        return
    pct_1h, slope_1h = ta_1h
    pct_4h, slope_4h = ta_4h

    ha_aligned: Optional[bool] = None
    entropy: Optional[float] = None
    pct_volta: Optional[float] = None
    agree_count: Optional[int] = None
    slope_5m: Optional[float] = None
    slope_15m: Optional[float] = None
    if signal_type == "Long":
        ta_base = pct_1h >= _LONG_TA_BASE_TH and pct_4h >= _LONG_TA_BASE_TH
        kovalama = (pct_1h >= _LONG_KOVALAMA_TH and slope_1h > 0) or (
            pct_4h >= _LONG_KOVALAMA_TH and slope_4h > 0
        )
        if not (ta_base and kovalama):
            return
        ha_aligned = await _get_ha_alignment(symbol, "Long")
        if not ha_aligned:
            return
        quality = await _quality_bundle(symbol)
        if quality is not None:
            entropy, pct_volta = quality
            if entropy >= _ENTROPY_BAD_TH and pct_volta <= _VOLTA_BAD_TH:
                logger.info(
                    "[TAKovalama] %s Long KÖTÜ-KOMBİNASYON elendi (entropy=%.2f pct_volta=%.1f)",
                    symbol,
                    entropy,
                    pct_volta,
                )
                return
    else:
        ta_base = pct_1h <= _SHORT_TA_BASE_TH and pct_4h <= _SHORT_TA_BASE_TH
        kovalama = (pct_1h <= _SHORT_KOVALAMA_TH and slope_1h < 0) or (
            pct_4h <= _SHORT_KOVALAMA_TH and slope_4h < 0
        )
        if not (ta_base and kovalama):
            return
        # HA_Cross Short'ta HA-hizalanma kasıtlı olarak KULLANILMIYOR.

        ta_5m = await _ta_bundle(symbol, "5m")
        ta_15m = await _ta_bundle(symbol, "15m")
        if ta_5m is None or ta_15m is None:
            return
        _pct_5m, slope_5m = ta_5m
        _pct_15m, slope_15m = ta_15m
        agree_count = sum(
            s < 0 for s in (slope_5m, slope_15m, slope_1h, slope_4h)
        )
        if agree_count < _SHORT_MIN_TF_AGREE:
            logger.info(
                "[TAKovalama] %s Short TF-UYUMSUZLUK elendi (agree=%d/4, "
                "eğimler 5m=%+.2f 15m=%+.2f 1h=%+.2f 4h=%+.2f)",
                symbol,
                agree_count,
                slope_5m,
                slope_15m,
                slope_1h,
                slope_4h,
            )
            return

    concurrent = await _concurrent_count(signal_type)

    enriched = dict(signal_data)
    enriched.update(
        {
            "ta_pct_1h": round(pct_1h, 1),
            "ta_pct_4h": round(pct_4h, 1),
            "ta_slope_1h": round(slope_1h, 3),
            "ta_slope_4h": round(slope_4h, 3),
            "ha_aligned": ha_aligned,
            "concurrent_count": concurrent,
            "kovalama_hit": True,
            "entropy": round(entropy, 3) if entropy is not None else None,
            "pct_volta": round(pct_volta, 1) if pct_volta is not None else None,
            "tf_agree_count": agree_count,
            "ta_slope_5m": round(slope_5m, 3) if slope_5m is not None else None,
            "ta_slope_15m": round(slope_15m, 3) if slope_15m is not None else None,
        }
    )

    logger.info(
        "[TAKovalama] ★ ŞART SAĞLANDI %s %s %s | pct=%.1f/%.1f slope=%+.2f/%+.2f "
        "HA=%s kalabalık=%d",
        symbol,
        signal_type,
        signal_data.get("indicators"),
        pct_1h,
        pct_4h,
        slope_1h,
        slope_4h,
        ha_aligned,
        concurrent,
    )

    await ta_kovalama_live_manager.on_new_signal(
        signal_data=enriched,
        signal_id=signal_id,
        current_price=current_price,
        **extra,
    )
