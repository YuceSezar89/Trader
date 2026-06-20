import pandas as pd
from typing import Optional

from utils.logger import get_logger
from signals.signal_engine import signal_engine
from indicators.financial_metrics import calculate_metrics
from signals.signal_lifecycle_manager import signal_lifecycle_manager
from config import Config
from indicators.core import calculate_rsi, calculate_atr
from signals.vpm_calculator import VPMCalculator
from utils.preprocessing import (
    normalize_volume_0_100,
    normalize_momentum_0_100,
    normalize_volatility_0_100,
    normalize_price_0_100,
)
from utils.redis_client import RedisClient

logger = get_logger(__name__)

_MTF_HIGHER: dict[str, list[str]] = {
    "1m":  ["5m",  "15m"],
    "5m":  ["15m", "1h"],
    "15m": ["1h",  "4h"],
    "1h":  ["4h",  "1d"],
    "4h":  ["1d"],
    "1d":  [],
}


async def _compute_mtf_score(symbol: str, interval: str, signal_type: str) -> float:
    """
    Üst TF'lerin ST yönüne göre MTF konfirmasyon skoru döner (0 / 50 / 100).

    Her üst TF için sinyal yönüyle ST direction uyuşuyorsa +1 puan.
    Toplam puan normalize edilerek 0-100 arasında döner.
    """
    higher_tfs = _MTF_HIGHER.get(interval, [])
    if not higher_tfs:
        return 100.0

    confirmed = 0
    checked = 0
    for tf in higher_tfs:
        try:
            df = await RedisClient.get_mtf_klines(symbol, tf)
            if df is None or df.empty or "st_direction" not in df.columns:
                continue
            valid = df["st_direction"].dropna()
            if valid.empty:
                continue
            st_bullish = float(valid.iloc[-1]) == -1
            checked += 1
            if (signal_type == "Long" and st_bullish) or \
               (signal_type == "Short" and not st_bullish):
                confirmed += 1
        except Exception:
            pass

    if checked == 0:
        return 100.0

    return round(confirmed / checked * 100)


def _compute_vpmv_scores(df: pd.DataFrame, signal_type: str) -> tuple[float, float, float, float]:
    """
    df üzerinden rolling normalize bileşen skorlarını hesaplar.

    Returns:
        (vol_score, momentum_score, vlt_score, price_score) — hepsi 0-100
    """
    side = 1.0 if signal_type == "Long" else -1.0

    # Volume: log + rolling min-max
    vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])

    # Volume yönlü (A/B karşılaştırma — sadece log)
    hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
    buy_vol  = df["volume"] * (df["close"] - df["low"])  / hl_range
    sell_vol = df["volume"] * (df["high"]  - df["close"]) / hl_range
    vol_delta = (buy_vol - sell_vol) * side
    vol_dir_score = float(normalize_momentum_0_100(vol_delta).iloc[-1])
    logger.info(
        "VOL_AB | %s | %s | yonsuz=%.1f yonlu=%.1f fark=%.1f",
        signal_type, df["close"].iloc[-1], vol_score, vol_dir_score, vol_dir_score - vol_score,
    )

    # Momentum: yönlü RSI delta + z-score sigmoid
    rsi_series = calculate_rsi(df, period=14)
    rsi_delta_series = rsi_series.diff().fillna(0.0) * side
    momentum_score = float(normalize_momentum_0_100(rsi_delta_series).iloc[-1])

    # Volatility: ATR percentile rank
    atr_series = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt_score = float(normalize_volatility_0_100(atr_series).iloc[-1])

    # Price: yönlü % değişim + rolling IQR
    price_pct = df["close"].pct_change().fillna(0.0) * 100.0 * side
    price_score = float(normalize_price_0_100(price_pct).iloc[-1])

    return vol_score, momentum_score, vlt_score, price_score


async def process_and_enrich_signals(
    symbol: str,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    interval: str,
    oi_data: Optional[str] = None,
) -> None:
    """
    Bir sembol için teknik sinyalleri hesaplar, finansal metriklerle zenginleştirir
    ve veritabanına kaydeder.
    """
    logger.info(f"[{symbol}] Sinyal işleme başlatıldı - DataFrame boyutu: {len(df)}, Ref boyutu: {len(ref_df)}")

    if df.empty or ref_df.empty:
        logger.warning(f"[{symbol}] Veri çerçevelerinden biri boş, atlanıyor.")
        return

    # 1. Finansal Metrikleri Hesapla
    alpha, beta, latest_metrics = None, None, {}
    try:
        if len(df) >= 50 and len(ref_df) >= 50:
            df_prepared = df.copy()
            df_prepared.index = pd.Index(pd.to_datetime(df_prepared["open_time"], unit="ms"))
            ref_df_prepared = ref_df.copy()
            ref_df_prepared.index = pd.Index(pd.to_datetime(ref_df_prepared["open_time"], unit="ms"))

            df_with_metrics = calculate_metrics(df_prepared, ref_df_prepared)
            latest_metrics = df_with_metrics.iloc[-1].to_dict()
            alpha = latest_metrics.get("alpha")
            beta = latest_metrics.get("beta")

            alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
            beta_str  = f"{beta:.4f}"  if beta  is not None else "N/A"
            logger.info(f"[{symbol}] Finansal metrikler -> Alpha: {alpha_str}, Beta: {beta_str}")
        else:
            logger.warning(f"[{symbol}] Metrik için yeterli veri yok (gerekli: 50), atlanıyor.")
    except Exception as e:
        logger.error(f"[{symbol}] Finansal metrik hatası: {e}", exc_info=False)

    # 2. Teknik Sinyalleri Hesapla
    try:
        logger.info(f"[{symbol}] Teknik sinyal hesaplama başlatılıyor...")
        technical_signals = await signal_engine.calculate_all_signals(df, symbol=symbol, interval=interval)
        logger.info(f"[{symbol}] Teknik sinyal hesaplama tamamlandı - {len(technical_signals) if technical_signals else 0} tür")
    except Exception as e:
        logger.error(f"[{symbol}] Teknik sinyal hatası: {e}", exc_info=True)
        return

    if not technical_signals:
        logger.info(f"[{symbol}] Teknik sinyal bulunamadı.")
        return

    min_vpmv = float(Config.VPM.get("MIN_SCORE", 40.0))
    vpm_weights = Config.VPM.get("WEIGHTS")

    # Mevcut ST yönünü bir kez oku (st_confirmed hesabı için)
    st_direction = None
    if "st_direction" in df.columns and len(df) >= 2:
        st_dir_val = df["st_direction"].iloc[-2]
        if st_dir_val is not None and not (isinstance(st_dir_val, float) and st_dir_val != st_dir_val):
            st_direction = float(st_dir_val)  # -1=bullish, 1=bearish

    for signal_name, signal_list in technical_signals.items():
        if not isinstance(signal_list, list) or not signal_list:
            continue

        for signal_data in signal_list:
            try:
                sig_type = signal_data.get("signal_type", "Long")

                # st_confirmed: ST yönüyle uyumlu mu? (supertrend kendi sinyali için her zaman True)
                if signal_name == "supertrend" or st_direction is None:
                    st_confirmed = True
                else:
                    st_bullish = st_direction == -1
                    st_confirmed = (sig_type == "Long" and st_bullish) or \
                                   (sig_type == "Short" and not st_bullish)
                    if not st_confirmed:
                        logger.info(
                            f"[{symbol}] {signal_name} {sig_type} ST onaysız "
                            f"(ST {'bullish' if st_bullish else 'bearish'}) — yine de kaydediliyor"
                        )

                # 3. VPMV Hesapla
                vpms_score: Optional[float] = None
                try:
                    vol_s, mom_s, vlt_s, prc_s = _compute_vpmv_scores(df, sig_type)
                    vpms_score = VPMCalculator.calculate(
                        vol_score=vol_s,
                        momentum_score=mom_s,
                        vlt_score=vlt_s,
                        price_score=prc_s,
                        weights=vpm_weights,
                    )
                    logger.info(
                        f"[{symbol}] VPMV | type={sig_type} "
                        f"V={vol_s:.1f} M={mom_s:.1f} Vlt={vlt_s:.1f} P={prc_s:.1f} "
                        f"→ score={vpms_score:.1f}"
                    )
                except Exception as vpm_err:
                    logger.warning(f"[{symbol}] VPMV hesaplama atlandı: {vpm_err}")

                # Pre-signal directional volume log (test)
                try:
                    _pre = df.iloc[-6:-1]
                    _hl  = (_pre["high"] - _pre["low"]).clip(lower=1e-8)
                    _bv  = (_pre["volume"] * (_pre["close"] - _pre["low"])  / _hl).sum()
                    _sv  = (_pre["volume"] * (_pre["high"]  - _pre["close"]) / _hl).sum()
                    _tot = _bv + _sv
                    _buy_pct = _bv / _tot * 100 if _tot > 0 else 50.0
                    logger.info(
                        "PREVOL | %s | %s | %s | buy_pct=%.1f",
                        symbol, sig_type, interval, _buy_pct,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

                # 4. Minimum skor filtresi (B kapısı)
                if vpms_score is not None and vpms_score < min_vpmv:
                    logger.info(
                        f"[{symbol}] VPMV={vpms_score:.1f} < {min_vpmv} — sinyal atlandı ({signal_name})"
                    )
                    continue

                # 5. MTF konfirmasyon skoru hesapla
                mtf_score = await _compute_mtf_score(symbol, interval, sig_type)
                logger.info(
                    f"[{symbol}] MTF konfirmasyon | {interval} {sig_type} "
                    f"→ score={mtf_score:.0f} "
                    f"(higher TFs: {_MTF_HIGHER.get(interval, [])})"
                )

                # 6. Z-score hesapla (EMA200 ayrışması)
                z_score_entry = None
                try:
                    if len(df) >= 210 and "close" in df.columns:
                        closes = df["close"].astype(float)
                        ema200 = closes.ewm(span=200, adjust=False).mean()
                        std200 = closes.rolling(200).std()
                        z_score_entry = round(
                            float((closes.iloc[-1] - ema200.iloc[-1]) / (std200.iloc[-1] + 1e-12)), 3
                        )
                except Exception:
                    pass

                z_momentum_ok = (
                    z_score_entry is not None and (
                        (sig_type == "Long"  and z_score_entry > 0) or
                        (sig_type == "Short" and z_score_entry < 0)
                    )
                )
                is_confluence = (
                    vpms_score is not None and vpms_score >= Config.CONFLUENCE_VPMV_MIN and
                    abs(z_score_entry) >= Config.CONFLUENCE_Z_MIN and
                    z_momentum_ok
                )

                if is_confluence:
                    logger.info(
                        "[%s] ★ KONFLUANS SİNYALİ %s %s | VPMV=%.1f Z=%+.2f",
                        symbol, interval, sig_type, vpms_score, z_score_entry,
                    )

                # 7. Sinyali işle
                current_price = float(df["close"].iloc[-1]) if len(df) >= 1 else None
                enriched_signal = {
                    "symbol":         symbol,
                    "interval":       interval,
                    "indicators":     signal_data.get("indicators"),
                    "signal_type":    sig_type,
                    "opened_at":      signal_data.get("timestamp"),
                    "open_price":     signal_data.get("price"),
                    "vpms_score":     float(vpms_score) if vpms_score is not None else None,
                    "mtf_score":      mtf_score,
                    "st_confirmed":   st_confirmed,
                    "rsi":            signal_data.get("rsi"),
                    "strength":       signal_data.get("strength"),
                    "atr":            signal_data.get("atr"),
                    "alpha":          alpha,
                    "beta":           beta,
                    "sharpe_ratio":   latest_metrics.get("sharpe_ratio"),
                    "oi_data":        oi_data,
                    "z_score_entry":  z_score_entry,
                    "is_confluence":  is_confluence,
                }
                logger.info(f"[{symbol}] Sinyal işleniyor: {signal_name} - {sig_type}")
                signal_id = await signal_lifecycle_manager.process(enriched_signal, current_price=current_price)
                logger.info(f"[{symbol}] Sinyal işlendi: ID {signal_id}")

            except Exception as e:
                logger.error(f"[{symbol}] Sinyal kayıt hatası: {e}", exc_info=True)
