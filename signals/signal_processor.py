import pandas as pd
from typing import Optional

from utils.logger import get_logger
from signals.signal_engine import signal_engine
from indicators.financial_metrics import calculate_metrics
from signals.signal_lifecycle_manager import signal_lifecycle_manager
from config import Config
from indicators.core import calculate_rsi, calculate_macd, calculate_atr
from utils.data_provider import fetch_ohlcv
from signals.vpm_calculator import VPMCalculator
from utils.preprocessing import (
    normalize_volume_0_100,
    normalize_momentum_0_100,
    normalize_volatility_0_100,
    normalize_price_0_100,
)

logger = get_logger(__name__)


def _compute_vpmv_scores(df: pd.DataFrame, signal_type: str) -> tuple[float, float, float, float]:
    """
    df üzerinden rolling normalize bileşen skorlarını hesaplar.

    Returns:
        (vol_score, momentum_score, vlt_score, price_score) — hepsi 0-100
    """
    side = 1.0 if signal_type == "Long" else -1.0

    # Volume: log + rolling min-max
    vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])

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
    interval: str
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
        technical_signals = await signal_engine.calculate_all_signals(df)
        logger.info(f"[{symbol}] Teknik sinyal hesaplama tamamlandı - {len(technical_signals) if technical_signals else 0} tür")
    except Exception as e:
        logger.error(f"[{symbol}] Teknik sinyal hatası: {e}", exc_info=True)
        return

    if not technical_signals:
        logger.info(f"[{symbol}] Teknik sinyal bulunamadı.")
        return

    min_vpmv = float(Config.VPM.get("MIN_SCORE", 40.0))
    vpm_weights = Config.VPM.get("WEIGHTS")

    for signal_name, signal_list in technical_signals.items():
        if not isinstance(signal_list, list) or not signal_list:
            continue

        for signal_data in signal_list:
            try:
                sig_type = signal_data.get("signal_type", "Long")

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

                # 4. Minimum skor filtresi (B kapısı)
                if vpms_score is not None and vpms_score < min_vpmv:
                    logger.info(
                        f"[{symbol}] VPMV={vpms_score:.1f} < {min_vpmv} — sinyal atlandı ({signal_name})"
                    )
                    continue

                # 5. MTF bonus skoru
                mtf_score: Optional[float] = None
                combined_score: Optional[float] = None
                try:
                    mtf_cfg = Config.VPM.get("MTF", {})
                    if mtf_cfg.get("ENABLED", False):
                        tf_map = mtf_cfg.get("TF_MAP", {})
                        upper_tf = tf_map.get(interval)
                        if upper_tf:
                            res = await fetch_ohlcv(symbol, upper_tf, limit=300, source="auto")
                            if res is not None and not res.empty and len(res) >= 3:
                                rsi_series = calculate_rsi(res)
                                _, _, macd_hist = calculate_macd(
                                    res,
                                    fast=Config.MACD_FAST,
                                    slow=Config.MACD_SLOW,
                                    signal=Config.MACD_SIGNAL,
                                )
                                if len(rsi_series.dropna()) >= 2 and len(macd_hist.dropna()) >= 2:
                                    rsi_delta  = float(rsi_series.iloc[-1]  - rsi_series.iloc[-2])
                                    macd_delta = float(macd_hist.iloc[-1]   - macd_hist.iloc[-2])

                                    side = 1 if sig_type == "Long" else -1
                                    rsi_thr  = mtf_cfg.get("RSI_DELTA_THR",       {"long": 2.0,  "short": -2.0})
                                    macd_thr = mtf_cfg.get("MACD_HIST_DELTA_THR", {"long": 0.5,  "short": -0.5})

                                    rsi_pass  = (side == 1 and rsi_delta  >= float(rsi_thr.get("long", 2.0)))  or \
                                                (side == -1 and rsi_delta  <= float(rsi_thr.get("short", -2.0)))
                                    macd_pass = (side == 1 and macd_delta >= float(macd_thr.get("long", 0.5))) or \
                                                (side == -1 and macd_delta <= float(macd_thr.get("short", -0.5)))

                                    rsi_comp  = rsi_delta  * side
                                    macd_comp = macd_delta * side
                                    raw_score = (rsi_comp + macd_comp) / 2.0
                                    cap = float(mtf_cfg.get("SCORE_CAP", 1.0))
                                    mtf_score = max(0.0, min(cap, raw_score)) if (rsi_pass or macd_pass) else 0.0

                                    if vpms_score is not None:
                                        mtf_weight = float(mtf_cfg.get("WEIGHT", 0.2))
                                        combined_score = float(vpms_score) + mtf_weight * float(mtf_score)

                                    logger.info(
                                        f"[{symbol}] MTF | upper_tf={upper_tf} "
                                        f"rsi_d={rsi_delta:.3f} macd_d={macd_delta:.3f} "
                                        f"mtf_score={mtf_score} combined={combined_score}"
                                    )
                            else:
                                logger.warning(f"[{symbol}] MTF: Üst TF verisi boş (tf={upper_tf}).")
                except Exception as mtf_err:
                    logger.warning(f"[{symbol}] MTF hesaplama atlandı: {mtf_err}")

                # 6. Sinyali zenginleştir ve kaydet
                signal_data_clean = {k: v for k, v in signal_data.items() if k != "id"}
                enriched_signal = {
                    "symbol":                   symbol,
                    "interval":                 interval,
                    **signal_data_clean,
                    "alpha":                    alpha,
                    "beta":                     beta,
                    "sharpe_ratio":             latest_metrics.get("sharpe_ratio"),
                    "sortino_ratio":            latest_metrics.get("sortino_ratio"),
                    "calmar_ratio":             latest_metrics.get("calmar_ratio"),
                    "omega_ratio":              latest_metrics.get("omega_ratio"),
                    "treynor_ratio":            latest_metrics.get("treynor_ratio"),
                    "information_ratio":        latest_metrics.get("information_ratio"),
                    "scaled_avg_normalized":    latest_metrics.get("scaled_avg_normalized"),
                    "normalized_composite":     latest_metrics.get("normalized_composite"),
                    "normalized_price_change":  latest_metrics.get("normalized_price_diff"),
                    "zscore_ratio_percent":     latest_metrics.get("zscore_ratio_percent"),
                    "vpms_score":               float(vpms_score) if vpms_score is not None else None,
                    "vpm_confirmed":            True,  # filtreden geçti
                    "mtf_score":                float(mtf_score) if mtf_score is not None else 0.0,
                    "vpms_mtf_score": (
                        float(combined_score)
                        if combined_score is not None
                        else (float(vpms_score) if vpms_score is not None else None)
                    ),
                }

                logger.info(f"[{symbol}] Sinyal kaydediliyor: {signal_name} - {sig_type}")
                signal_id = await signal_lifecycle_manager.add_new_signal(enriched_signal)
                logger.info(f"[{symbol}] Sinyal kaydedildi: ID {signal_id}")

            except Exception as e:
                logger.error(f"[{symbol}] Sinyal kayıt hatası: {e}", exc_info=True)
