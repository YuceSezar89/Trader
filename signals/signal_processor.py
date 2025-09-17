import pandas as pd
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from signals.signal_engine import signal_engine
from utils.financial_metrics import calculate_metrics
from signals.signal_lifecycle_manager import signal_lifecycle_manager
from config import Config
from indicators.core import calculate_rsi, calculate_macd
from utils.data_provider import fetch_ohlcv

logger = get_logger(__name__)

async def process_and_enrich_signals(
    symbol: str,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    interval: str
) -> None:
    """
    Bir sembol için teknik sinyalleri hesaplar, finansal metriklerle zenginleştirir
    ve veritabanına kaydeder.

    Args:
        symbol (str): İşlem yapılan sembol (örn. 'ETHUSDT').
        df (pd.DataFrame): Sembolün mum verilerini içeren DataFrame.
        ref_df (pd.DataFrame): Referans piyasanın (örn. 'BTCUSDT') mum verilerini içeren DataFrame.
        interval (str): Mum verisi aralığı (örn. '15m').
    """
    if df.empty or ref_df.empty:
        logger.warning(f"[{symbol}] Sinyal işleme için veri çerçevelerinden biri boş, atlanıyor.")
        return

    # 1. Finansal Metrikleri Hesapla
    alpha, beta, latest_metrics = None, None, {}
    try:
        if len(df) >= 50 and len(ref_df) >= 50:
            # Metrik hesaplaması için DataFrame'leri hazırla (DatetimeIndex ekle)
            df_prepared = df.copy()
            # mypy uyumu: pd.to_datetime çıktısını açıkça Index'e sar
            df_prepared.index = pd.Index(pd.to_datetime(df_prepared['open_time'], unit='ms'))
            ref_df_prepared = ref_df.copy()
            ref_df_prepared.index = pd.Index(pd.to_datetime(ref_df_prepared['open_time'], unit='ms'))

            df_with_metrics = calculate_metrics(df_prepared, ref_df_prepared)
            latest_metrics = df_with_metrics.iloc[-1].to_dict()
            alpha = latest_metrics.get('alpha')
            beta = latest_metrics.get('beta')
            
            # Güvenli loglama
            alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
            beta_str = f"{beta:.4f}" if beta is not None else "N/A"
            logger.info(f"[{symbol}] Finansal metrikler hesaplandı -> Alpha: {alpha_str}, Beta: {beta_str}")
        else:
            logger.warning(f"[{symbol}] Metrik hesaplaması için yeterli veri yok (gerekli: 50), atlanıyor.")
    except Exception as e:
        logger.error(f"[{symbol}] Finansal metrik hesaplama hatası: {e}", exc_info=False)

    # 2. Teknik Sinyalleri Hesapla
    try:
        technical_signals = await signal_engine.calculate_all_signals(df)
    except Exception as e:
        logger.error(f"[{symbol}] Teknik sinyal hesaplama hatası: {e}", exc_info=True)
        return

    # 3. Sinyalleri Zenginleştir ve Kaydet
    if not technical_signals:
        return

    for signal_name, signal_list in technical_signals.items():
        if not isinstance(signal_list, list) or not signal_list:
            continue

        for signal_data in signal_list:
            try:
                # V-P-M onay ve skor hesapla (DB şeması değişmeden, sadece log ve karar için)
                vpms_score = None
                vpm_confirmed = None
                mtf_score: Optional[float] = None
                combined_score: Optional[float] = None
                try:
                    vpm_cfg = getattr(Config, 'VPM', {})
                    mode = vpm_cfg.get('MODE', 'two_of_three')
                    thr = vpm_cfg.get('THRESHOLDS', {})
                    w = vpm_cfg.get('WEIGHTS', {'P': 0.4, 'V': 0.3, 'M': 0.3})

                    # Yön: Long=+1, Short=-1
                    side = 1 if signal_data.get('signal_type') == 'Long' else -1

                    # Metrikler
                    p_pct = latest_metrics.get('percent_price_diff')  # % değişim
                    v_z = latest_metrics.get('normalized_volume_diff')  # z-normalize hacim farkı
                    rsi_delta = latest_metrics.get('rsi_diff')  # RSI farkı (ham)

                    # Güvenli tip dönüşü ve None koruması
                    def safe_float(x):
                        try:
                            return float(x)
                        except Exception:
                            return None

                    p_pct = safe_float(p_pct)
                    v_z = safe_float(v_z)
                    rsi_delta = safe_float(rsi_delta)

                    # Eşikler
                    p_min_abs = float(thr.get('P_MIN_ABS_PCT', 0.3))
                    v_min_z = float(thr.get('V_MIN_Z', 1.0))
                    m_long = float(thr.get('M_RSI_DELTA_LONG', 2.0))
                    m_short = float(thr.get('M_RSI_DELTA_SHORT', -2.0))

                    # Pass koşulları
                    p_pass = (p_pct is not None) and (abs(p_pct) >= p_min_abs) and ((p_pct * side) > 0)
                    v_pass = (v_z is not None) and (v_z >= v_min_z)
                    m_pass = False
                    if rsi_delta is not None:
                        m_pass = (side == 1 and rsi_delta >= m_long) or (side == -1 and rsi_delta <= m_short)

                    passes = [p_pass, v_pass, m_pass]
                    pass_count = sum(1 for x in passes if x)

                    # Skor: işaret ayarı (Short için P ve M ters)
                    p_comp = (p_pct * side) if (p_pct is not None) else 0.0
                    v_comp = v_z if (v_z is not None) else 0.0
                    m_comp = (rsi_delta * side) if (rsi_delta is not None) else 0.0
                    vpms_score = (w.get('P', 0.4) * p_comp) + (w.get('V', 0.3) * v_comp) + (w.get('M', 0.3) * m_comp)

                    # Onay modu
                    if mode == 'and':
                        vpm_confirmed = all(passes)
                    else:  # two_of_three
                        vpm_confirmed = pass_count >= 2

                    logger.info(
                        f"[{symbol}] VPM | type={signal_data.get('signal_type')} p%={p_pct} v_z={v_z} rsi_d={rsi_delta} "
                        f"passes={passes} score={vpms_score:.3f} confirmed={vpm_confirmed}"
                    )
                except Exception as vpm_err:
                    logger.warning(f"[{symbol}] VPM hesaplama atlandı: {vpm_err}")

                # 2. MTF bonus skoru (üst zaman dilimi hizalaması)
                try:
                    mtf_cfg = getattr(Config, 'VPM', {}).get('MTF', {})
                    if mtf_cfg.get('ENABLED', False):
                        tf_map = mtf_cfg.get('TF_MAP', {})
                        upper_tf = tf_map.get(interval)
                        if upper_tf:
                            # Üst TF OHLCV'yi tek kapıdan çek
                            res = await fetch_ohlcv(symbol, upper_tf, limit=300, source='auto')
                            if res is not None and not res.empty and len(res) >= 3:
                                # RSI ve MACD üst TF
                                rsi_series = calculate_rsi(res)
                                _, _, macd_hist = calculate_macd(res, fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)

                                # Delta hesapları (son iki KAPANMIŞ bar)
                                if len(rsi_series.dropna()) >= 2 and len(macd_hist.dropna()) >= 2:
                                    rsi_delta = float(rsi_series.iloc[-1] - rsi_series.iloc[-2])
                                    macd_delta = float(macd_hist.iloc[-1] - macd_hist.iloc[-2])

                                    side = 1 if signal_data.get('signal_type') == 'Long' else -1
                                    rsi_thr = mtf_cfg.get('RSI_DELTA_THR', {'long':2.0,'short':-2.0})
                                    macd_thr = mtf_cfg.get('MACD_HIST_DELTA_THR', {'long':0.5,'short':-0.5})

                                    rsi_pass = (side == 1 and rsi_delta >= float(rsi_thr.get('long',2.0))) or (side == -1 and rsi_delta <= float(rsi_thr.get('short',-2.0)))
                                    macd_pass = (side == 1 and macd_delta >= float(macd_thr.get('long',0.5))) or (side == -1 and macd_delta <= float(macd_thr.get('short',-0.5)))

                                    # Skor: her iki delta'yı yönlü normalize et, ortalamasını al ve cap uygula
                                    rsi_comp = rsi_delta * side
                                    macd_comp = macd_delta * side
                                    raw_score = (rsi_comp + macd_comp) / 2.0
                                    cap = float(mtf_cfg.get('SCORE_CAP', 1.0))
                                    mtf_score = max(0.0, min(cap, raw_score)) if (rsi_pass or macd_pass) else 0.0

                                    # Birleşik skor
                                    if vpms_score is not None:
                                        mtf_weight = float(mtf_cfg.get('WEIGHT', 0.2))
                                        combined_score = float(vpms_score) + mtf_weight * float(mtf_score)

                                    logger.info(
                                        f"[{symbol}] MTF | upper_tf={upper_tf} rsi_d={rsi_delta:.3f} macd_d={macd_delta:.3f} "
                                        f"mtf_score={mtf_score} combined={combined_score}"
                                    )
                            else:
                                logger.warning(f"[{symbol}] MTF: Üst TF verisi boş veya yetersiz (tf={upper_tf}). Varsayılan skorlar uygulanacak.")
                except Exception as mtf_err:
                    logger.warning(f"[{symbol}] MTF hesaplama atlandı: {mtf_err}")

                # Temel sinyal verilerini hazırla
                enriched_signal = {
                    'symbol': symbol,
                    'interval': interval,
                    **signal_data,
                    'alpha': alpha,
                    'beta': beta,
                    'sharpe_ratio': latest_metrics.get('sharpe_ratio'),
                    'sortino_ratio': latest_metrics.get('sortino_ratio'),
                    'calmar_ratio': latest_metrics.get('calmar_ratio'),
                    'omega_ratio': latest_metrics.get('omega_ratio'),
                    'treynor_ratio': latest_metrics.get('treynor_ratio'),
                    'information_ratio': latest_metrics.get('information_ratio'),
                    'scaled_avg_normalized': latest_metrics.get('scaled_avg_normalized'),
                    'normalized_composite': latest_metrics.get('normalized_composite'),
                    'normalized_price_change': latest_metrics.get('normalized_price_diff'), # financial_metrics'teki 'normalized_price_diff'i DB'deki 'normalized_price_change'e mapliyoruz
                    'vpms_score': float(vpms_score) if vpms_score is not None else None,
                    'vpm_confirmed': bool(vpm_confirmed) if vpm_confirmed is not None else None,
                    'mtf_score': float(mtf_score) if mtf_score is not None else 0.0,
                    'vpms_mtf_score': (
                        float(combined_score)
                        if combined_score is not None
                        else (float(vpms_score) if vpms_score is not None else None)
                    ),
                }

                logger.info(f"[{symbol}] Veritabanına kaydedilecek sinyal bulundu: {signal_name} - {enriched_signal.get('signal_type')}")
                signal_id = await signal_lifecycle_manager.add_new_signal(enriched_signal)
                logger.info(f"[{symbol}] Sinyal lifecycle manager ile kaydedildi: ID {signal_id}")

            except Exception as e:
                logger.error(f"[{symbol}] Sinyal veritabanına kaydedilirken hata: {e}", exc_info=True)
