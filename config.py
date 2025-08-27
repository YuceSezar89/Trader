"""
TRader Panel - Merkezi Konfigürasyon Dosyası
Bu dosya tüm sabit değerleri ve ayarları içerir.
"""

import os
import logging
from typing import Dict, Any


class Config:
    # --- Genel Ayarlar ---
    MARKET_REFERENCE_SYMBOL = 'BTCUSDT'  # Alpha ve Beta hesaplamaları için referans sembol
    SYMBOL_LIMIT = 100  # Binance'ten çekilecek en yüksek hacimli sembol sayısı

    """Merkezi konfigürasyon sınıfı"""
    
    # =============================================================================
    # TIMEZONE AYARLARI
    # =============================================================================
    TIMEZONE = 'Europe/Istanbul'  # UTC+3 İstanbul
    TIMEZONE_OFFSET = '+03:00'
    
    # =============================================================================
    # DATABASE AYARLARI
    # =============================================================================
    DB_PATH = 'trader_signals.db'
    DB_BACKUP_DAYS = 7
    DB_TIMEOUT = 30  # saniye

    # =============================================================================
    # REDIS AYARLARI
    # =============================================================================
    REDIS_URL = "redis://localhost:6379/0"
    REDIS_LIVE_DATA_KEY_PREFIX = "live_kline_data"

    # =============================================================================
    # WEBSOCKET AYARLARI
    # =============================================================================
    WEBSOCKET_TIMEOUT = 60  # Saniye cinsinden. Bu süre boyunca mesaj gelmezse yeniden bağlan.
    # Heartbeat ve reconnect backoff
    WS_HEARTBEAT_CHECK_INTERVAL = 5  # saniye; ana döngüde ping/pong watchdog kontrol sıklığı
    WS_RECONNECT_BACKOFF_BASE = 5    # saniye; üstel backoff tabanı
    WS_RECONNECT_BACKOFF_MAX = 60    # saniye; üstel backoff üst sınırı

    
    # =============================================================================
    # BINANCE API AYARLARI
    # =============================================================================
    BINANCE_BASE_URL = 'https://fapi.binance.com'
    BINANCE_FUTURES_INFO_URL = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    API_TIMEOUT = 10  # saniye
    MAX_RETRIES = 3
    KLINE_INTERVAL = "15m"  # Veri çekme ve sinyal üretme zaman aralığı
    MIN_VOLUME_THRESHOLD = 1000 # Bir sembolün izlenmesi için gereken minimum 24 saatlik hacim
    DEFAULT_LIMIT = 75  # Scanner için varsayılan
    MA200_LIMIT = 250  # MA200 hesaplaması için minimum
    
    # =============================================================================
    # TEKNIK İNDİKATÖR AYARLARI
    # =============================================================================
    
    # RSI Ayarları
    RSI_PERIOD_FAST = 9
    RSI_PERIOD_DEFAULT = 14
    RSI_PERIOD_SLOW = 24
    
    # MA Ayarları
    MA200_PERIOD = 200
    EMA_DEFAULT_PERIOD = 21
    SMA_DEFAULT_PERIOD = 21
    
    # MACD Ayarları
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Diğer İndikatörler
    ATR_PERIOD = 14
    MFI_PERIOD = 14
    ROC_PERIOD = 9
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    WILLIAMS_R_PERIOD = 14
    EMA_ROC_PERIOD = 50
    
    # =============================================================================
    # SİNYAL İŞLEME AYARLARI
    # =============================================================================

    # RSI eşik değerleri (dictionary)
    RSI_THRESHOLDS = {
        'C10': {'long': 10, 'short': -10}
    }
    # MACD eşik değerleri (dictionary)
    MACD_THRESHOLDS = {
        'M2': {'long': 2, 'short': -2},
        'M3': {'long': 3, 'short': -3},
        'M4': {'long': 4, 'short': -4},
        'M5': {'long': 5, 'short': -5}
    }

    # RSI Sinyal Eşikleri
    C10_RSI_THRESHOLD = 10

    # MACD Sinyal Eşikleri
    M2_MACD_THRESHOLD = 2
    M3_MACD_THRESHOLD = 3
    M4_MACD_THRESHOLD = 4
    M5_MACD_THRESHOLD = 5
    
    # --- V-P-M Onay Ayarları ---
    # Not: Şimdilik DB şeması değişmeden kullanılacak; panel ve processor tarafında hesaplanır.
    VPM = {
        'MODE': 'two_of_three',  # options: 'and', 'two_of_three'
        'LOOKBACK': 1,           # aynı bar (1) veya kısa geçmişteki farklar
        'THRESHOLDS': {
            # P: yüzde fiyat değişimi mutlak eşik (örn. 0.3%)
            'P_MIN_ABS_PCT': 0.3,
            # V: hacim için z-normalize edilmiş fark eşiği (financial_metrics -> normalized_volume_diff)
            'V_MIN_Z': 1.0,
            # M: momentum için RSI fark eşiği (AL/SAT işaretleri yönlü)
            'M_RSI_DELTA_LONG': 2.0,
            'M_RSI_DELTA_SHORT': -2.0,
            # Alternatif: MACD histogram delta eşiği eklenebilir (şimdilik hesap yok)
            'M_MACD_HIST_DELTA_LONG': 0.5,
            'M_MACD_HIST_DELTA_SHORT': -0.5,
        },
        'WEIGHTS': {
            'P': 0.4,
            'V': 0.3,
            'M': 0.3,
        },
        # MTF bonus ayarları
        'MTF': {
            'ENABLED': True,
            'WEIGHT': 0.2,
            'SCORE_CAP': 1.0,
            'TF_MAP': {
                '1m': '5m',
                '5m': '15m',
                '15m': '1h',
                '1h': '4h',
                '4h': '1d'
            },
            'RSI_DELTA_THR': {'long': 2.0, 'short': -2.0},
            'MACD_HIST_DELTA_THR': {'long': 0.5, 'short': -0.5},
        }
    }
    
    # =============================================================================
    # SCANNER AYARLARI
    # =============================================================================
    MAX_WORKERS = 10  # ThreadPoolExecutor için
    SCAN_INTERVAL_MINUTES = 15  # Otomatik tarama aralığı
    
    # Scanner Limitleri
    SCANNER_DEFAULT_LIMIT = 75
    SCANNER_MA200_LIMIT = 250
    
    # RSI Crossover Ayarları
    RSI_FAST_WINDOW = 9
    RSI_SLOW_WINDOW = 24
    
    # RSI Log Change Ayarları
    RSI_LOG_WINDOW = 14
    RSI_LOG_THRESHOLD = 20
    
    # =============================================================================
    # LOGGING AYARLARI
    # =============================================================================
    LOG_LEVEL = logging.WARNING
    LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Log dosya ayarları
    LOG_DIR = 'logs'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # =============================================================================
    # STREAMLIT PANEL AYARLARI
    # =============================================================================
    STREAMLIT_PORT = 8501
    STREAMLIT_TITLE = "TRader Panel - Binance Futures Analiz"
    
    # Cache ayarları
    CACHE_TTL = 300  # 5 dakika
    SIGNALS_CACHE_TTL = 60  # 1 dakika
    
    # Tablo ayarları
    MAX_SIGNALS_DISPLAY = 100
    DEFAULT_HOURS_FILTER = 24
    
    # =============================================================================
    # PERFORMANS AYARLARI
    # =============================================================================
    
    # Async ayarları (gelecek için)
    ASYNC_SEMAPHORE_LIMIT = 10
    ASYNC_TIMEOUT = 30
    
    # Batch işlem ayarları
    BATCH_SIZE = 50
    BATCH_TIMEOUT = 5
    
    # =============================================================================
    # ENVIRONMENT AYARLARI
    # =============================================================================
    
    @classmethod
    def get_env_config(cls) -> Dict[str, Any]:
        """Environment değişkenlerinden config değerlerini al"""
        return {
            'DB_PATH': os.getenv('TRADER_DB_PATH', cls.DB_PATH),
            'LOG_LEVEL': os.getenv('TRADER_LOG_LEVEL', cls.LOG_LEVEL),
            'MAX_WORKERS': int(os.getenv('TRADER_MAX_WORKERS', cls.MAX_WORKERS)),
            'API_TIMEOUT': int(os.getenv('TRADER_API_TIMEOUT', cls.API_TIMEOUT)),
        }
    
    @classmethod
    def get_indicator_config(cls) -> Dict[str, int]:
        """Tüm indikatör ayarlarını döndür"""
        return {
            'rsi_fast': cls.RSI_PERIOD_FAST,
            'rsi_default': cls.RSI_PERIOD_DEFAULT,
            'rsi_slow': cls.RSI_PERIOD_SLOW,
            'ma200': cls.MA200_PERIOD,
            'ema_default': cls.EMA_DEFAULT_PERIOD,
            'sma_default': cls.SMA_DEFAULT_PERIOD,
            'atr': cls.ATR_PERIOD,
            'mfi': cls.MFI_PERIOD,
            'roc': cls.ROC_PERIOD,
            'stoch_k': cls.STOCH_K_PERIOD,
            'stoch_d': cls.STOCH_D_PERIOD,
            'williams_r': cls.WILLIAMS_R_PERIOD,
        }
    
    @classmethod
    def get_signal_thresholds(cls) -> Dict[str, float]:
        """Sinyal eşiklerini döndür"""
        return {
            'c10_rsi': cls.C10_RSI_THRESHOLD,
            'm2_macd': cls.M2_MACD_THRESHOLD,
            'm3_macd': cls.M3_MACD_THRESHOLD,
            'm4_macd': cls.M4_MACD_THRESHOLD,
            'm5_macd': cls.M5_MACD_THRESHOLD,
        }


# Kolay erişim için global değişkenler
INDICATORS = Config.get_indicator_config()
THRESHOLDS = Config.get_signal_thresholds()
ENV_CONFIG = Config.get_env_config()

# Geriye uyumluluk için eski değişken isimleri
RSI_PERIOD = Config.RSI_PERIOD_DEFAULT
MA200_PERIOD = Config.MA200_PERIOD
DEFAULT_LIMIT = Config.DEFAULT_LIMIT
