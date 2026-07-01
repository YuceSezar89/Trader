"""
TRader Panel - Merkezi Konfigürasyon Dosyası
Bu dosya tüm sabit değerleri ve ayarları içerir.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


class Config:
    # --- Genel Ayarlar ---
    MARKET_REFERENCE_SYMBOL = 'BTCUSDT'  # Alpha ve Beta hesaplamaları için referans sembol
    SYMBOL_LIMIT = 200  # Binance'ten çekilecek en yüksek hacimli sembol sayısı (200 × 6 TF = 1,200 stream → 6 connection)

    """Merkezi konfigürasyon sınıfı"""
    
    # =============================================================================
    # TIMEZONE AYARLARI
    # =============================================================================
    TIMEZONE = 'Europe/Istanbul'  # UTC+3 İstanbul
    TIMEZONE_OFFSET = '+03:00'
    
    # =============================================================================
    # DATABASE AYARLARI (PostgreSQL)
    # =============================================================================
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'trader_panel')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_TIMEOUT = 30  # saniye
    
    # Legacy SQLite (deprecated)
    DB_PATH = 'trader_signals.db'
    DB_BACKUP_DAYS = 7

    # =============================================================================
    # REDIS AYARLARI
    # =============================================================================
    REDIS_URL = "redis://localhost:6379/0"
    REDIS_LIVE_DATA_KEY_PREFIX = "live_kline_data"

    # =============================================================================
    # WEBSOCKET AYARLARI
    # =============================================================================
    WEBSOCKET_TIMEOUT = 45  # Saniye cinsinden. Bu süre boyunca mesaj gelmezse yeniden bağlan.
    # Heartbeat ve reconnect backoff
    WS_HEARTBEAT_CHECK_INTERVAL = 3  # saniye; ana döngüde ping/pong watchdog kontrol sıklığı
    WS_RECONNECT_BACKOFF_BASE = 2    # saniye; üstel backoff tabanı
    WS_RECONNECT_BACKOFF_MAX = 30    # saniye; üstel backoff üst sınırı
    WS_MAX_RECONNECT_ATTEMPTS = 10   # Maksimum yeniden bağlanma denemesi
    WS_CONNECTION_RESET_THRESHOLD = 5 # Bu kadar connection reset sonrası daha uzun bekleme
    WS_PING_INTERVAL = 20            # Ping gönderme aralığı (saniye)
    WS_PONG_TIMEOUT = 10             # Pong yanıt bekleme süresi (saniye)

    # =============================================================================
    # MULTI-TIMEFRAME (MTF) AYARLARI
    # =============================================================================
    MTF_ENABLED = True  # MTF sistemini etkinleştir/devre dışı bırak
    MTF_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d']

    # Interval → dakika eşlemesi (tek kaynak; risk_manager, signals_model, live_data_manager kullanır)
    INTERVAL_MINUTES = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1440,
    }
    MTF_BUFFER_LIMITS = {
        '1m':  1000,
        '5m':  300,
        '15m': 250,
        '30m': 250,
        '1h':  250,
        '4h':  250,
        '6h':  250,
        '8h':  250,
        '12h': 250,
        '1d':  250,
    }

    # WebSocket per-connection stream limit (tunable)
    # Binance combined stream limits can change; lower this if you see server CLOSE frames.
    WS_MAX_STREAMS_PER_CONNECTION = int(os.getenv('WS_MAX_STREAMS_PER_CONNECTION', 100))
    
    
    # =============================================================================
    # BINANCE API AYARLARI
    # =============================================================================
    BINANCE_BASE_URL = 'https://fapi.binance.com'
    BINANCE_FUTURES_INFO_URL = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    # WebSocket base URL to use for streaming. Updated for Binance new endpoint layout.
    # By default point to the new announced market base so that library will form
    # wss://<base>/stream or wss://<base>/ws depending on client usage.
    BINANCE_WS_BASE = os.getenv('BINANCE_WS_BASE', 'wss://fstream.binance.com/market')
    API_TIMEOUT = 10  # saniye
    MAX_RETRIES = 3
    KLINE_INTERVAL = "1m"   # Veri çekme ve sinyal üretme zaman aralığı (hızlı test için 1m)
    MIN_VOLUME_THRESHOLD = 10000 # Bir sembolün izlenmesi için gereken minimum 24 saatlik hacim (USDT)
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
    
    # Risk Yönetimi
    RISK_SL_MULTIPLIER = 1.5   # Stop loss: ATR × bu çarpan
    RISK_TP_MULTIPLIER = 3.0   # Take profit: ATR × bu çarpan (baz)

    # Dinamik R:R — her faktör TP çarpanına +0.5 ekler (max toplam: 4.5)
    DYNAMIC_RR_VPMV_THRESHOLD = 80.0   # VPMV ≥ bu değer → +0.5 TP bonus
    DYNAMIC_RR_MTF_FULL       = 100.0  # MTF = 100% → +0.5 TP bonus
    DYNAMIC_RR_BONUS_INTERVALS = ("15m",)  # Bu TF'lerde → +0.5 TP bonus
    DYNAMIC_RR_TP_MAX         = 4.5    # Maksimum TP çarpanı

    # Konfluans Filtresi
    CONFLUENCE_VPMV_MIN = 75.0
    CONFLUENCE_Z_MIN    = 2.0

    # HA HTF Konfirmasyon
    HA_HTF_MIN_COUNT = 3  # [4h,6h,8h,12h,1d] içinden en az kaçı bullish olmalı

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
        'MIN_SCORE': 50.0,       # VPMV minimum skor — altında sinyal kaydedilmez
        'MIN_RATIO': 1.3,        # sinyal barı / önceki 5 bar ort — altında sinyal atlanır
        'LONG_Z_MAX':  None,     # Long sinyali için z_score üst sınırı (None = kapalı)
        'LONG_Z_MIN': -2.0,      # Long sinyali için z_score alt sınırı (None = kapalı)
        'SHORT_Z_MAX': 2.0,      # Short sinyali için z_score üst sınırı (None = kapalı)
        'SHORT_Z_MIN': None,     # Short sinyali için z_score alt sınırı (None = kapalı)
        'EARLY_EXIT_ENABLED':  True,   # İlk N barda MAE eşiği aşılırsa erken çıkış
        'EARLY_EXIT_MAE_ATR': -1.5,    # ATR cinsinden eşik (negatif = aleyhte)
        'EARLY_EXIT_MAX_BARS': 10,     # Kaç barda aktif (sonrasında devre dışı)
        'WEIGHTS': {
            'V':   0.35,         # Volume   — rolling log+minmax
            'M':   0.35,         # Momentum — yönlü z-score+sigmoid
            'Vlt': 0.20,         # Volatility — ATR percentile rank
            'P':   0.10,         # Price    — rolling IQR
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

    # SuperTrend Filtresi — diğer indikatörler ST yönüyle uyumlu değilse atlanır
    ST_FILTER_ENABLED: bool = True

    # Open Interest Filtresi — OI azalıyorsa sinyal üretilmez
    OI_FILTER_ENABLED: bool = True
    OI_MIN_CHANGE_PCT: float = 3.0  # OI'nin bu kadar düştüğünde sinyal atlanır (%)
    
    # RSI Log Change Ayarları
    RSI_LOG_WINDOW = 14
    RSI_LOG_THRESHOLD = 20
    
    # =============================================================================
    # LOGGING AYARLARI
    # =============================================================================
    LOG_LEVEL = logging.INFO  # Keep-Alive sistem loglarını görmek için INFO'ya çekildi
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
