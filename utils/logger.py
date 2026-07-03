"""
TRader Panel - Merkezi Logging Sistemi
Tüm modüller için standardize edilmiş logging yapısı
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
from config import Config


class TRaderLogger:
    """Merkezi logging sınıfı"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls):
        """Merkezi logging yapılandırması"""
        if cls._configured:
            return
            
        # Log dizinini oluştur
        if not os.path.exists(Config.LOG_DIR):
            os.makedirs(Config.LOG_DIR)
        
        # Root logger yapılandırması
        root_logger = logging.getLogger()
        root_logger.setLevel(Config.LOG_LEVEL)
        
        # Mevcut handler'ları temizle
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler — sadece terminale bağlıyken (nohup/servis modunda
        # stdout services.log'a yönlendirilir, kopya akış dosyayı sınırsız büyütür)
        console_handler = None
        if sys.stdout.isatty():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(Config.LOG_LEVEL)
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)

        # File handler (rotating)
        file_handler = RotatingFileHandler(
            os.path.join(Config.LOG_DIR, 'trader_panel.log'),
            maxBytes=Config.LOG_FILE_MAX_SIZE,
            backupCount=Config.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            Config.LOG_FORMAT,
            datefmt=Config.LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        
        # Handler'ları ekle
        if console_handler is not None:
            root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Also set python-binance websocket client logger to DEBUG to capture CLOSE frames and details
        try:
            ws_logger = logging.getLogger('binance.websocket.websocket_client')
            # Force debug to get protocol-level messages about close frames
            ws_logger.setLevel(logging.DEBUG)
        except Exception:
            # Non-critical if this logger doesn't exist in older/newer lib versions
            root_logger.debug('Could not set debug level for binance.websocket.websocket_client')
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Modül için logger al"""
        if not cls._configured:
            cls.setup_logging()
            
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def get_module_logger(cls, module_file: str) -> logging.Logger:
        """Dosya yolundan modül logger'ı al"""
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        return cls.get_logger(module_name)


# Kolay kullanım için yardımcı fonksiyonlar
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Logger al - modül adı verilmezse çağıran modülün adını kullan"""
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'unknown')
            if name == '__main__':
                filename = caller_frame.f_globals.get('__file__', 'main')
                if filename:
                    name = os.path.splitext(os.path.basename(filename))[0]
                else:
                    name = 'main'
        else:
            name = 'unknown'
    
    # Ensure name is never None
    if name is None:
        name = 'unknown'
    
    return TRaderLogger.get_logger(name)


def setup_module_logger(module_file: str) -> logging.Logger:
    """Modül dosyası için logger kurulumu"""
    return TRaderLogger.get_module_logger(module_file)


# Özel log seviyeleri için yardımcı fonksiyonlar
def log_performance(logger: logging.Logger, message: str, duration: Optional[float] = None):
    """Performans logları için özel format"""
    if duration is not None:
        logger.info(f"[PERF] {message}: {duration:.3f}s")
    else:
        logger.info(f"[PERF] {message}")


def log_signal(logger: logging.Logger, symbol: str, signal_type: str, details: str = ""):
    """Sinyal logları için özel format"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if details:
        logger.info(f"📊 [{timestamp}] {symbol} | {signal_type} | {details}")
    else:
        logger.info(f"📊 [{timestamp}] {symbol} | {signal_type}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
    """Hata logları için özel format"""
    import traceback
    if context:
        logger.error(f"[ERROR] {context}: {str(error)}")
    else:
        logger.error(f"[ERROR] {str(error)}")
    logger.debug(f"[ERROR_TRACE] {traceback.format_exc()}")


def log_automation(logger: logging.Logger, message: str, level: str = "info"):
    """Otomasyon logları için özel format"""
    formatted_msg = f"[AUTO] {message}"
    if level == "info":
        logger.info(formatted_msg)
    elif level == "warning":
        logger.warning(formatted_msg)
    elif level == "error":
        logger.error(formatted_msg)
    elif level == "debug":
        logger.debug(formatted_msg)


def log_metrics(logger: logging.Logger, symbol: str, metrics: dict):
    """Metrik logları için özel format"""
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    logger.info(f"[METRICS] {symbol} - {metrics_str}")


# Başlangıçta logging'i kur
TRaderLogger.setup_logging()
