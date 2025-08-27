"""
TRader Panel - Özel Exception Sınıfları
Sistemdeki farklı hata türleri için özelleştirilmiş exception'lar
"""

from typing import Optional, Any, Dict


class TRaderBaseException(Exception):
    """TRader Panel için temel exception sınıfı"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            return f"{self.message} ({details_str})"
        return self.message


class DataError(TRaderBaseException):
    """Veri ile ilgili hatalar"""
    pass


class APIError(TRaderBaseException):
    """API çağrıları ile ilgili hatalar"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 endpoint: Optional[str] = None, **kwargs):
        details = kwargs
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint
        super().__init__(message, details)


class BinanceAPIError(APIError):
    """Binance API özel hatalar"""
    pass


class CalculationError(TRaderBaseException):
    """İndikatör hesaplama hatalar"""
    
    def __init__(self, message: str, indicator: Optional[str] = None, 
                 symbol: Optional[str] = None, **kwargs):
        details = kwargs
        if indicator:
            details['indicator'] = indicator
        if symbol:
            details['symbol'] = symbol
        super().__init__(message, details)


class DatabaseError(TRaderBaseException):
    """Veritabanı işlem hatalar"""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, **kwargs):
        details = kwargs
        if operation:
            details['operation'] = operation
        if table:
            details['table'] = table
        super().__init__(message, details)


class ConfigurationError(TRaderBaseException):
    """Konfigürasyon hatalar"""
    pass


class ValidationError(TRaderBaseException):
    """Veri doğrulama hatalar"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        details = kwargs
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        super().__init__(message, details)


class SignalError(TRaderBaseException):
    """Sinyal üretimi hatalar"""
    
    def __init__(self, message: str, signal_type: Optional[str] = None, 
                 symbol: Optional[str] = None, **kwargs):
        details = kwargs
        if signal_type:
            details['signal_type'] = signal_type
        if symbol:
            details['symbol'] = symbol
        super().__init__(message, details)


class ScannerError(TRaderBaseException):
    """Scanner işlem hatalar"""
    
    def __init__(self, message: str, scanner_type: Optional[str] = None, 
                 symbols_count: Optional[int] = None, **kwargs):
        details = kwargs
        if scanner_type:
            details['scanner_type'] = scanner_type
        if symbols_count:
            details['symbols_count'] = symbols_count
        super().__init__(message, details)


# Hata kodları için enum
class ErrorCodes:
    """Standart hata kodları"""
    
    # Data errors
    DATA_NOT_FOUND = "DATA_001"
    DATA_INSUFFICIENT = "DATA_002"
    DATA_INVALID_FORMAT = "DATA_003"
    DATA_EMPTY = "DATA_004"
    
    # API errors
    API_TIMEOUT = "API_001"
    API_RATE_LIMIT = "API_002"
    API_INVALID_RESPONSE = "API_003"
    API_CONNECTION_ERROR = "API_004"
    
    # Calculation errors
    CALC_INVALID_PERIOD = "CALC_001"
    CALC_MISSING_COLUMN = "CALC_002"
    CALC_INSUFFICIENT_DATA = "CALC_003"
    CALC_DIVISION_BY_ZERO = "CALC_004"
    
    # Database errors
    DB_CONNECTION_ERROR = "DB_001"
    DB_QUERY_ERROR = "DB_002"
    DB_CONSTRAINT_ERROR = "DB_003"
    DB_TIMEOUT = "DB_004"
    
    # Signal errors
    SIGNAL_GENERATION_ERROR = "SIGNAL_001"
    SIGNAL_VALIDATION_ERROR = "SIGNAL_002"
    SIGNAL_SAVE_ERROR = "SIGNAL_003"


def create_error_with_code(error_class: type, code: str, message: str, **kwargs) -> TRaderBaseException:
    """Hata kodu ile birlikte exception oluştur"""
    details = kwargs
    details['error_code'] = code
    return error_class(message, details)


# Yaygın hata durumları için yardımcı fonksiyonlar
def raise_data_not_found(symbol: str, data_type: str = "data"):
    """Veri bulunamadı hatası"""
    raise create_error_with_code(
        DataError, 
        ErrorCodes.DATA_NOT_FOUND,
        f"{data_type} not found for symbol {symbol}",
        symbol=symbol,
        data_type=data_type
    )


def raise_insufficient_data(symbol: str, required: int, available: int):
    """Yetersiz veri hatası"""
    raise create_error_with_code(
        DataError,
        ErrorCodes.DATA_INSUFFICIENT,
        f"Insufficient data for {symbol}: required {required}, available {available}",
        symbol=symbol,
        required=required,
        available=available
    )


def raise_api_timeout(endpoint: str, timeout: float):
    """API timeout hatası"""
    raise create_error_with_code(
        APIError,
        ErrorCodes.API_TIMEOUT,
        f"API timeout for {endpoint} after {timeout}s",
        endpoint=endpoint,
        timeout=timeout
    )


def raise_calculation_error(indicator: str, symbol: str, reason: str):
    """İndikatör hesaplama hatası"""
    raise create_error_with_code(
        CalculationError,
        ErrorCodes.CALC_INVALID_PERIOD,
        f"Failed to calculate {indicator} for {symbol}: {reason}",
        indicator=indicator,
        symbol=symbol,
        reason=reason
    )


def raise_missing_column(column: str, available_columns: list):
    """Eksik sütun hatası"""
    raise create_error_with_code(
        ValidationError,
        ErrorCodes.CALC_MISSING_COLUMN,
        f"Required column '{column}' not found. Available: {available_columns}",
        column=column,
        available_columns=available_columns
    )
