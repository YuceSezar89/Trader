"""
Bar Counter Utility - Sinyal geldiği andan itibaren kaç bar geçtiğini hesaplar
"""
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import re

def parse_interval_to_minutes(interval: str) -> int:
    """
    Interval string'ini dakika cinsinden parse eder.
    
    Args:
        interval: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    
    Returns:
        int: Dakika cinsinden interval süresi
    """
    interval_map = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200  # Yaklaşık 30 gün
    }
    
    if interval in interval_map:
        return interval_map[interval]
    
    # Regex ile parse etmeye çalış
    pattern = r'^(\d+)([mhd])$'
    match = re.match(pattern, interval.lower())
    
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'm':
            return number
        elif unit == 'h':
            return number * 60
        elif unit == 'd':
            return number * 1440
    
    # Varsayılan olarak 1 saat
    return 60

def calculate_bars_since_signal(signal_timestamp: datetime, interval: str, current_time: Optional[datetime] = None) -> int:
    """
    Sinyal geldiği andan itibaren kaç bar geçtiğini hesaplar.
    
    Args:
        signal_timestamp: Sinyalin geldiği zaman
        interval: Zaman aralığı (örn: '1h', '4h', '1d')
        current_time: Şu anki zaman (None ise datetime.now() kullanılır)
    
    Returns:
        int: Geçen bar sayısı
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Zaman farkını hesapla
    time_diff = current_time - signal_timestamp
    total_minutes = time_diff.total_seconds() / 60
    
    # Interval'ı dakika cinsine çevir
    interval_minutes = parse_interval_to_minutes(interval)
    
    # Bar sayısını hesapla
    bars_passed = int(total_minutes // interval_minutes)
    
    return max(0, bars_passed)  # Negatif değer döndürme

def get_bars_info(signal_timestamp: datetime, interval: str, current_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Sinyal hakkında detaylı bar bilgisi döndürür.
    
    Args:
        signal_timestamp: Sinyalin geldiği zaman
        interval: Zaman aralığı
        current_time: Şu anki zaman
    
    Returns:
        Dict: Bar bilgileri içeren dictionary
    """
    if current_time is None:
        current_time = datetime.now()
    
    bars_passed = calculate_bars_since_signal(signal_timestamp, interval, current_time)
    interval_minutes = parse_interval_to_minutes(interval)
    
    # Bir sonraki bar'a kalan süre
    time_diff = current_time - signal_timestamp
    total_minutes = time_diff.total_seconds() / 60
    minutes_in_current_bar = total_minutes % interval_minutes
    minutes_to_next_bar = interval_minutes - minutes_in_current_bar
    
    return {
        'bars_passed': bars_passed,
        'interval': interval,
        'interval_minutes': interval_minutes,
        'minutes_to_next_bar': int(minutes_to_next_bar),
        'signal_age_hours': round(time_diff.total_seconds() / 3600, 2),
        'signal_age_days': round(time_diff.days + (time_diff.seconds / 86400), 2)
    }

def format_bar_info(bars_info: Dict[str, Any]) -> str:
    """
    Bar bilgisini okunabilir formatta döndürür.
    
    Args:
        bars_info: get_bars_info() fonksiyonundan dönen dictionary
    
    Returns:
        str: Formatlanmış bar bilgisi
    """
    bars = bars_info['bars_passed']
    interval = bars_info['interval']
    age_hours = bars_info['signal_age_hours']
    
    if bars == 0:
        return f"Yeni sinyal ({interval})"
    elif bars == 1:
        return f"1 bar geçti ({interval})"
    else:
        return f"{bars} bar geçti ({interval})"

# Test fonksiyonu
if __name__ == "__main__":
    # Test
    test_time = datetime.now() - timedelta(hours=5, minutes=30)
    
    print("Test - 5.5 saat önce gelen sinyal:")
    print(f"1h interval: {calculate_bars_since_signal(test_time, '1h')} bar")
    print(f"4h interval: {calculate_bars_since_signal(test_time, '4h')} bar")
    print(f"1d interval: {calculate_bars_since_signal(test_time, '1d')} bar")
    
    print("\nDetaylı bilgi (1h):")
    info = get_bars_info(test_time, '1h')
    print(info)
    print(format_bar_info(info))
