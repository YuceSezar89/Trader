"""
Merkezi Sinyal Motoru - Tüm sinyal fonksiyonları bu dosyada toplanmıştır.

Bu modül, çeşitli teknik göstergelere dayalı alım/satım sinyalleri üretmek için merkezi bir `SignalEngine` sınıfı sağlar.

Sinyal Türleri:
- RSI Crossover
- MA200 Crossover ve Seviye

Tüm fonksiyonlar asenkron olarak çalışır ve sağlam hata yönetimi içerir.
"""

import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional, Callable, Coroutine
from datetime import timedelta
from utils.logger import get_logger
from config import Config
from indicators.core import calculate_rsi, calculate_supertrend
from signals.signal_filter import SignalFilter

# Eşik değerlerini merkezi yapılandırmadan al
RSI_THRESHOLDS = Config.RSI_THRESHOLDS
MACD_THRESHOLDS = Config.MACD_THRESHOLDS

# Sütun adları için sabitler
COL_CLOSE = "close"
COL_LOW = "low"
COL_HIGH = "high"
COL_MA200 = "ma200"
COL_RSI_CHANGE = "rsi_change"
COL_MACD_CHANGE = "macd_change"
COL_ADX = "adx"
COL_PLUS_DI = "plus_di"
COL_MINUS_DI = "minus_di"


class SignalEngine:
    """
    Merkezi sinyal üretim motoru. Tüm sinyal türlerini tek bir sınıfta toplar ve yönetir.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._filter = SignalFilter()
        self._st_last_valid: dict[tuple, str] = {}

    def _validate_dataframe(
        self, df: pd.DataFrame, required_cols: List[str], min_len: int = 2
    ) -> bool:
        """DataFrame'i doğrulamak için merkezi yardımcı fonksiyon."""
        if df is None or df.empty:
            self.logger.warning("DataFrame boş veya None.")
            return False

        if len(df) < min_len:
            self.logger.warning(
                f"Sinyal hesaplaması için yeterli veri yok (en az {min_len} bar gerekli)."
            )
            return False

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Gerekli sütunlar eksik: {missing_cols}")
            return False

        # Son iki satırda NaN kontrolü
        for col in required_cols:
            if df[col].iloc[-min_len:].isna().any():
                self.logger.warning(
                    f"'{col}' sütununda son {min_len} barda NaN değerler var."
                )
                return False
        return True

    def _create_signal_output(
        self,
        df: pd.DataFrame,
        signal_type: str,
        indicators: str,
        strength: int = 1,
        pullback_level: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Standart bir sinyal çıktı sözlüğü oluşturur."""
        if signal_type not in ["Long", "Short"]:
            return []

        # Sinyal zamanı: KAPANIŞ zamanı (TV ile uyum)
        # 1) Son iki open_time farkından interval'i çıkar
        # 2) Bulunamazsa Config.KLINE_INTERVAL'den düşür
        ot_utc = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)
        inferred_delta: Optional[timedelta] = None
        try:
            if len(df) >= 2 and not pd.isna(df["open_time"].iloc[-2]):
                prev_ot_utc = pd.to_datetime(df["open_time"].iloc[-2], unit="ms", utc=True)
                d = ot_utc - prev_ot_utc
                if isinstance(d, pd.Timedelta):
                    d = d.to_pytimedelta()
                if isinstance(d, timedelta) and d.total_seconds() > 0:
                    inferred_delta = d
        except Exception:
            inferred_delta = None

        if inferred_delta is None:
            # Basit parser: '15m', '1h', '4h', '1d' gibi değerleri destekle
            iv = getattr(Config, "KLINE_INTERVAL", "15m")
            try:
                unit = iv[-1].lower()
                val = int(iv[:-1])
                if unit == 'm':
                    inferred_delta = timedelta(minutes=val)
                elif unit == 'h':
                    inferred_delta = timedelta(hours=val)
                elif unit == 'd':
                    inferred_delta = timedelta(days=val)
                else:
                    inferred_delta = timedelta(minutes=15)
            except Exception:
                inferred_delta = timedelta(minutes=15)

        close_utc = ot_utc + inferred_delta
        # Zaman damgası üretimi ve teşhis amaçlı ayrıntılı log
        tz_name = getattr(Config, "TIMEZONE", "Europe/Istanbul")
        try:
            now_local = pd.Timestamp.now(tz="UTC").tz_convert(tz_name)
        except Exception:
            # Beklenmedik bir timezone adı durumunda güvenli geri dönüş
            tz_name = "Europe/Istanbul"
            now_local = pd.Timestamp.now(tz="UTC").tz_convert(tz_name)

        close_local = close_utc.tz_convert(tz_name)
        self.logger.debug(
            f"Signal TS Debug | ot_utc={ot_utc} | inferred_delta={inferred_delta} | "
            f"close_utc={close_utc} | close_local={close_local} | now_local={now_local} | tz={tz_name}"
        )
        current_time = close_local.replace(tzinfo=None).to_pydatetime()  # timezone-naive datetime nesnesi

        # ADX değerlerini güvenli bir şekilde al
        adx = (
            df[COL_ADX].iloc[-1]
            if COL_ADX in df.columns and not pd.isna(df[COL_ADX].iloc[-1])
            else None
        )
        plus_di = (
            df[COL_PLUS_DI].iloc[-1]
            if COL_PLUS_DI in df.columns and not pd.isna(df[COL_PLUS_DI].iloc[-1])
            else None
        )
        minus_di = (
            df[COL_MINUS_DI].iloc[-1]
            if COL_MINUS_DI in df.columns and not pd.isna(df[COL_MINUS_DI].iloc[-1])
            else None
        )

        # Momentum değerini güvenli bir şekilde al
        momentum = (
            df["momentum"].iloc[-1]
            if "momentum" in df.columns and not pd.isna(df["momentum"].iloc[-1])
            else None
        )

        # RSI, MACD ve ATR değerlerini güvenli bir şekilde al
        fast_rsi_col = f"rsi_{Config.RSI_FAST_WINDOW}"
        rsi = (
            df[fast_rsi_col].iloc[-1]
            if fast_rsi_col in df.columns and not pd.isna(df[fast_rsi_col].iloc[-1])
            else None
        )
        macd = (
            df["macd"].iloc[-1]
            if "macd" in df.columns and not pd.isna(df["macd"].iloc[-1])
            else None
        )
        atr = (
            df["atr"].iloc[-1]
            if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1])
            else None
        )

        effective_pullback = pullback_level

        signal_data = {
            "signal_type": signal_type,
            "timestamp": current_time,
            "price": df[COL_CLOSE].iloc[-1],
            "pullback_level": effective_pullback,
            "strength": strength,
            "indicators": indicators,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "momentum": momentum,
            "rsi": rsi,
            "macd": macd,
            "atr": atr,
        }

        return [signal_data]

    async def rsi_crossover_signal(
        self, df: pd.DataFrame, symbol: str = "", interval: str = ""
    ) -> List[Dict[str, Any]]:
        fast_len = Config.RSI_FAST_WINDOW
        slow_len = Config.RSI_SLOW_WINDOW
        """RSI crossover sinyalini hesaplar ve standart formatta döndürür.
        TradingView ile uyum için RSI hesaplaması indicators.core.calculate_rsi ile yapılır
        ve kesişim sadece kapanmış barlarda kontrol edilir.
        """
        fast_col, slow_col = f"rsi_{fast_len}", f"rsi_{slow_len}"

        # Önce temel kolonları doğrula; RSI kolonlarını eksikse burada üreteceğiz
        base_required = [COL_CLOSE, COL_LOW, COL_HIGH, "open_time"]
        if not self._validate_dataframe(df, base_required, min_len=3):
            return []

        # RSI kolonlarını eksikse hesapla (Wilder/TV uyumlu)
        try:
            if fast_col not in df.columns:
                df[fast_col] = calculate_rsi(df, period=fast_len, price_col=COL_CLOSE)
            if slow_col not in df.columns:
                df[slow_col] = calculate_rsi(df, period=slow_len, price_col=COL_CLOSE)
        except Exception as e:
            self.logger.warning(f"RSI hesaplama hatası: {e}")
            return []

        # Kapanmış barlarda kontrol: [-3] -> önceki kapalı, [-2] -> son kapalı
        rsi_fast_prev, rsi_slow_prev = df[fast_col].iloc[-3], df[slow_col].iloc[-3]
        rsi_fast_now, rsi_slow_now = df[fast_col].iloc[-2], df[slow_col].iloc[-2]

        # NaN koruması
        if any(pd.isna([rsi_fast_prev, rsi_slow_prev, rsi_fast_now, rsi_slow_now])):
            return []

        signal_type = ""
        if rsi_fast_prev < rsi_slow_prev and rsi_fast_now > rsi_slow_now:
            signal_type = "Long"
        elif rsi_fast_prev > rsi_slow_prev and rsi_fast_now < rsi_slow_now:
            signal_type = "Short"

        if signal_type:
            indicator_key = f"RSI_Cross({fast_len},{slow_len})"
            high = float(df[COL_HIGH].iloc[-2])
            low  = float(df[COL_LOW].iloc[-2])
            if symbol and interval and not self._filter.check(
                signal_type, high, low, symbol, interval, indicator_key
            ):
                self.logger.info(
                    f"[{symbol}] {interval} {signal_type} filtreden geçemedi ({indicator_key})"
                )
                return []
            return self._create_signal_output(
                df=df.iloc[:-1],
                signal_type=signal_type,
                indicators=indicator_key,
                strength=1,
            )
        return []

    async def ma200_crossover_signal(
        self, df: pd.DataFrame, symbol: str = "", interval: str = ""
    ) -> List[Dict[str, Any]]:
        """MA200 ve fiyat kesişimini kontrol eder (TradingView basit crossover mantığı).

        TradingView ta.crossover/crossunder mantığı:
        - Sadece close fiyatlarının kesişimine bakar
        - Önceki close bir tarafta, şimdiki close diğer tarafta
        """
        required_cols = [COL_CLOSE, COL_MA200, "open_time"]
        if len(df) < 3:
            return []
        df_closed = df.iloc[:-1]
        if not self._validate_dataframe(df_closed, required_cols):
            return []

        prev = df_closed.iloc[-2]
        curr = df_closed.iloc[-1]

        signal_type = ""
        if prev[COL_CLOSE] < prev[COL_MA200] and curr[COL_CLOSE] > curr[COL_MA200]:
            signal_type = "Long"
        elif prev[COL_CLOSE] > prev[COL_MA200] and curr[COL_CLOSE] < curr[COL_MA200]:
            signal_type = "Short"

        if signal_type:
            indicator_key = "MA200_Cross"
            high = float(curr[COL_HIGH])
            low  = float(curr[COL_LOW])
            if symbol and interval and not self._filter.check(
                signal_type, high, low, symbol, interval, indicator_key
            ):
                self.logger.info(
                    f"[{symbol}] {interval} {signal_type} filtreden geçemedi ({indicator_key})"
                )
                return []
            return self._create_signal_output(
                df=df_closed, signal_type=signal_type, indicators=indicator_key, strength=1
            )
        return []

    async def supertrend_signal(
        self, df: pd.DataFrame, symbol: str = "", interval: str = ""
    ) -> List[Dict[str, Any]]:
        """SuperTrend(10,3.0) sinyali — ChartPrime Filtered Signals uyumlu.

        Kurallar:
        - Son kapanmış barda direction değişimi → raw sinyal
        - SignalFilter'dan geçmeli (high/low filtresi)
        - Son geçerli ST sinyaliyle aynı yönde ise üretilmez (trend devam)
        """
        required = ["open_time", COL_HIGH, COL_LOW, COL_CLOSE]
        if not self._validate_dataframe(df, required, min_len=3):
            return []

        df_closed = df.iloc[:-1]

        # st_direction sütunu yoksa hesapla
        if "st_direction" not in df_closed.columns or df_closed["st_direction"].isna().all():
            _, direction, _, _ = calculate_supertrend(df_closed)
            df_closed = df_closed.copy()
            df_closed["st_direction"] = direction

        dir_now  = df_closed["st_direction"].iloc[-1]
        dir_prev = df_closed["st_direction"].iloc[-2]

        if pd.isna(dir_now) or pd.isna(dir_prev):
            return []

        long_signal  = (dir_now == -1 and dir_prev != -1)
        short_signal = (dir_now ==  1 and dir_prev !=  1)

        if not long_signal and not short_signal:
            return []

        signal_type   = "Long" if long_signal else "Short"
        indicator_key = "Supertrend(10,3.0)"
        high = float(df_closed[COL_HIGH].iloc[-1])
        low  = float(df_closed[COL_LOW].iloc[-1])

        if symbol and interval:
            if not self._filter.check(signal_type, high, low, symbol, interval, indicator_key):
                self.logger.info(
                    f"[{symbol}] {interval} {signal_type} filtreden geçemedi ({indicator_key})"
                )
                return []

            key = (symbol, interval)
            if self._st_last_valid.get(key) == signal_type:
                self.logger.info(
                    f"[{symbol}] {interval} {signal_type} tekrar — trend devam, atlandı ({indicator_key})"
                )
                return []
            self._st_last_valid[key] = signal_type

        return self._create_signal_output(
            df=df_closed,
            signal_type=signal_type,
            indicators=indicator_key,
            strength=1,
        )

    async def ha_crossover_signal(
        self, df: pd.DataFrame, symbol: str = "", interval: str = ""
    ) -> List[Dict[str, Any]]:
        """Heiken Ashi crossover sinyali — kapanmış son barda flip tespiti."""
        required = ["open_time", "ha_open", "ha_close", COL_HIGH, COL_LOW]
        if len(df) < 3:
            return []
        df_closed = df.iloc[:-1]
        if not self._validate_dataframe(df_closed, required, min_len=2):
            return []

        curr = df_closed.iloc[-1]
        prev = df_closed.iloc[-2]

        curr_bull = float(curr["ha_close"]) > float(curr["ha_open"])
        prev_bull = float(prev["ha_close"]) > float(prev["ha_open"])

        signal_type = ""
        if curr_bull and not prev_bull:
            signal_type = "Long"
        elif not curr_bull and prev_bull:
            signal_type = "Short"

        if not signal_type:
            return []

        indicator_key = "HA_Cross"
        high = float(curr[COL_HIGH])
        low  = float(curr[COL_LOW])

        if symbol and interval and not self._filter.check(
            signal_type, high, low, symbol, interval, indicator_key
        ):
            self.logger.info(
                f"[{symbol}] {interval} {signal_type} filtreden geçemedi ({indicator_key})"
            )
            return []

        return self._create_signal_output(
            df=df_closed, signal_type=signal_type, indicators=indicator_key, strength=1
        )

    async def calculate_all_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        interval: str = "",
        signal_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Tüm sinyal türlerini hesaplar ve birleştirilmiş sonuç döndürür.
        Daha sağlam ve genişletilebilir görev yönetimi kullanır.
        """
        signal_methods: Dict[
            str, Callable[[pd.DataFrame], Coroutine[Any, Any, Any]]
        ] = {
            "rsi_crossover": self.rsi_crossover_signal,
            "ma200_crossover": self.ma200_crossover_signal,
            "supertrend": self.supertrend_signal,
            "ha_crossover": self.ha_crossover_signal,
        }

        active_signals: List[str]
        if signal_types is None:
            active_signals = list(signal_methods.keys())
        else:
            active_signals = [st for st in signal_types if st in signal_methods]

        tasks: Dict[str, asyncio.Task] = {
            name: asyncio.create_task(method(df, symbol, interval))
            for name, method in signal_methods.items()
            if name in active_signals
        }

        if not tasks:
            return {}

        try:
            # Görevleri paralel olarak çalıştır
            await asyncio.gather(*tasks.values())

            # Sonuçları topla
            results = {name: task.result() for name, task in tasks.items()}
            self.logger.info(f"Tüm sinyaller hesaplandı: {list(results.keys())}")
            return results

        except Exception as e:
            self.logger.error(
                f"Sinyal hesaplama sırasında genel hata: {e}", exc_info=True
            )
            # Hatalı görevleri belirle ve logla
            for name, task in tasks.items():
                if task.done() and task.exception():
                    self.logger.error(f"'{name}' sinyalinde hata: {task.exception()}")
            return {name: "ERROR" for name in tasks}


# --- Global Signal Engine Instance ---
# Kodun diğer kısımlarından bu tekil örnek kullanılmalıdır.
signal_engine = SignalEngine()

# --- Dışa Aktarılacaklar ---
# Geriye dönük uyumluluk fonksiyonları kaldırıldı.
# Sadece ana engine ve sınıf dışa aktarılıyor.
__all__ = ["SignalEngine", "signal_engine"]
