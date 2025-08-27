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
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable, Coroutine
from datetime import timedelta
import numpy as np
from utils.logger import get_logger
from utils.exceptions import ValidationError, CalculationError
from config import Config
from signals.c20mx_core import compute_features as c20mx_compute_features, detect_signals as c20mx_detect_signals
from indicators.core import calculate_rsi

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
        current_time = close_local.strftime("%Y-%m-%d %H:%M:%S")

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

        # Pullback level:
        # - C20MX dışındaki sinyallerde pullback_level verilmemişse None bırak
        # - C20MX sinyallerinde verilmemişse bar low/high fallback uygula
        effective_pullback = pullback_level
        if effective_pullback is None and isinstance(indicators, str) and indicators.startswith("C20MX:"):
            effective_pullback = (
                df[COL_LOW].iloc[-1] if signal_type == "Long" else df[COL_HIGH].iloc[-1]
            )

        signal_data = {
            "signal_type": signal_type,
            "signal_time": current_time,
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

    async def rsi_crossover_signal(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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
            # Çıktıyı kapanmış bara göre üretmek için DF'i son açık barı atarak gönder
            return self._create_signal_output(
                df=df.iloc[:-1],
                signal_type=signal_type,
                indicators=f"RSI_Cross({fast_len},{slow_len})",
                strength=1,
            )
        return []

    async def ma200_crossover_signal(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """MA200 ve fiyat kesişimini kontrol eder ve standart formatta döndürür."""
        required_cols = [COL_CLOSE, COL_MA200, COL_LOW, COL_HIGH, "open_time"]
        # Kapalı mum üzerinde çalış: son açık mumu at
        if len(df) < 3:
            return []
        df_closed = df.iloc[:-1]
        if not self._validate_dataframe(df_closed, required_cols):
            return []

        price, ma = df_closed[COL_CLOSE], df_closed[COL_MA200]
        signal_type = ""
        if price.iloc[-2] < ma.iloc[-2] and price.iloc[-1] > ma.iloc[-1]:
            signal_type = "Long"
        elif price.iloc[-2] > ma.iloc[-2] and price.iloc[-1] < ma.iloc[-1]:
            signal_type = "Short"

        if signal_type:
            return self._create_signal_output(
                df=df_closed, signal_type=signal_type, indicators="MA200_Cross", strength=1
            )
        return []

    

    async def c20mx_signal(self, df: pd.DataFrame, interval: Optional[str] = None) -> List[Dict[str, Any]]:
        """C20MX (RSI%Change + MACD-türevi) sinyallerini üretir ve pullback seviyesini yönüne göre yazar.
        TradingView Pine ile uyumlu olacak şekilde kapalı mumda (i=-2) değerlendirme yapar.
        """
        base_required = ["open", "high", "low", "close", "open_time"]
        if not self._validate_dataframe(df, base_required, min_len=3):
            return []

        try:
            feat = c20mx_compute_features(df[["open", "high", "low", "close"]].copy())
        except Exception as e:
            self.logger.error(f"C20MX feature hesaplama hatası: {e}")
            return []

        i = -2
        try:
            codes = c20mx_detect_signals(feat, i=i, interval=interval)
        except Exception as e:
            self.logger.error(f"C20MX sinyal tespiti hatası: {e}")
            return []

        long_cnt = sum(1 for c in codes if c.endswith("L"))
        short_cnt = sum(1 for c in codes if c.endswith("S"))
        if long_cnt == 0 and short_cnt == 0:
            return []

        signal_type = "Long" if long_cnt >= short_cnt else "Short"

        weights = {"C10": 1, "C20": 2, "L20": 2, "M2": 2, "M3": 3, "M4": 4, "M5": 5}
        def side_code_weight(code: str) -> int:
            for k, w in weights.items():
                if code.startswith(k):
                    return w
            return 1
        strength = sum(
            side_code_weight(c)
            for c in codes
            if (c.endswith("L") and signal_type == "Long") or (c.endswith("S") and signal_type == "Short")
        )
        strength = max(1, int(strength))

        pull_lvl: Optional[float] = None
        try:
            if signal_type == "Long" and "long_pullback_level" in feat.columns and pd.notna(feat["long_pullback_level"].iloc[i]):
                pull_lvl = float(feat["long_pullback_level"].iloc[i])
            elif signal_type == "Short" and "short_pullback_level" in feat.columns and pd.notna(feat["short_pullback_level"].iloc[i]):
                pull_lvl = float(feat["short_pullback_level"].iloc[i])
        except Exception:
            pull_lvl = None

        out_df = df.iloc[:-1].copy()
        ind_str = "C20MX:" + "|".join([c for c in codes if not c.startswith("[")])
        if interval:
            ind_str += f"[{interval}]"

        return self._create_signal_output(
            df=out_df,
            signal_type=signal_type,
            indicators=ind_str,
            strength=strength,
            pullback_level=pull_lvl,
        )


    async def calculate_all_signals(
        self, df: pd.DataFrame, signal_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tüm sinyal türlerini hesaplar ve birleştirilmiş sonuç döndürür.
        Daha sağlam ve genişletilebilir görev yönetimi kullanır.
        """
        # `method(df)` bir Coroutine döndürdüğü için Callable tanımını güncelledik.
        signal_methods: Dict[
            str, Callable[[pd.DataFrame], Coroutine[Any, Any, Any]]
        ] = {
            "rsi_crossover": self.rsi_crossover_signal,
            "ma200_crossover": self.ma200_crossover_signal,
            "c20mx": self.c20mx_signal,
        }

        active_signals: List[str]
        if signal_types is None:
            # Tür uyumluluğu için dict_keys'i listeye çeviriyoruz.
            active_signals = list(signal_methods.keys())
        else:
            active_signals = [st for st in signal_types if st in signal_methods]

        # `tasks` için tür tanımı eklendi.
        tasks: Dict[str, asyncio.Task] = {
            name: asyncio.create_task(method(df))
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
