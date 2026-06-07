"""
MTF Backfill Sistemi
Geçmiş 1m verilerinden 5m, 15m, 1h MTF sinyalleri üretir.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
import numpy as np
import pandas as pd

# Local imports
import indicators.core as indicators
from signals.vpm_calculator import VPMCalculator

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    """Backfill işlem istatistikleri"""

    processed_symbols: int = 0
    processed_bars: int = 0
    generated_signals: int = 0
    timeframe_signals: Optional[Dict[str, int]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        if self.timeframe_signals is None:
            self.timeframe_signals = {"5m": 0, "15m": 0, "1h": 0}


class MTFBackfillEngine:
    """MTF Backfill Ana Motoru"""

    def __init__(self):
        self.db_url = os.getenv(
            "DATABASE_URL", "postgresql://yusuf@localhost:6432/trader_panel"
        )
        self.supported_timeframes = ["5m", "15m", "1h"]
        self.batch_size = 1000  # Toplu insert için
        self.stats = BackfillStats()

        # Engines - fonksiyon tabanlı

    async def run_backfill(
        self, symbols: Optional[List[str]] = None, days_back: int = 30
    ) -> BackfillStats:
        """Ana backfill işlemini çalıştır"""
        self.stats.start_time = datetime.now()
        logger.info(f"🚀 MTF Backfill başlatılıyor - {days_back} gün geriye")

        try:
            conn = await asyncpg.connect(self.db_url)

            # Sembolleri al
            if not symbols:
                symbols = await self._get_available_symbols(conn)

            logger.info(f"📊 {len(symbols)} sembol işlenecek")

            # Her sembol için backfill yap
            for i, symbol in enumerate(symbols):
                logger.info(f"🔄 [{i+1}/{len(symbols)}] {symbol} işleniyor...")

                try:
                    await self._process_symbol_backfill(conn, symbol, days_back)
                    self.stats.processed_symbols += 1
                except Exception as e:
                    logger.error(f"❌ {symbol} işlenirken hata: {e}")
                    continue

            await conn.close()

        except Exception as e:
            logger.error(f"❌ Backfill genel hatası: {e}")
            raise

        self.stats.end_time = datetime.now()
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()

        logger.info("✅ Backfill tamamlandı!")
        logger.info(f"   Süre: {duration:.1f} saniye")
        logger.info(f"   İşlenen sembol: {self.stats.processed_symbols}")
        logger.info(f"   İşlenen bar: {self.stats.processed_bars}")
        logger.info(f"   Üretilen sinyal: {self.stats.generated_signals}")

        for tf, count in self.stats.timeframe_signals.items():
            logger.info(f"   {tf} sinyaller: {count}")

        return self.stats

    async def _get_available_symbols(self, conn) -> List[str]:
        """Mevcut sembolleri al"""
        result = await conn.fetch(
            """
            SELECT DISTINCT symbol 
            FROM price_data 
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY symbol
        """
        )
        return [row["symbol"] for row in result]

    async def _process_symbol_backfill(self, conn, symbol: str, days_back: int):
        """Tek sembol için backfill işlemi"""

        # 1m verilerini al
        raw_data = await self._get_symbol_1m_data(conn, symbol, days_back)

        if len(raw_data) < 60:  # En az 1 saatlik veri gerekli
            logger.warning(f"⚠️ {symbol}: Yetersiz veri ({len(raw_data)} bar)")
            return

        logger.info(f"   📊 {len(raw_data)} adet 1m bar alındı")
        self.stats.processed_bars += len(raw_data)

        # Her timeframe için işle
        for timeframe in self.supported_timeframes:
            try:
                signals = await self._generate_mtf_signals(raw_data, symbol, timeframe)

                if signals:
                    await self._bulk_insert_signals(conn, signals)
                    self.stats.timeframe_signals[timeframe] += len(signals)
                    self.stats.generated_signals += len(signals)
                    logger.info(f"   ✅ {timeframe}: {len(signals)} sinyal üretildi")

            except Exception as e:
                logger.error(f"❌ {symbol} {timeframe} hatası: {e}")

    async def _get_symbol_1m_data(
        self, conn, symbol: str, days_back: int
    ) -> pd.DataFrame:
        """Sembol için 1m verilerini al"""

        result = await conn.fetch(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = $1
            AND timestamp >= NOW() - ($2 * INTERVAL '1 day')
            ORDER BY timestamp ASC
            """,
            symbol,
            days_back,
        )

        if not result:
            return pd.DataFrame()

        # DataFrame'e çevir
        df = pd.DataFrame([dict(row) for row in result])

        # Timestamp'i open_time'a çevir (milisaniye)
        df["open_time"] = df["timestamp"].apply(lambda x: int(x.timestamp() * 1000))

        return df[["open_time", "open", "high", "low", "close", "volume"]]

    async def _generate_mtf_signals(
        self, df_1m: pd.DataFrame, symbol: str, timeframe: str
    ) -> List[Dict]:
        """MTF sinyalleri üret"""

        # 1m → MTF agregasyon
        df_mtf = self._aggregate_to_timeframe(df_1m, timeframe)

        if len(df_mtf) < 200:  # Yeterli veri yok
            return []

        # İndikatörleri hesapla
        df_mtf = self._calculate_indicators(df_mtf)

        # Finansal metrikleri hesapla (basit versiyon)
        try:
            df_mtf = self._calculate_simple_financial_metrics(df_mtf)
            logger.info("   📊 Finansal metrikler hesaplandı")

        except Exception as e:
            logger.warning(f"   ⚠️ Finansal metrik hatası: {e}")
            # Varsayılan değerler ekle
            for col in [
                "alpha",
                "beta",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "normalized_composite",
                "zscore_ratio_percent",
            ]:
                if col not in df_mtf.columns:
                    df_mtf[col] = 0.0

        # Sinyalleri üret
        signals = []

        for i in range(200, len(df_mtf)):  # İlk 200 bar'ı atla (indikatör warmup)
            row = df_mtf.iloc[i]
            prev_row = df_mtf.iloc[i - 1] if i > 0 else None

            # Sinyal koşullarını kontrol et
            signal_data = self._check_signal_conditions(
                row, symbol, timeframe, prev_row
            )

            if signal_data:
                signals.append(signal_data)

        return signals

    def _aggregate_to_timeframe(
        self, df_1m: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """1m verilerini belirtilen timeframe'e agregasyon yap (TradingView uyumlu)

        Pandas resample kullanarak timestamp'leri doğru align eder:
        - 5m: 00:00, 00:05, 00:10, 00:15, 00:20...
        - 15m: 00:00, 00:15, 00:30, 00:45...
        - 1h: 00:00, 01:00, 02:00...
        """

        # DataFrame'i kopyala ve timestamp'i datetime'a çevir
        df = df_1m.copy()
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Timeframe mapping (pandas resample formatı)
        tf_map = {"5m": "5min", "15m": "15min", "1h": "1h"}
        resample_rule = tf_map.get(timeframe, "5min")

        # OHLCV agregasyonu
        agg_df = (
            df.resample(resample_rule, label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # open_time'ı timestamp'den geri oluştur (milliseconds)
        agg_df["open_time"] = (agg_df.index.astype(np.int64) // 10**6).astype(np.int64)

        # Index'i resetle ve timestamp kolonunu kaldır
        agg_df.reset_index(drop=True, inplace=True)

        return agg_df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temel indikatörleri hesapla"""

        try:
            # RSI
            df["rsi_14"] = indicators.calculate_rsi(df, period=14)

            # MACD
            macd, signal, histogram = indicators.calculate_macd(df)
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = histogram

            # MA200
            df["ma200"] = indicators.calculate_sma(df, period=200)

            # ATR
            df["atr"] = indicators.calculate_atr(df, period=14)

            # ADX
            adx, plus_di, minus_di = indicators.calculate_adx(df)
            df["adx"] = adx
            df["plus_di"] = plus_di
            df["minus_di"] = minus_di

            # Volume SMA
            df["volume_sma_20"] = indicators.calculate_sma(
                df, period=20, price_col="volume"
            )

            # Momentum (basit fiyat değişimi)
            df["momentum"] = df["close"].pct_change(5)  # 5 periyot momentum

        except Exception as e:
            logger.warning(f"İndikatör hesaplama hatası: {e}")
            # Hata durumunda varsayılan değerler
            for col in [
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_hist",
                "ma200",
                "atr",
                "adx",
                "plus_di",
                "minus_di",
                "volume_sma_20",
                "momentum",
            ]:
                if col not in df.columns:
                    df[col] = 0

        return df

    def _calculate_simple_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basit finansal metrikleri hesapla"""

        try:
            # Returns hesapla
            df["returns"] = df["close"].pct_change()

            # Rolling volatility (20 period)
            df["volatility"] = df["returns"].rolling(20).std()

            # Alpha (basit momentum)
            df["alpha"] = df["returns"].rolling(20).mean() * 252  # Yıllık

            # Beta (market correlation - kendisiyle korelasyon = 1)
            df["beta"] = (
                1.0
                + (df["returns"].rolling(20).std() - df["returns"].std())
                / df["returns"].std()
            )

            # Sharpe ratio (basit)
            risk_free_rate = 0.02  # %2 risk-free rate
            excess_returns = df["returns"] - (risk_free_rate / 252)
            df["sharpe_ratio"] = excess_returns.rolling(20).mean() / df["volatility"]

            # Sortino ratio (downside deviation)
            downside_returns = excess_returns[excess_returns < 0]
            downside_vol = downside_returns.rolling(20).std()
            df["sortino_ratio"] = excess_returns.rolling(20).mean() / downside_vol

            # Calmar ratio (return/max drawdown)
            rolling_max = df["close"].rolling(20).max()
            drawdown = (df["close"] - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(20).min().abs()
            df["calmar_ratio"] = (df["alpha"] / 252) / max_drawdown

            # Normalized composite (0-1 arası)
            df["normalized_composite"] = (
                (df["alpha"].rank(pct=True) * 0.3)
                + (df["sharpe_ratio"].rank(pct=True) * 0.4)
                + ((1 - df["volatility"].rank(pct=True)) * 0.3)
            )

            # Z-Score ratio percent
            returns_mean = df["returns"].rolling(50).mean()
            returns_std = df["returns"].rolling(50).std()
            df["zscore_ratio_percent"] = (
                (df["returns"] - returns_mean) / returns_std
            ) * 100

            # Omega ratio (upside/downside ratio)
            upside_returns = df["returns"][df["returns"] > 0].rolling(20).sum()
            downside_returns = df["returns"][df["returns"] < 0].rolling(20).sum().abs()
            df["omega_ratio"] = upside_returns / (
                downside_returns + 0.001
            )  # Avoid division by zero

            # Treynor ratio (excess return / beta)
            df["treynor_ratio"] = excess_returns.rolling(20).mean() / (
                df["beta"] + 0.001
            )

            # Information ratio (active return / tracking error)
            benchmark_return = df["returns"].rolling(50).mean()  # Market benchmark
            active_return = df["returns"] - benchmark_return
            tracking_error = active_return.rolling(20).std()
            df["information_ratio"] = active_return.rolling(20).mean() / (
                tracking_error + 0.001
            )

            # Eksik kolonları ekle
            # MTF Score (composite score)
            df["mtf_score"] = (
                (df["sharpe_ratio"].rank(pct=True) * 0.25)
                + (df["alpha"].rank(pct=True) * 0.25)
                + (df["information_ratio"].rank(pct=True) * 0.25)
                + (df["omega_ratio"].rank(pct=True) * 0.25)
            ) * 100

            # Normalized price change
            df["normalized_price_change"] = df["returns"] * 100  # Percentage change

            # Pullback level (distance from recent high)
            rolling_high = df["close"].rolling(20).max()
            df["pullback_level"] = (df["close"] - rolling_high) / rolling_high * 100

            # Scaled avg normalized
            df["scaled_avg_normalized"] = (
                df["returns"].rolling(20).mean() - df["returns"].mean()
            ) / df["returns"].std()

            # VPM MTF Score (timeframe weighted VPM)
            base_vpm = (
                (df["volume"] / df["volume_sma_20"])
                * abs(df["momentum"])
                * (1 - abs(df["rsi_14"] - 50) / 50)
            )
            df["vpms_mtf_score"] = base_vpm * 100  # Scale to 0-100

            # NaN değerleri 0 ile doldur
            financial_cols = [
                "alpha",
                "beta",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "normalized_composite",
                "zscore_ratio_percent",
                "omega_ratio",
                "treynor_ratio",
                "information_ratio",
                "mtf_score",
                "normalized_price_change",
                "pullback_level",
                "scaled_avg_normalized",
                "vpms_mtf_score",
            ]
            for col in financial_cols:
                df[col] = df[col].fillna(0)

        except Exception as e:
            logger.error(f"Basit finansal metrik hatası: {e}")
            # Hata durumunda 0.0 kullan
            for col in [
                "alpha",
                "beta",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "normalized_composite",
                "zscore_ratio_percent",
            ]:
                df[col] = 0.0

        return df

    def _check_signal_conditions(
        self,
        row: pd.Series,
        symbol: str,
        timeframe: str,
        prev_row: Optional[pd.Series] = None,
    ) -> Optional[Dict]:
        """Sinyal koşullarını kontrol et"""

        # Temel sinyal koşulları
        signals = []

        # RSI Cross sinyali
        rsi_signal = self._check_rsi_cross_signal(row)
        if rsi_signal:
            signals.append(f"RSI_Cross({rsi_signal})")

        # MA200 Cross sinyali
        if self._check_ma200_cross_signal(row, prev_row):
            signals.append("MA200_Cross")

        if not signals:
            return None

        # Sinyal türünü belirle (AL/SAT)
        signal_type = self._determine_signal_type(row)

        # VPM skoru hesapla
        vpm_score = self._calculate_vpm_score(row, timeframe)

        # Timestamp'i datetime'a çevir (UTC)
        from datetime import timezone

        timestamp = datetime.fromtimestamp(
            row["open_time"] / 1000, tz=timezone.utc
        ).replace(tzinfo=None)

        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "signal_type": signal_type,
            "interval": timeframe,
            "price": float(row["close"]),
            "indicators": "|".join(signals),
            "vpms_score": vpm_score,
            "strength": min(len(signals) * 20, 100),  # Çoklu sinyal = güçlü
            "rsi": float(row.get("rsi_14", 50)),
            "macd": float(row.get("macd", 0)),
            "momentum": float(row.get("momentum", 0)),
            "atr": float(row.get("atr", 0)),
            "adx": float(row.get("adx", 0)),
            "plus_di": float(row.get("plus_di", 0)),
            "minus_di": float(row.get("minus_di", 0)),
            # Finansal metrikler
            "alpha": float(row.get("alpha", 0)),
            "beta": float(row.get("beta", 0)),
            "sharpe_ratio": float(row.get("sharpe_ratio", 0)),
            "sortino_ratio": float(row.get("sortino_ratio", 0)),
            "calmar_ratio": float(row.get("calmar_ratio", 0)),
            "omega_ratio": float(row.get("omega_ratio", 0)),
            "treynor_ratio": float(row.get("treynor_ratio", 0)),
            "information_ratio": float(row.get("information_ratio", 0)),
            "mtf_score": float(row.get("mtf_score", 0)),
            "normalized_price_change": float(row.get("normalized_price_change", 0)),
            "pullback_level": float(row.get("pullback_level", 0)),
            "scaled_avg_normalized": float(row.get("scaled_avg_normalized", 0)),
            "vpms_mtf_score": float(row.get("vpms_mtf_score", 0)),
            "normalized_composite": float(row.get("normalized_composite", 0)),
            "zscore_ratio_percent": float(row.get("zscore_ratio_percent", 0)),
            "status": "active",
        }

    def _check_rsi_cross_signal(self, row: pd.Series) -> Optional[str]:
        """RSI Cross sinyal koşulunu kontrol et"""
        rsi = row.get("rsi_14", 50)

        if rsi < 30:
            return "9,24"  # Oversold
        elif rsi > 70:
            return "9,24"  # Overbought

        return None

    def _check_ma200_cross_signal(
        self, row: pd.Series, prev_row: Optional[pd.Series] = None
    ) -> bool:
        """MA200 Cross sinyal koşulunu kontrol et (TradingView basit crossover mantığı)

        TradingView ta.crossover/crossunder mantığı:
        - Sadece close fiyatlarının kesişimine bakar
        - Önceki close bir tarafta, şimdiki close diğer tarafta
        """
        if prev_row is None:
            return False

        close = row.get("close", 0)
        ma200 = row.get("ma200", 0)

        prev_close = prev_row.get("close", 0)
        prev_ma200 = prev_row.get("ma200", 0)

        # MA200 değeri geçersizse atla
        if ma200 <= 0 or prev_ma200 <= 0:
            return False

        # Basit crossover kontrolü (TradingView ta.crossover/crossunder)
        # Yukarı kesişim: önceki close altında, şimdiki close üstte
        # Aşağı kesişim: önceki close üstte, şimdiki close altında
        if (prev_close < prev_ma200 and close > ma200) or (
            prev_close > prev_ma200 and close < ma200
        ):
            return True

        return False

    def _determine_signal_type(self, row: pd.Series) -> str:
        """Sinyal türünü belirle (AL/SAT)"""
        momentum = row.get("momentum", 0)
        rsi = row.get("rsi_14", 50)

        if momentum > 0 and rsi < 70:
            return "AL"
        elif momentum < 0 and rsi > 30:
            return "SAT"
        else:
            return "AL"  # Varsayılan

    def _calculate_vpm_score(self, row: pd.Series, timeframe: str) -> float:
        """VPM skoru hesapla (standardize VPMCalculator kullan)"""

        try:
            # Sinyal türünü belirle
            signal_type = (
                "Long" if row.get("close", 0) > row.get("ma200", 0) else "Short"
            )

            # VPMCalculator ile hesapla
            vpm_score = VPMCalculator.calculate(
                volume=float(row.get("volume", 0)),
                volume_sma=float(row.get("volume_sma_20", 1)),
                price_change_pct=float(row.get("momentum", 0) * 100),  # momentum -> %
                rsi_delta=float(row.get("rsi_14", 50) - 50),  # RSI delta from neutral
                interval=timeframe,
                signal_type=signal_type,
            )

            return vpm_score

        except Exception as e:
            logger.warning(f"VPM hesaplama hatası: {e}")
            return 0.0

    async def _bulk_insert_signals(self, conn, signals: List[Dict]):
        """Sinyalleri toplu olarak database'e ekle"""

        if not signals:
            return

        # Batch'lere böl
        for i in range(0, len(signals), self.batch_size):
            batch = signals[i : i + self.batch_size]

            # SQL hazırla
            values = []
            for signal in batch:
                values.append(
                    (
                        signal["symbol"],
                        signal["timestamp"],
                        signal["signal_type"],
                        signal["interval"],
                        signal["price"],
                        signal["indicators"],
                        signal["vpms_score"],
                        signal["strength"],
                        signal["rsi"],
                        signal["macd"],
                        signal["momentum"],
                        signal["atr"],
                        signal["adx"],
                        signal["plus_di"],
                        signal["minus_di"],
                        signal["alpha"],
                        signal["beta"],
                        signal["sharpe_ratio"],
                        signal["sortino_ratio"],
                        signal["calmar_ratio"],
                        signal["omega_ratio"],
                        signal["treynor_ratio"],
                        signal["information_ratio"],
                        signal["mtf_score"],
                        signal["normalized_price_change"],
                        signal["pullback_level"],
                        signal["scaled_avg_normalized"],
                        signal["vpms_mtf_score"],
                        signal["normalized_composite"],
                        signal["zscore_ratio_percent"],
                        signal["status"],
                    )
                )

            # Toplu insert
            await conn.executemany(
                """
                INSERT INTO signals (
                    symbol, timestamp, signal_type, interval, price,
                    indicators, vpms_score, strength, rsi, macd,
                    momentum, atr, adx, plus_di, minus_di,
                    alpha, beta, sharpe_ratio, sortino_ratio, calmar_ratio,
                    omega_ratio, treynor_ratio, information_ratio,
                    mtf_score, normalized_price_change, pullback_level,
                    scaled_avg_normalized, vpms_mtf_score,
                    normalized_composite, zscore_ratio_percent, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31)
                ON CONFLICT (symbol, timestamp) DO NOTHING
            """,
                values,
            )


# Standalone çalıştırma fonksiyonu
async def run_mtf_backfill(symbols: Optional[List[str]] = None, days_back: int = 30):
    """MTF Backfill'i çalıştır"""

    engine = MTFBackfillEngine()
    stats = await engine.run_backfill(symbols=symbols, days_back=days_back)

    print("\n🎉 MTF BACKFILL TAMAMLANDI!")
    print("=" * 40)

    # Süre hesaplama (None kontrolü ile)
    if stats.start_time and stats.end_time:
        duration = (stats.end_time - stats.start_time).total_seconds()
        print(f"⏱️ Süre: {duration:.1f} saniye")
    else:
        print("⏱️ Süre: Hesaplanamadı")

    print(f"📊 İşlenen sembol: {stats.processed_symbols}")
    print(f"📈 İşlenen bar: {stats.processed_bars:,}")
    print(f"🎯 Üretilen sinyal: {stats.generated_signals:,}")
    print("\n📊 Timeframe bazlı sinyaller:")

    # Timeframe signals None kontrolü ile
    if stats.timeframe_signals:
        for tf, count in stats.timeframe_signals.items():
            print(f"   {tf}: {count:,} sinyal")
    else:
        print("   Timeframe verileri bulunamadı")

    return stats


if __name__ == "__main__":
    # Test çalıştırması
    asyncio.run(run_mtf_backfill(symbols=["BTCUSDT", "ETHUSDT"], days_back=7))
