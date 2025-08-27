import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np

# Test edilecek modül
from signals.signal_processor import process_and_enrich_signals

class TestSignalProcessor(unittest.TestCase):

    def _create_test_dataframe(self, periods=100):
        """Örnek bir DataFrame oluşturur."""
        close_prices = 100 + np.random.randn(periods).cumsum()
        data = {
            'open_time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=periods, freq='15min')).astype(np.int64) // 10**6,
            'open': close_prices - np.random.uniform(0, 1, periods),
            'high': close_prices + np.random.uniform(0, 2, periods),
            'low': close_prices - np.random.uniform(0, 2, periods),
            'close': close_prices,
            'volume': np.random.uniform(1000, 5000, periods)
        }
        df = pd.DataFrame(data)
        return df

    @patch('signals.signal_processor.create_signal', new_callable=AsyncMock)
    @patch('signals.signal_processor.calculate_metrics')
    @patch('signals.signal_processor.signal_engine.calculate_all_signals', new_callable=AsyncMock)
    def test_process_and_enrich_signals_flow(self, mock_calculate_signals, mock_calculate_metrics, mock_create_signal):
        """
        `process_and_enrich_signals` fonksiyonunun tam akışını test eder:
        - Teknik sinyalleri alır.
        - Finansal metrikleri hesaplar.
        - İkisini birleştirir ve veritabanına kaydeder.
        """
        # --- Mock'ları Ayarla ---
        # 1. Teknik sinyal mock'u
        mock_calculate_signals.return_value = {
            'rsi_crossover': [
                {
                    'signal_type': 'Long',
                    'price': 105.5,
                    'indicators': 'RSI_Cross(9,21)',
                    'strength': 1
                }
            ]
        }

        # 2. Finansal metrik mock'u
        mock_metrics_df = pd.DataFrame([{'alpha': 0.05, 'beta': 1.2, 'sharpe_ratio': 1.5}])
        mock_calculate_metrics.return_value = mock_metrics_df

        # --- Test Verilerini Oluştur ---
        symbol_df = self._create_test_dataframe(periods=100)
        ref_df = self._create_test_dataframe(periods=100)

        # --- Testi Çalıştır ---
        asyncio.run(process_and_enrich_signals(
            symbol='ETHUSDT',
            df=symbol_df,
            ref_df=ref_df,
            interval='15m'
        ))

        # --- Doğrulamalar ---
        # Gerekli fonksiyonların çağrıldığını kontrol et
        mock_calculate_signals.assert_called_once()
        mock_calculate_metrics.assert_called_once()
        mock_create_signal.assert_called_once()

        # Kaydedilen verinin doğru şekilde zenginleştirildiğini doğrula
        call_args, _ = mock_create_signal.call_args
        saved_data = call_args[0]

        self.assertEqual(saved_data['symbol'], 'ETHUSDT')
        self.assertEqual(saved_data['interval'], '15m')
        self.assertEqual(saved_data['signal_type'], 'Long')
        self.assertEqual(saved_data['price'], 105.5)
        self.assertEqual(saved_data['alpha'], 0.05)
        self.assertEqual(saved_data['beta'], 1.2)
        self.assertEqual(saved_data['sharpe_ratio'], 1.5)

if __name__ == '__main__':
    unittest.main()
