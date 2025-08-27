import pandas as pd
import numpy as np


import logging
from utils.preprocessing import normalize_series, smooth_series

# Varsayılan parametreler (gerekirse güncellenebilir)
LOOKBACK_COMMON = 50
SMOOTH_PERIOD_COMMON = 3
MOMENTUM_LENGTH = 14
RSI_LENGTH = 14

logger = logging.getLogger(__name__)

def calculate_metrics(df, ref_df=None, beta_window=50, alpha_window=20):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input DataFrame head:\n{df.head(5)}")
    logger.debug(f"DataFrame dtypes:\n{df.dtypes}")
    # SettingWithCopyWarning uyarılarını önlemek için olası dilimleri kopyaya çevir
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas.DataFrame, got {type(df)}")
        raise ValueError(f"Expected pandas.DataFrame, got {type(df)}")
    if df.empty:
        logger.error("Empty DataFrame provided")
        raise ValueError("Empty DataFrame")
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        logger.error("Missing required columns in DataFrame")
        raise ValueError("Missing required columns")
    if df['close'].le(0).any():
        logger.error("Invalid close prices (zero or negative)")
        raise ValueError("Invalid close prices")

    # Index'in datetime tipinde olup olmadığını ve tz özelliği olup olmadığını kontrol et
    if hasattr(df.index, 'tz'):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
        else:
            df.index = df.index.tz_convert('Europe/Istanbul')
    else:
        # RangeIndex veya başka bir index tipi - datetime index'e çevir
        if 'date' in df.columns:
            df.index = pd.to_datetime(df['date'])
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
            else:
                df.index = df.index.tz_convert('Europe/Istanbul')
        else:
            # date sütunu yoksa, metrik hesaplamayı atla
            logger.warning("DataFrame'de date sütunu yok ve index datetime değil, metrik hesaplama atlanıyor.")
            df['alpha'] = 0.0
            df['beta'] = 0.0
            df['scaled_avg_normalized'] = 0.0
            df['normalized_composite'] = 0.0
            return df
    logger.debug(f"After timezone conversion:\n{df.head(5)}")

    # Alpha ve Beta için varsayılan değerleri ata
    df['alpha'] = 0.0
    df['beta'] = 0.0

    # --- Alpha ve Beta Hesaplaması (Yeniden yapılandırıldı) ---
    if ref_df is not None and not ref_df.empty:
        try:
            # Referans verinin de zaman dilimini ayarla (sadece DatetimeIndex için)
            if hasattr(ref_df.index, 'tz') and hasattr(ref_df.index, 'tz_localize'):
                if ref_df.index.tz is None:
                    ref_df.index = ref_df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
                else:
                    ref_df.index = ref_df.index.tz_convert('Europe/Istanbul')
            # RangeIndex veya diğer index tipleri için date sütununu kullan
            elif 'date' in ref_df.columns:
                ref_df = ref_df.set_index('date')
                if ref_df.index.tz is None:
                    ref_df.index = ref_df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
                else:
                    ref_df.index = ref_df.index.tz_convert('Europe/Istanbul')

            # Ensure 'date' column is the index and remove duplicates
            if 'date' in df.columns:
                df = df.set_index('date')
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep='first')]

            if 'date' in ref_df.columns:
                ref_df = ref_df.set_index('date')
            if not ref_df.index.is_unique:
                ref_df = ref_df[~ref_df.index.duplicated(keep='first')]

            # Ortak zaman indeksini bul ve her iki df'i de bu indekse göre hizala
            common_index = df.index.intersection(ref_df.index)
            logger.info(f"Ortak index uzunluğu: {len(common_index)}, df: {len(df)}, ref_df: {len(ref_df)}")
            if len(common_index) > beta_window:
                aligned_df = df.loc[common_index]
                aligned_ref_df = ref_df.loc[common_index]

                # Getirileri hesapla
                asset_returns = aligned_df['close'].pct_change(fill_method=None).fillna(0)
                market_returns = aligned_ref_df['close'].pct_change(fill_method=None).fillna(0)

                # Kovaryans ve varyansı hesapla (en az 2 veri noktası gerekli)
                if len(asset_returns) > 1 and len(market_returns) > 1:
                    # Tekrar hizala, çünkü pct_change sonrası bir index kayabilir
                    common_returns_index = asset_returns.index.intersection(market_returns.index)
                    asset_returns = asset_returns.reindex(common_returns_index)
                    market_returns = market_returns.reindex(common_returns_index)

                    covariance_matrix = np.cov(asset_returns, market_returns)
                    covariance = covariance_matrix[0, 1]
                    market_variance = np.var(market_returns)

                    # --- Rolling Beta Hesaplaması (TradingView'e göre) ---
                    rolling_cov = asset_returns.rolling(window=beta_window).cov(market_returns)
                    rolling_var = market_returns.rolling(window=beta_window).var()
                    rolling_beta = rolling_cov / rolling_var
                    
                    logger.info(f"Rolling beta hesaplandı: min={rolling_beta.min():.4f}, max={rolling_beta.max():.4f}, son={rolling_beta.iloc[-1]:.4f}")

                    # --- Özel Alpha Hesaplaması (TradingView'e göre) ---
                    # Pine Script: ret2 = (close - close[y]) / close, y=90
                    asset_returns_alpha = (aligned_df['close'] - aligned_df['close'].shift(alpha_window)) / aligned_df['close']
                    market_returns_alpha = (aligned_ref_df['close'] - aligned_ref_df['close'].shift(alpha_window)) / aligned_ref_df['close']
                    
                    # Pine Script Formülü: alpha = (ret2 - retb2 * beta) * 100
                    rolling_alpha = (asset_returns_alpha - market_returns_alpha * rolling_beta) * 100

                    # Hesaplanan rolling değerleri ana DataFrame'e (df) geri birleştir
                    df['beta'] = rolling_beta.reindex(df.index, method='ffill')
                    df['alpha'] = rolling_alpha.reindex(df.index, method='ffill')
                    
                    # NaN değerleri doldur (FutureWarning düzeltmesi)
                    df['beta'] = df['beta'].fillna(0.0)
                    df['alpha'] = df['alpha'].fillna(0.0)

                    logger.info(f"Rolling Alpha ve Beta, {beta_window} bar pencere ile hesaplandı.")
                else:
                    logger.warning("Alpha/Beta hesaplamak için yeterli getiri verisi yok.")
            else:
                logger.warning("Varlık ve referans verisi arasında ortak zaman aralığı bulunamadı.")
        except Exception as e:
            logger.error(f"Alpha/Beta hesaplama sırasında bir hata oluştu: {e}", exc_info=True)


    df.loc[:, 'price'] = df['close']
    df.loc[:, 'price_diff'] = df['price'] - df['price'].shift(1)
    df.loc[:, 'percent_price_diff'] = (df['price_diff'] / df['price'].shift(1).replace(0, 1)) * 100
    df['normalized_price_diff'] = normalize_series(df['percent_price_diff'], index=df.index).fillna(0.0)
    df['smoothed_price_diff'] = smooth_series(df['normalized_price_diff'], index=df.index)

    df.loc[:, 'volume_diff'] = df['volume'] - df['volume'].shift(1)
    df.loc[:, 'percent_volume_diff'] = (df['volume_diff'] / df['volume'].shift(1).replace(0, 1)) * 100
    df['normalized_volume_diff'] = normalize_series(df['percent_volume_diff'], index=df.index)
    df['smoothed_volume'] = smooth_series(df['normalized_volume_diff'], index=df.index)

    df.loc[:, 'volatility'] = df['high'] - df['low']
    df.loc[:, 'volatility_diff'] = df['volatility'] - df['volatility'].shift(1)
    df.loc[:, 'percent_volatility_diff'] = (df['volatility_diff'] / df['volatility'].shift(1).replace(0, 1)) * 100
    df['normalized_volatility_diff'] = normalize_series(df['percent_volatility_diff'], index=df.index)
    df['smoothed_volatility'] = smooth_series(df['normalized_volatility_diff'], index=df.index)

    df.loc[:, 'momentum'] = df['price'] - df['price'].shift(MOMENTUM_LENGTH)
    df.loc[:, 'momentum_diff'] = df['momentum'] - df['momentum'].shift(1)
    df.loc[:, 'percent_momentum_diff'] = (df['momentum_diff'] / np.maximum(1, df['momentum'].shift(1).abs())) * 100
    df['normalized_momentum_diff'] = normalize_series(df['percent_momentum_diff'], index=df.index)
    df['smoothed_momentum'] = smooth_series(df['normalized_momentum_diff'], index=df.index)

    # Kendi RSI fonksiyonumuzu kullan (pandas_ta NumPy uyumsuzluğu yüzünden)
    from indicators.core import calculate_rsi
    # RSI hesapla (price sütunu üzerinden)
    rsi_series = calculate_rsi(df, period=RSI_LENGTH, price_col='price')
    df.loc[:, 'rsi'] = rsi_series
    df.loc[:, 'rsi_diff'] = df['rsi'] - df['rsi'].shift(1)
    df.loc[:, 'percent_rsi_diff'] = (df['rsi_diff'] / np.maximum(1, df['rsi'].shift(1).abs())) * 100
    df['normalized_rsi_diff'] = normalize_series(df['percent_rsi_diff'], index=df.index)
    df['smoothed_rsi'] = smooth_series(df['normalized_rsi_diff'], index=df.index)

    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'is_new_day'] = (df['hour'] == 3) & (df.index.minute == 0)
    df.loc[:, 'daily_ref_price'] = np.where(df['is_new_day'], df['price'], np.nan)
    df.loc[:, 'daily_ref_price'] = df['daily_ref_price'].ffill().fillna(df['price'].iloc[0])
    df.loc[:, 'cumulative_change'] = ((df['price'] / df['daily_ref_price']) - 1) * 100
    df['normalized_cumulative'] = normalize_series(df['cumulative_change'], index=df.index)
    df['smoothed_cumulative'] = smooth_series(df['normalized_cumulative'], index=df.index)

    df.loc[:, 'returns'] = np.log(df['price'] / df['price'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    df.loc[:, 'avg_return'] = df['returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).mean()
    df.loc[:, 'std_dev_return'] = df['returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).std().fillna(0)
    df.loc[:, 'sharpe_ratio'] = np.where(df['std_dev_return'] == 0, 0, df['avg_return'] / (df['std_dev_return'] + 1e-6))
    if df['sharpe_ratio'].isna().any():
        logger.warning("NaN detected in sharpe_ratio")
        df['sharpe_ratio'] = df['sharpe_ratio'].fillna(0)

    df.loc[:, 'downside_returns'] = np.where(df['returns'] < 0, df['returns'], 0)
    df.loc[:, 'std_dev_downside'] = df['downside_returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).std().fillna(0)
    df.loc[:, 'sortino_ratio'] = np.where(df['std_dev_downside'] == 0, 0, df['avg_return'] / (df['std_dev_downside'] + 1e-6))
    if df['sortino_ratio'].isna().any():
        logger.warning("NaN detected in sortino_ratio")
        df['sortino_ratio'] = df['sortino_ratio'].fillna(0)

    df.loc[:, 'max_drawdown'] = df['low'].rolling(window=LOOKBACK_COMMON, min_periods=1).min()
    df.loc[:, 'drawdown'] = df['price'] - df['max_drawdown']
    df.loc[:, 'drawdown_percent'] = (df['drawdown'] / df['price']) * 100
    min_drawdown = df['drawdown_percent'].rolling(window=LOOKBACK_COMMON, min_periods=1).min().abs().div(100.0).add(1e-6)
    df.loc[:, 'calmar_ratio'] = df['avg_return'] / min_drawdown
    if df['calmar_ratio'].isna().any():
        logger.warning("NaN detected in calmar_ratio")
        df['calmar_ratio'] = df['calmar_ratio'].fillna(0)

    df.loc[:, 'positive_returns'] = np.where(df['returns'] > 0, df['returns'], 0)
    df['positive_returns'] = df['positive_returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).sum()
    df.loc[:, 'negative_returns'] = np.where(df['returns'] < 0, -df['returns'], 0)
    df['negative_returns'] = df['negative_returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).sum()
    df.loc[:, 'omega_ratio'] = np.where(df['negative_returns'] == 0, 0, df['positive_returns'] / (df['negative_returns'] + 1e-6))
    if df['omega_ratio'].isna().any():
        logger.warning("NaN detected in omega_ratio")
        df['omega_ratio'] = df['omega_ratio'].fillna(0)

    # --- Treynor Ratio ---
    if ref_df is not None and not ref_df.empty:
        # Risk-free rate (devisso uyumluluğu için)
        RISK_FREE_RATE = 0.001  # devisso'daki sabit değer
        risk_free_adjusted = RISK_FREE_RATE / 600  # devisso formülü
        
        # Get market returns (ref_df)
        aligned_df = df.copy()
        aligned_ref_df = ref_df.reindex(df.index).copy()
        asset_returns = aligned_df['returns']
        # Explicitly set fill_method=None to avoid FutureWarning in newer pandas versions
        market_returns = aligned_ref_df['close'].pct_change(fill_method=None).fillna(0)
        mean_market_return = market_returns.rolling(window=LOOKBACK_COMMON, min_periods=1).mean()
        covar = (asset_returns - df['avg_return']) * (market_returns - mean_market_return)
        rolling_cov = covar.rolling(window=LOOKBACK_COMMON, min_periods=1).mean()
        rolling_var = market_returns.rolling(window=LOOKBACK_COMMON, min_periods=1).var().replace(0, np.nan)
        beta = rolling_cov / rolling_var
        beta = beta.replace([np.inf, -np.inf], 0).fillna(0)
        # Devisso uyumlu Treynor formülü: (avg_return - risk_free_rate) / beta
        df['treynor_ratio'] = np.where(beta == 0, 0, (df['avg_return'] - risk_free_adjusted) / (beta + 1e-6))
        if df['treynor_ratio'].isna().any():
            logger.warning("NaN detected in treynor_ratio")
            df['treynor_ratio'] = df['treynor_ratio'].fillna(0)
        # --- Information Ratio ---
        # Excess returns hesaplama (asset - market)
        excess_returns = asset_returns - market_returns
        
        # Rolling tracking error hesaplama (minimum 30 veri noktası)
        tracking_error = excess_returns.rolling(window=max(30, LOOKBACK_COMMON), min_periods=30).std()
        
        # Rolling excess return ortalama
        mean_excess_returns = excess_returns.rolling(window=max(30, LOOKBACK_COMMON), min_periods=30).mean()
        
        # Information Ratio hesaplama: mean_excess_return / tracking_error
        df['information_ratio'] = np.where(
            (tracking_error.isna()) | (tracking_error == 0) | (tracking_error < 1e-8),
            0.0,
            mean_excess_returns / tracking_error
        )
        
        # NaN değerleri 0 ile doldur
        df['information_ratio'] = df['information_ratio'].fillna(0.0)
        
        # Aşırı değerleri sınırla (-10 ile +10 arası)
        df['information_ratio'] = df['information_ratio'].clip(-10, 10)
        
        # Debug için sadece çok fazla NaN varsa log
        nan_count = df['information_ratio'].isna().sum()
        if nan_count > len(df) * 0.1:  # %10'dan fazla NaN varsa
            logger.debug(f"High NaN count in information_ratio: {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)")
        # --- Normalize ratios (0-1 scale) ---
        df['n_treynor'] = normalize_series(df['treynor_ratio'], index=df.index)
        df['n_information'] = normalize_series(df['information_ratio'], index=df.index)
    else:
        df['treynor_ratio'] = 0.0
        df['information_ratio'] = 0.0
        df['n_treynor'] = 0.0
        df['n_information'] = 0.0

    # Composite ratio, normalize, smooth (Treynor ve Info dahil)
    df.loc[:, 'composite_ratio'] = (df['sharpe_ratio'] + df['sortino_ratio'] + df['calmar_ratio'] + df['omega_ratio'] + df['treynor_ratio'] + df['information_ratio']) / 6
    df['normalized_composite'] = normalize_series(df['composite_ratio'], index=df.index).fillna(0.0)
    df['smoothed_composite'] = smooth_series(df['normalized_composite'], index=df.index)
    # Scaled avg normalized (0-100) devisso uyumlu
    df['scaled_avg_normalized'] = (df[['sharpe_ratio','sortino_ratio','treynor_ratio','calmar_ratio','information_ratio']].apply(normalize_series, axis=0, index=df.index).mean(axis=1) * 100).fillna(0.0)

    df.loc[:, 'signal_momentum_percent'] = ((df['price'] / df['open'].replace(0, 1)) - 1) * 100
    df['normalized_signal_momentum'] = normalize_series(df['signal_momentum_percent'], index=df.index)
    df['smoothed_signal_momentum'] = smooth_series(df['normalized_signal_momentum'], index=df.index)

    df.loc[:, 'buy_crossover'] = (df['smoothed_cumulative'] > 0) & (df['smoothed_cumulative'].shift(1) <= 0)
    df.loc[:, 'sell_crossunder'] = (df['smoothed_cumulative'] < 0) & (df['smoothed_cumulative'].shift(1) >= 0)

    df.loc[:, 'is_buy_active'] = False
    df.loc[:, 'is_sell_active'] = False
    df.loc[:, 'buy_mum_sayisi'] = 0
    df.loc[:, 'sell_mum_sayisi'] = 0
    df.loc[:, 'saved_signal_momentum'] = np.nan
    df.loc[:, 'buy_metin'] = ""
    df.loc[:, 'sell_metin'] = ""

    for i in range(1, len(df)):
        if df['buy_crossover'].iloc[i]:
            df.loc[df.index[i], 'is_buy_active'] = True
            df.loc[df.index[i], 'is_sell_active'] = False
            df.loc[df.index[i], 'buy_mum_sayisi'] = 0
            df.loc[df.index[i], 'sell_mum_sayisi'] = 0
            df.loc[df.index[i], 'saved_signal_momentum'] = df['smoothed_signal_momentum'].iloc[i]
        elif df['sell_crossunder'].iloc[i]:
            df.loc[df.index[i], 'is_buy_active'] = False
            df.loc[df.index[i], 'is_sell_active'] = True
            df.loc[df.index[i], 'buy_mum_sayisi'] = 0
            df.loc[df.index[i], 'sell_mum_sayisi'] = 0
            df.loc[df.index[i], 'saved_signal_momentum'] = df['smoothed_signal_momentum'].iloc[i]
        else:
            df.loc[df.index[i], 'is_buy_active'] = df['is_buy_active'].iloc[i-1]
            df.loc[df.index[i], 'is_sell_active'] = df['is_sell_active'].iloc[i-1]
            df.loc[df.index[i], 'buy_mum_sayisi'] = df['buy_mum_sayisi'].iloc[i-1] + 1 if df['is_buy_active'].iloc[i-1] else 0
            df.loc[df.index[i], 'sell_mum_sayisi'] = df['sell_mum_sayisi'].iloc[i-1] + 1 if df['is_sell_active'].iloc[i-1] else 0
            df.loc[df.index[i], 'saved_signal_momentum'] = df['saved_signal_momentum'].iloc[i-1]

        if df['is_buy_active'].iloc[i]:
            df.loc[df.index[i], 'buy_metin'] = f"{df['saved_signal_momentum'].iloc[i]:.2f}/{df['buy_mum_sayisi'].iloc[i]}"
        if df['is_sell_active'].iloc[i]:
            df.loc[df.index[i], 'sell_metin'] = f"{df['saved_signal_momentum'].iloc[i]:.2f}/{df['sell_mum_sayisi'].iloc[i]}"

    logger.debug(f"buy_metin:\n{df['buy_metin'].tail(5)}")
    logger.debug(f"sell_metin:\n{df['sell_metin'].tail(5)}")

    return df
