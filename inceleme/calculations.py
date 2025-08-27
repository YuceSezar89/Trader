import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

# Pandas ayarları
pd.options.mode.copy_on_write = True

# Loglama ayarları
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parametreler
LOOKBACK_COMMON = 50
SMOOTH_PERIOD_COMMON = 3
MOMENTUM_LENGTH = 14
RSI_LENGTH = 14

# Normalize fonksiyonu
def normalize_series(s, lookback=LOOKBACK_COMMON, index=None):
    logger.debug(f"Normalizing series with lookback {lookback}, input type: {type(s)}")
    if not isinstance(s, pd.Series):
        logger.warning(f"Expected pandas.Series, got {type(s)}, converting to Series")
        s = pd.Series(s, index=index)
    s = s.fillna(0)
    max_abs = s.abs().rolling(window=lookback, min_periods=1).max()
    max_abs = max_abs.fillna(1)
    normalized = (s / max_abs) * 100
    if normalized.isna().any():
        logger.warning("NaN detected in normalized series")
        normalized = normalized.fillna(0)
    return pd.Series(normalized, index=s.index).fillna(0)

# Düzleştirme fonksiyonu
def smooth_series(s, window=SMOOTH_PERIOD_COMMON, index=None):
    logger.debug(f"Smoothing series with window {window}, input type: {type(s)}")
    if not isinstance(s, pd.Series):
        logger.warning(f"Expected pandas.Series, got {type(s)}, converting to Series")
        s = pd.Series(s, index=index)
    smoothed = s.rolling(window=window, min_periods=1).mean().fillna(0)
    if smoothed.isna().any():
        logger.warning("NaN detected in smoothed series")
        smoothed = smoothed.fillna(0)
    return smoothed

# Tek sembol için hesaplamalar
def calculate_metrics(df):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input DataFrame head:\n{df.head(5)}")
    logger.debug(f"DataFrame dtypes:\n{df.dtypes}")

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

    df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
    logger.debug(f"After timezone conversion:\n{df.head(5)}")

    df['price'] = pd.Series(df['close'], index=df.index)
    df['price_diff'] = pd.Series(df['price'] - df['price'].shift(1), index=df.index)
    df['percent_price_diff'] = pd.Series((df['price_diff'] / df['price'].shift(1).replace(0, 1)) * 100, index=df.index)
    df['normalized_price_diff'] = normalize_series(df['percent_price_diff'], index=df.index)
    df['smoothed_price_diff'] = smooth_series(df['normalized_price_diff'], index=df.index)

    df['volume_diff'] = pd.Series(df['volume'] - df['volume'].shift(1), index=df.index)
    df['percent_volume_diff'] = pd.Series((df['volume_diff'] / df['volume'].shift(1).replace(0, 1)) * 100, index=df.index)
    df['normalized_volume_diff'] = normalize_series(df['percent_volume_diff'], index=df.index)
    df['smoothed_volume'] = smooth_series(df['normalized_volume_diff'], index=df.index)

    df['volatility'] = pd.Series(df['high'] - df['low'], index=df.index)
    df['volatility_diff'] = pd.Series(df['volatility'] - df['volatility'].shift(1), index=df.index)
    df['percent_volatility_diff'] = pd.Series((df['volatility_diff'] / df['volatility'].shift(1).replace(0, 1)) * 100, index=df.index)
    df['normalized_volatility_diff'] = normalize_series(df['percent_volatility_diff'], index=df.index)
    df['smoothed_volatility'] = smooth_series(df['normalized_volatility_diff'], index=df.index)

    df['momentum'] = pd.Series(df['price'] - df['price'].shift(MOMENTUM_LENGTH), index=df.index)
    df['momentum_diff'] = pd.Series(df['momentum'] - df['momentum'].shift(1), index=df.index)
    df['percent_momentum_diff'] = pd.Series((df['momentum_diff'] / np.maximum(1, df['momentum'].shift(1).abs())) * 100, index=df.index)
    df['normalized_momentum_diff'] = normalize_series(df['percent_momentum_diff'], index=df.index)
    df['smoothed_momentum'] = smooth_series(df['normalized_momentum_diff'], index=df.index)

    df['rsi'] = pd.Series(ta.rsi(df['price'], length=RSI_LENGTH), index=df.index)
    df['rsi_diff'] = pd.Series(df['rsi'] - df['rsi'].shift(1), index=df.index)
    df['percent_rsi_diff'] = pd.Series((df['rsi_diff'] / np.maximum(1, df['rsi'].shift(1).abs())) * 100, index=df.index)
    df['normalized_rsi_diff'] = normalize_series(df['percent_rsi_diff'], index=df.index)
    df['smoothed_rsi'] = smooth_series(df['normalized_rsi_diff'], index=df.index)

    df['hour'] = pd.Series(df.index.hour, index=df.index)
    df['is_new_day'] = pd.Series((df['hour'] == 3) & (df.index.minute == 0), index=df.index)
    df['daily_ref_price'] = pd.Series(np.where(df['is_new_day'], df['price'], np.nan), index=df.index)
    df['daily_ref_price'] = pd.Series(df['daily_ref_price'].ffill().fillna(df['price'].iloc[0]), index=df.index)
    df['cumulative_change'] = pd.Series(((df['price'] / df['daily_ref_price']) - 1) * 100, index=df.index)
    df['normalized_cumulative'] = normalize_series(df['cumulative_change'], index=df.index)
    df['smoothed_cumulative'] = smooth_series(df['normalized_cumulative'], index=df.index)

    df['returns'] = pd.Series(np.log(df['price'] / df['price'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0), index=df.index)
    df['avg_return'] = pd.Series(df['returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).mean(), index=df.index)
    df['std_dev_return'] = pd.Series(df['returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).std().fillna(0), index=df.index)
    df['sharpe_ratio'] = pd.Series(np.where(df['std_dev_return'] == 0, 0, df['avg_return'] / (df['std_dev_return'] + 1e-6)), index=df.index)
    if df['sharpe_ratio'].isna().any():
        logger.warning("NaN detected in sharpe_ratio")
        df['sharpe_ratio'] = df['sharpe_ratio'].fillna(0)

    df['downside_returns'] = pd.Series(np.where(df['returns'] < 0, df['returns'], 0), index=df.index)
    df['std_dev_downside'] = pd.Series(df['downside_returns'].rolling(window=LOOKBACK_COMMON, min_periods=1).std().fillna(0), index=df.index)
    df['sortino_ratio'] = pd.Series(np.where(df['std_dev_downside'] == 0, 0, df['avg_return'] / (df['std_dev_downside'] + 1e-6)), index=df.index)
    if df['sortino_ratio'].isna().any():
        logger.warning("NaN detected in sortino_ratio")
        df['sortino_ratio'] = df['sortino_ratio'].fillna(0)

    df['max_drawdown'] = pd.Series(df['low'].rolling(window=LOOKBACK_COMMON, min_periods=1).min(), index=df.index)
    df['drawdown'] = pd.Series(df['price'] - df['max_drawdown'], index=df.index)
    df['drawdown_percent'] = pd.Series((df['drawdown'] / df['price']) * 100, index=df.index)
    min_drawdown = df['drawdown_percent'].rolling(window=LOOKBACK_COMMON, min_periods=1).min().abs().div(100.0).add(1e-6)
    df['calmar_ratio'] = pd.Series(df['avg_return'] / min_drawdown, index=df.index)
    if df['calmar_ratio'].isna().any():
        logger.warning("NaN detected in calmar_ratio")
        df['calmar_ratio'] = df['calmar_ratio'].fillna(0)

    df['positive_returns'] = pd.Series(np.where(df['returns'] > 0, df['returns'], 0), index=df.index).rolling(window=LOOKBACK_COMMON, min_periods=1).sum()
    df['negative_returns'] = pd.Series(np.where(df['returns'] < 0, -df['returns'], 0), index=df.index).rolling(window=LOOKBACK_COMMON, min_periods=1).sum()
    df['omega_ratio'] = pd.Series(np.where(df['negative_returns'] == 0, 0, df['positive_returns'] / (df['negative_returns'] + 1e-6)), index=df.index)
    if df['omega_ratio'].isna().any():
        logger.warning("NaN detected in omega_ratio")
        df['omega_ratio'] = df['omega_ratio'].fillna(0)

    df['composite_ratio'] = pd.Series((df['sharpe_ratio'] + df['sortino_ratio'] + df['calmar_ratio'] + df['omega_ratio']) / 4, index=df.index)
    df['normalized_composite'] = normalize_series(df['composite_ratio'], index=df.index)
    df['smoothed_composite'] = smooth_series(df['normalized_composite'], index=df.index)

    df['signal_momentum_percent'] = pd.Series(((df['price'] / df['open'].replace(0, 1)) - 1) * 100, index=df.index)
    df['normalized_signal_momentum'] = normalize_series(df['signal_momentum_percent'], index=df.index)
    df['smoothed_signal_momentum'] = smooth_series(df['normalized_signal_momentum'], index=df.index)

    df['buy_crossover'] = pd.Series((df['smoothed_cumulative'] > 0) & (df['smoothed_cumulative'].shift(1) <= 0), index=df.index)
    df['sell_crossunder'] = pd.Series((df['smoothed_cumulative'] < 0) & (df['smoothed_cumulative'].shift(1) >= 0), index=df.index)

    df['is_buy_active'] = pd.Series(False, index=df.index)
    df['is_sell_active'] = pd.Series(False, index=df.index)
    df['buy_mum_sayisi'] = pd.Series(0, index=df.index)
    df['sell_mum_sayisi'] = pd.Series(0, index=df.index)
    df['saved_signal_momentum'] = pd.Series(np.nan, index=df.index)
    df['buy_metin'] = pd.Series("", index=df.index)
    df['sell_metin'] = pd.Series("", index=df.index)

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

def calculate_indicators(data_list):
    """
    Birden fazla sembol için metrikleri hesaplar ve dictionary listesi döndürür.
    """
    results = []
    
    for item in data_list:
        symbol = item['symbol']
        df = item['data']
        try:
            if df.empty:
                raise ValueError(f"Empty DataFrame for {symbol}")
            df_metrics = calculate_metrics(df)
            latest = df_metrics.iloc[-1]
            result = {
                'symbol': symbol,
                'Score': "{:.2f}".format(float(latest['smoothed_cumulative'])),
                'Fiyat': "{:.2f}".format(float(latest['smoothed_price_diff'])),
                'Volume': "{:.2f}".format(float(latest['smoothed_volume'])),
                'Volatil': "{:.2f}".format(float(latest['smoothed_volatility'])),
                'Moment': "{:.2f}".format(float(latest['smoothed_momentum'])),
                'RSI': "{:.2f}".format(float(latest['smoothed_rsi'])),
                'Ratio': "{:.2f}".format(float(latest['smoothed_composite'])),
                'L/S/T': latest['buy_metin'] if latest['buy_metin'] else '-',
                'S/S/T': latest['sell_metin'] if latest['sell_metin'] else '-'
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            raise
    
    return results