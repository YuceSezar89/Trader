import ccxt
import pandas as pd
import logging
from dotenv import load_dotenv
import os

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

# .env dosyasını yükle
load_dotenv()

# Binance Futures bağlantısı
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'  # USDT-M Futures
    }
})

def fetch_ohlcv_df(exchange, symbol, timeframe='5m', limit=1000):
    logger.debug(f"Fetching OHLCV data for {symbol}, timeframe={timeframe}, limit={limit}")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        logger.debug(f"Raw OHLCV data length: {len(ohlcv)}")
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ).astype({
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logger.debug(f"DataFrame created for {symbol}:\n{df.tail(5)}")
        logger.debug(f"DataFrame dtypes:\n{df.dtypes}")
        logger.debug(f"Column types: {[(col, type(df[col])) for col in df.columns]}")
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            raise ValueError("Empty DataFrame returned")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
        raise

def fetch_ohlcv_data(symbol, timeframe='5m'):
    logger.debug(f"Calling fetch_ohlcv_df for {symbol}, timeframe={timeframe}")
    df = fetch_ohlcv_df(exchange, symbol, timeframe)
    return df

def fetch_data(timeframe='5m'):
    """
    Birden fazla sembol için veri çeker, timeframe parametresiyle.
    """
    symbols = ['RPL/USDT', 'TRB/USDT', 'ZEN/USDT', 'AERGO/USDT', 'ICX/USDT', 'SXP/USDT', 'QUICK/USDT', 'FLM/USDT', 'LISTA/USDT', 'LINA/USDT', 'LIT/USDT', 'XMR/USDT', 'LPT/USDT', 'SFP/USDT', 'LDO/USDT', 'KAVA/USDT', 'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT']
    results = []
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            df = fetch_ohlcv_data(symbol, timeframe)
            results.append({'symbol': f"{idx} - {symbol.split('/')[0]}", 'data': df})
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            raise
    
    return results