import pandas as pd
import ta
import aiohttp
from indicators.core import calculate_ema
from typing import Optional, List, Tuple, Dict, Any

# Config import
from config import Config

# Error handling
from utils.exceptions import (
    BinanceAPIError,
    DataError,
    raise_api_timeout,
    ErrorCodes
)
from utils.logger import get_logger, log_error_with_context

# Logger
logger = get_logger(__name__)

def get_live_binance(
    symbol: str, 
    interval: str, 
    limit: int = 1000, 
    api_key: Optional[str] = None, 
    api_secret: Optional[str] = None, 
    **kwargs: Any
) -> pd.DataFrame:
    """
    Binance API'dan OHLCV verisi çeker. Sadece ham veri ve temel teknik göstergeler hesaplanır.
    
    Args:
        symbol: Trading sembolü (örn: 'BTCUSDT')
        interval: Zaman aralığı (örn: '1h', '4h', '1d')
        limit: Çekilecek mum sayısı (varsayılan: 1000)
        api_key: Binance API anahtarı (opsiyonel)
        api_secret: Binance API gizli anahtarı (opsiyonel)
        **kwargs: Ek parametreler
        
    Returns:
        pd.DataFrame: OHLCV verisi içeren DataFrame
    """
    try:
        import requests
        symbol = symbol.upper()
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            resp = requests.get(url, timeout=Config.API_TIMEOUT)
            klines = resp.json()
            if isinstance(klines, dict) and klines.get("code"):
                print(f"Binance API Hatası: {klines}")
                return pd.DataFrame()
        except Exception as api_err:
            print(f"Binance APIError: {api_err}")
            return pd.DataFrame()
        expected_cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(klines)
        if df.shape[1] > len(expected_cols):
            df = df.iloc[:, :len(expected_cols)]
        df.columns = expected_cols  # type: ignore
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["symbol"] = symbol
        # Sadece ham veri döndürülüyor; indikatörler dışarıda hesaplanacak
        return df
    except Exception as e:
        print(f"Veri çekilemedi! Hata: {e}")
        return pd.DataFrame()

import asyncio
import aiohttp

# --- Central Asynchronous Client Manager ---

class BinanceClientManager:
    """
    Manages aiohttp requests for the Binance API by creating a session per request.
    This approach is more compatible with Streamlit's script execution model.
    """

    @classmethod
    async def fetch_klines(cls, symbol: str, interval: str, limit: int = 500, startTime: Optional[int] = None) -> pd.DataFrame:
        """Fetches historical klines for a single symbol."""
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if startTime:
            url += f"&startTime={startTime}"
        try:
            async def request():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        return await resp.json()
            data = await asyncio.wait_for(request(), timeout=15)
            if isinstance(data, dict) and data.get("code"):
                logger.warning(f"[{symbol}] Binance API Error: {data}")
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            return df
        except aiohttp.ClientError as e:
            # Ağ/bağlantı sorunları için
            logger.error(f"[{symbol}] Kline verisi çekilirken ağ hatası: {e}", exc_info=True)
            raise BinanceAPIError(f"Network error for {symbol}: {e}", endpoint=url) from e
        except asyncio.TimeoutError:
            # Zaman aşımları için
            logger.error(f"[{symbol}] Kline verisi çekilirken zaman aşımı.", exc_info=True)
            raise BinanceAPIError(f"Timeout for {symbol}", endpoint=url)
        except Exception as e:
            # JSON'da döndürülen API düzeyindeki hatalar da dahil olmak üzere diğer beklenmedik hatalar için
            logger.error(f"[{symbol}] Kline verisi çekilirken beklenmedik hata: {e}", exc_info=True)
            raise BinanceAPIError(f"Failed to fetch klines for {symbol}: {e}", endpoint=url) from e

    @classmethod
    async def fetch_all_klines(cls, symbols: List[str], interval: str, limit: int = 500) -> List[Tuple[str, pd.DataFrame]]:
        """Asynchronously fetches klines for a list of symbols."""
        tasks = [cls.fetch_klines(symbol, interval, limit) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results: List[Tuple[str, pd.DataFrame]] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame):
                processed_results.append((symbol, result))
            elif isinstance(result, Exception):
                logger.error(f"Error fetching klines for {symbol}: {result}")
                processed_results.append((symbol, pd.DataFrame()))
            else:
                logger.error(f"Unexpected result type for {symbol}: {type(result)!r}")
                processed_results.append((symbol, pd.DataFrame()))
        return processed_results

    @classmethod
    async def get_24hr_ticker_stats(cls) -> List[Dict[str, Any]]:
        """Fetches 24-hour ticker statistics for all symbols."""
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        logger.info("Fetching 24hr ticker stats asynchronously...")
        try:
            async def request():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
            data = await asyncio.wait_for(request(), timeout=Config.API_TIMEOUT)
            if not isinstance(data, list):
                raise DataError("Expected data format not found: response is not a list.")
            logger.info(f"✅ Successfully fetched 24hr ticker stats for {len(data)} symbols.")
            return data
        except aiohttp.ClientError as e:
            raise BinanceAPIError(f"Network or connection error: {e}", endpoint=url) from e
        except Exception as e:
            raise BinanceAPIError(f"An unexpected error occurred: {e}", endpoint=url) from e

    @classmethod
    async def get_top_volume_symbols_async(cls, limit: int = 200) -> List[str]:
        """Fetches top symbols by 24-hour volume from Binance Futures."""
        logger.info(f"Fetching top {limit} symbols by 24hr volume...")
        try:
            stats = await cls.get_24hr_ticker_stats()
            # Filter for USDT pairs with a minimum volume threshold
            initial_usdt_pairs = [s for s in stats if s['symbol'].endswith('USDT')]
            volume_threshold = 10000.0  # Minimum 24hr volume in USDT
            
            valid_symbols = [
                s for s in initial_usdt_pairs 
                if float(s.get('quoteVolume', 0)) > volume_threshold
            ]
            
            logger.info(f"{len(initial_usdt_pairs) - len(valid_symbols)} symbols were filtered out due to low volume (under {volume_threshold} USDT).")

            sorted_symbols = sorted(
                valid_symbols,
                key=lambda x: float(x.get('quoteVolume', 0)),
                reverse=True
            )
            top_symbols = [s['symbol'] for s in sorted_symbols[:limit]]
            logger.info(f"✅ Successfully identified top {limit} symbols by volume.")
            return top_symbols
        except Exception as e:
            logger.error(f"Unexpected error fetching top volume symbols: {e}")
            raise BinanceAPIError(f"An unexpected error occurred: {e}") from e

# For direct testing of this module
if __name__ == "__main__":
    async def main_test():
        await BinanceClientManager.initialize()
        try:
            # Test fetching top symbols
            print("--- Testing get_top_volume_symbols_async ---")
            top_symbols = await BinanceClientManager.get_top_volume_symbols_async(limit=10)
            print(f"Top 10 symbols: {top_symbols}")

            # Test fetching klines for top symbols
            if top_symbols:
                print("\n--- Testing fetch_all_klines ---")
                async_results = await BinanceClientManager.fetch_all_klines(top_symbols[:3], "1h", limit=2)
                for sym, df in async_results:
                    print(f"{sym} data fetched (async), shape: {df.shape}")
                    if not df.empty:
                        print(df.head(2))
        finally:
            await BinanceClientManager.close()

    asyncio.run(main_test())
