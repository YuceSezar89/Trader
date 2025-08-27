import asyncio
import asyncio
from sqlalchemy import select, update
import pandas as pd
from datetime import datetime, timedelta
import pytz

from database.engine import get_session
from database.models import Signal
from binance_client import BinanceClientManager
from utils.logger import get_logger # Merkezi logger'ı import et

# --- Configuration (can be moved to config.py later) ---
PERFORMANCE_TRACKER_SLEEP_INTERVAL = 60  # seconds
PERFORMANCE_TRACKER_LOOKBACK_HOURS = 72 # Check signals from the last 3 days
# --------------------------------------------------------

# Logging setup
logger = get_logger("PerformanceTracker")


def _interval_to_ms(iv: str) -> int:
    """Parse Binance interval string (e.g., '1m','5m','15m','1h','4h','1d') to milliseconds."""
    try:
        iv = (iv or "15m").strip().lower()
        unit = iv[-1]
        val = int(iv[:-1])
        if unit == 'm':
            return val * 60_000
        if unit == 'h':
            return val * 3_600_000
        if unit == 'd':
            return val * 86_400_000
    except Exception:
        pass
    return 15 * 60_000


async def get_klines(symbol, interval, start_time_ms, limit=2):
    """Fetch klines from Binance using the BinanceClientManager."""
    try:
        df = await BinanceClientManager.fetch_klines(symbol=symbol, interval=interval, startTime=start_time_ms, limit=limit)
        if df.empty or len(df.index) < limit:
            logger.info(f"Not enough kline data for {symbol} at start time {start_time_ms}. Found {len(df.index)} candle(s). Will retry after next close.")
            return None
        # The manager already returns a DataFrame with correct types and column names.
        # We just need to rename 'open_time' to 'timestamp' for consistency with our script.
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        # Convert to datetime and make it timezone-aware (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol} via BinanceClientManager: {e}")
        return None

async def track_signal_performance():
    """
    Tracks the performance of signals marked as 'pending' by calculating 
    the momentum and volume change of the next candle.
    """
    async with get_session() as session:
        try:
            lookback_time = datetime.now(pytz.utc) - timedelta(hours=PERFORMANCE_TRACKER_LOOKBACK_HOURS)
            
            stmt = select(Signal).where(
                Signal.perf_status == 'pending',
                Signal.signal_time >= lookback_time.strftime("%Y-%m-%d %H:%M:%S")
            ).order_by(Signal.signal_time.asc())
            
            result = await session.execute(stmt)
            pending_signals = result.scalars().all()

            if not pending_signals:
                logger.info("No pending signals to track.")
                return

            logger.info(f"Found {len(pending_signals)} pending signals to track.")

            for signal in pending_signals:
                logger.info(f"Tracking signal ID {signal.id} for {signal.symbol} ({signal.signal_type}) at {signal.signal_time}")
                
                # --- Timezone-Aware Timestamp Handling ---
                # 1. Parse the signal_time string from DB into a naive datetime object.
                signal_dt_naive = datetime.strptime(signal.signal_time, "%Y-%m-%d %H:%M:%S")

                # 2. Localize the naive datetime to 'Europe/Istanbul' to make it timezone-aware.
                istanbul_tz = pytz.timezone('Europe/Istanbul')
                signal_dt_istanbul = istanbul_tz.localize(signal_dt_naive)

                # 3. Convert the timezone-aware Istanbul time to UTC.
                # This gives us the correct UTC timestamp to request from Binance.
                signal_dt_utc = signal_dt_istanbul.astimezone(pytz.utc)
                signal_timestamp_ms = int(signal_dt_utc.timestamp() * 1000)

                # Fetch 3 candles: the one before the signal, the signal candle, and the one after.
                # We need to start from the timestamp of the candle BEFORE the signal candle.
                interval_ms = _interval_to_ms(getattr(signal, 'interval', '15m'))

                # If the next candle hasn't closed yet, skip this signal for now (avoid noisy warnings)
                now_utc = datetime.now(pytz.utc)
                next_close_utc = signal_dt_utc + timedelta(milliseconds=interval_ms)
                if now_utc < next_close_utc + timedelta(seconds=30):  # grace window to ensure closure & data propagation
                    logger.info(
                        f"Signal ID {signal.id}: Next candle not closed yet. Waiting until {next_close_utc} UTC"
                    )
                    continue

                start_time_ms = int(signal_dt_utc.timestamp() * 1000) - interval_ms
                df = await get_klines(signal.symbol, signal.interval, start_time_ms, limit=3)

                if df is None:
                    logger.info(f"Signal ID {signal.id}: No klines data received (will retry later).")
                    continue
                elif len(df) < 3:
                    logger.warning(f"Signal ID {signal.id}: Expected 3 klines, but got {len(df)}. Skipping.")
                    continue

                # 5. The first row should be our signal candle. Verify the timestamp in UTC.
                # The DataFrame's 'timestamp' is already timezone-aware (UTC).
                # The DataFrame's 'timestamp' is already a timezone-aware pandas Timestamp.
                # We convert it directly to a timezone-aware python datetime object for comparison.
                signal_candle_timestamp_utc = df['timestamp'].iloc[1].to_pydatetime()

                # Compare the two timezone-aware datetime objects.
                if signal_candle_timestamp_utc != signal_dt_utc:
                    logger.warning(
                        f"Signal ID {signal.id}: Timestamp mismatch. Skipping.\n"
                        f"  Expected: {signal_dt_utc} (Type: {type(signal_dt_utc)})\n"
                        f"  Got:      {signal_candle_timestamp_utc} (Type: {type(signal_candle_timestamp_utc)})"
                    )
                    continue

                prev_candle = df.iloc[0]
                signal_candle = df.iloc[1]
                next_candle = df.iloc[2]

                # --- Calculations for signal -> next candle ---
                momentum_signal = ((signal_candle['close'] - signal_candle['open']) / signal_candle['open']) * 100
                momentum_next = ((next_candle['close'] - next_candle['open']) / next_candle['open']) * 100
                
                # Oransal Yüzde Değişimi Hesaplaması
                if momentum_signal != 0:
                    signal_to_next_momentum_change = ((momentum_next - momentum_signal) / abs(momentum_signal)) * 100
                else:
                    # Eğer başlangıç momentumu 0 ise, değişim ya 0'dır ya da sonsuzdur.
                    # Veritabanı uyumluluğu için 0.0 olarak ayarlıyoruz.
                    signal_to_next_momentum_change = 0.0

                if signal_candle['volume'] > 0:
                    signal_to_next_volume_change = ((next_candle['volume'] - signal_candle['volume']) / signal_candle['volume']) * 100
                else:
                    signal_to_next_volume_change = 0.0

                # --- Calculations for prev -> signal candle ---
                momentum_prev = ((prev_candle['close'] - prev_candle['open']) / prev_candle['open']) * 100

                # Oransal Yüzde Değişimi Hesaplaması
                if momentum_prev != 0:
                    prev_to_signal_momentum_change = ((momentum_signal - momentum_prev) / abs(momentum_prev)) * 100
                else:
                    prev_to_signal_momentum_change = 0.0

                if prev_candle['volume'] > 0:
                    prev_to_signal_volume_change = ((signal_candle['volume'] - prev_candle['volume']) / prev_candle['volume']) * 100
                else:
                    prev_to_signal_volume_change = 0.0

                # --- YENİ: Mum-içi Kâr Potansiyeli Hesaplaması (Sinyal Sonrası Mum) ---
                intra_candle_profit_pct = 0.0
                next_candle_open = next_candle['open']
                next_candle_high = next_candle['high']
                next_candle_low = next_candle['low']

                next_candle_close = next_candle['close']

                if signal.signal_type == 'Long':
                    # AL sinyali sonrası: En düşükten alıp kapanışta satma potansiyeli
                    if next_candle_low > 0:
                        intra_candle_profit_pct = ((next_candle_close - next_candle_low) / next_candle_low) * 100
                elif signal.signal_type == 'Short':
                    # SAT sinyali sonrası: En yüksekten satıp kapanışta pozisyonu kapatma potansiyeli
                    if next_candle_high > 0:
                        intra_candle_profit_pct = ((next_candle_high - next_candle_close) / next_candle_high) * 100
                
                # --- Update Database ---
                update_stmt = update(Signal).where(Signal.id == signal.id).values(
                    # Post-signal performance
                    perf_next_candle_momentum_change_pct=signal_to_next_momentum_change,
                    perf_next_candle_volume_change_pct=signal_to_next_volume_change,
                    perf_intra_candle_profit_pct=intra_candle_profit_pct, # Yeni eklenen metrik
                    # Pre-signal context
                    perf_prev_to_signal_momentum_change_pct=prev_to_signal_momentum_change,
                    perf_prev_to_signal_volume_change_pct=prev_to_signal_volume_change,
                    perf_status='completed'
                )
                await session.execute(update_stmt)
                logger.info(f"Successfully updated signal ID {signal.id}. Post-Signal Momentum Change: {signal_to_next_momentum_change:.2f}%, Post-Signal Volume Change: {signal_to_next_volume_change:.2f}%")
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"An error occurred in the main tracking loop: {e}", exc_info=True)
            # The get_session context manager will handle the rollback.

async def main():
    """Main loop to run the tracker periodically."""
    try:
        while True:
            logger.info("--- Starting new performance tracking cycle ---")
            await track_signal_performance()
            logger.info(f"--- Cycle finished. Sleeping for {PERFORMANCE_TRACKER_SLEEP_INTERVAL} seconds ---")
            await asyncio.sleep(PERFORMANCE_TRACKER_SLEEP_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Performance tracker task cancelled. Shutting down gracefully...")
        raise
    except Exception as e:
        logger.error(f"Fatal error in performance tracker main loop: {e}", exc_info=True)
        raise
    finally:
        logger.info("Performance tracker main loop exited.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Performance tracker stopped by user.")
