"""
Basit Backtest Runner - Database'deki sinyalleri kullanarak backtest yapar
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import psycopg2
from config import Config

logger = logging.getLogger(__name__)


class SimpleBacktest:
    """
    Database'deki mevcut sinyalleri kullanarak basit backtest yapar.
    Gerçek sinyal performanslarını analiz eder.
    """
    
    def __init__(self):
        # Database config
        self.db_config = {
            'host': Config.DB_HOST,
            'port': Config.DB_PORT,
            'database': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD
        }
        
    def run_backtest(
        self,
        days_back: int = 7,
        intervals: Optional[List[str]] = None,
        min_vpm_score: float = 0.0,  # 0 = tüm sinyaller
        signal_filter: Optional[str] = None  # 'C20MX', 'RSI', 'MA200' vb.
    ) -> Dict:
        """
        Basit backtest çalıştır - mevcut sinyalleri analiz et
        
        Args:
            days_back: Kaç gün geriye git
            intervals: Test edilecek interval'ler (None = hepsi)
            min_vpm_score: Minimum VPM skoru
        
        Returns:
            Dict: Backtest sonuçları
        """
        logger.info(f"Backtest başlatılıyor: {days_back} gün, intervals={intervals}")
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Sinyalleri çek
            signals = self._get_signals(cursor, days_back, intervals, min_vpm_score, signal_filter)
            
            if not signals:
                logger.warning("Hiç sinyal bulunamadı!")
                return self._empty_result()
            
            # Her sinyali analiz et
            trades = []
            for signal in signals:
                trade = self._analyze_signal(cursor, signal)
                if trade:
                    trades.append(trade)
            
            # Sonuçları hesapla
            results = self._calculate_results(trades, days_back, intervals)
            
            # Database'e kaydet
            backtest_id = self._save_results(cursor, results)
            results['backtest_id'] = backtest_id
            
            conn.commit()
            logger.info(f"Backtest tamamlandı: {len(trades)} işlem, Win Rate: {results['win_rate']:.2f}%")
            
            return results
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Backtest hatası: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _interval_to_minutes(self, interval: str) -> int:
        """
        Interval string'ini dakikaya çevir
        
        Args:
            interval: '1m', '5m', '15m', '1h' vb.
            
        Returns:
            Dakika cinsinden süre
        """
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 1440
        
        return 5  # Default
    
    def _get_price_at_time(
        self,
        cursor,
        symbol: str,
        target_time,
        interval: str
    ) -> Optional[float]:
        """
        Belirli bir zamandaki fiyatı database'den çek
        
        Args:
            symbol: Coin sembolü (BTCUSDT)
            target_time: Hedef zaman
            interval: Zaman dilimi (5m, 15m)
            
        Returns:
            Close fiyatı veya None
        """
        
        # Database'den en yakın bar'ı çek
        query = """
            SELECT close
            FROM price_data
            WHERE symbol = %s
              AND interval = %s
              AND timestamp <= %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        cursor.execute(query, (symbol, interval, target_time))
        result = cursor.fetchone()
        
        if result:
            return float(result[0])
        
        return None
    
    def _get_signals(
        self,
        cursor,
        days_back: int,
        intervals: Optional[List[str]],
        min_vpm_score: float,
        signal_filter: Optional[str] = None
    ) -> List[Dict]:
        """Database'den sinyalleri çek"""
        
        interval_filter = ""
        if intervals:
            interval_list = "', '".join(intervals)
            interval_filter = f"AND interval IN ('{interval_list}')"
        
        # Signal filter (indicators LIKE)
        signal_type_filter = ""
        if signal_filter:
            signal_type_filter = f"AND indicators LIKE '%{signal_filter}%'"
        
        # Sadece mevcut kolonları kullan
        query = f"""
            SELECT 
                id, symbol, signal_type, interval, timestamp,
                price, vpms_mtf_score, strength,
                rsi, macd, momentum, atr
            FROM signals
            WHERE timestamp >= NOW() - INTERVAL '{days_back} days'
              AND vpms_mtf_score IS NOT NULL
              AND vpms_mtf_score >= %s
              {interval_filter}
              {signal_type_filter}
            ORDER BY timestamp ASC;
        """
        
        cursor.execute(query, (min_vpm_score,))
        
        columns = [desc[0] for desc in cursor.description]
        signals = []
        
        for row in cursor.fetchall():
            signal = dict(zip(columns, row))
            signals.append(signal)
        
        logger.info(f"Toplam {len(signals)} sinyal bulundu")
        return signals
    
    def _analyze_signal(self, cursor, signal: Dict) -> Optional[Dict]:
        """
        Tek bir sinyali analiz et ve trade oluştur
        
        GERÇEK FİYAT VERİSİ İLE:
        - AL sinyali → Long pozisyon
        - SAT sinyali → Short pozisyon
        - T+5 bar sonrası gerçek fiyatı kullan
        - Stop loss: -1.5%
        - Take profit: +2.0%
        """
        
        # Database'den Long/Short gelir, SAT/AL'a çevir (app.py ile aynı mantık)
        signal_type = signal['signal_type']
        if signal_type == 'Long':
            signal_type = 'SAT'
        elif signal_type == 'Short':
            signal_type = 'AL'
        
        entry_price = float(signal['price'])
        entry_time = signal['timestamp']
        symbol = signal['symbol']
        interval = signal['interval']
        
        # T+5 bar sonrası zamanı hesapla
        interval_minutes = self._interval_to_minutes(interval)
        exit_time = entry_time + timedelta(minutes=5 * interval_minutes)
        
        # Gerçek exit fiyatını çek
        exit_price = self._get_price_at_time(cursor, symbol, exit_time, interval)
        
        # Fiyat bulunamadıysa skip
        if exit_price is None:
            return None
        
        # Gerçek PnL hesapla
        if signal_type == 'AL':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SAT
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        # Stop loss / Take profit kontrolü
        exit_reason = 'T5_CLOSE'
        original_pnl = pnl_pct
        
        if pnl_pct <= -1.5:  # Stop loss
            exit_reason = 'STOP_LOSS'
            pnl_pct = -1.5
            exit_price = entry_price * (1 - 0.015) if signal_type == 'AL' else entry_price * (1 + 0.015)
        elif pnl_pct >= 2.0:  # Take profit
            exit_reason = 'TAKE_PROFIT'
            pnl_pct = 2.0
            exit_price = entry_price * (1 + 0.02) if signal_type == 'AL' else entry_price * (1 - 0.02)
        
        trade = {
            'signal_id': signal['id'],
            'symbol': symbol,
            'side': 'BUY' if signal_type == 'AL' else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 1.0,  # Normalized - 1 birim
            'entry_time': entry_time,
            'timeframe': interval,
            'vpm_score': float(signal['vpms_mtf_score']) if signal['vpms_mtf_score'] else 0,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'mfe': 0,  # TODO: Gerçek MFE hesapla
            'mae': 0,  # TODO: Gerçek MAE hesapla
        }
        
        return trade
    
    def _calculate_results(
        self,
        trades: List[Dict],
        days_back: int,
        intervals: Optional[List[str]]
    ) -> Dict:
        """Backtest sonuçlarını hesapla"""
        
        if not trades:
            return self._empty_result()
        
        # Temel metrikler
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl_pct'] < 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrikleri
        total_pnl_pct = sum(t['pnl_pct'] for t in trades)
        avg_pnl_pct = total_pnl_pct / total_trades if total_trades > 0 else 0
        
        wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Max drawdown (basit)
        cumulative_pnl = 0
        peak = 0
        max_dd = 0
        
        for trade in trades:
            cumulative_pnl += trade['pnl_pct']
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_dd = max(max_dd, drawdown)
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade['pnl_pct'] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Exit reason analizi
        exit_reasons: Dict[str, int] = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'strategy_name': 'Simple_VPM_Backtest',
            'timeframe': ','.join(intervals) if intervals else 'ALL',
            'days_back': days_back,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def _save_results(self, cursor, results: Dict) -> int:
        """Sonuçları database'e kaydet"""
        
        query = """
            INSERT INTO backtest_results (
                strategy_name, timeframe, start_date, end_date,
                total_trades, winning_trades, losing_trades, win_rate,
                total_pnl_percentage, avg_win, avg_loss,
                profit_factor, max_drawdown,
                max_consecutive_wins, max_consecutive_losses
            ) VALUES (
                %s, %s, NOW() - INTERVAL '%s days', NOW(),
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s
            ) RETURNING id;
        """
        
        cursor.execute(query, (
            results['strategy_name'],
            results['timeframe'],
            results['days_back'],
            results['total_trades'],
            results['winning_trades'],
            results['losing_trades'],
            results['win_rate'],
            results['total_pnl_pct'],
            results['avg_win'],
            results['avg_loss'],
            results['profit_factor'],
            results['max_drawdown'],
            results['max_consecutive_wins'],
            results['max_consecutive_losses']
        ))
        
        backtest_id = cursor.fetchone()[0]
        
        # Trade'leri kaydet
        for trade in results['trades']:
            trade_query = """
                INSERT INTO backtest_trades (
                    backtest_id, symbol, side, entry_price, exit_price,
                    quantity, pnl_percentage, entry_time, timeframe,
                    signal_id, vpm_score, exit_reason
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                );
            """
            
            cursor.execute(trade_query, (
                backtest_id,
                trade['symbol'],
                trade['side'],
                trade['entry_price'],
                trade['exit_price'],
                trade['quantity'],
                trade['pnl_pct'],
                trade['entry_time'],
                trade['timeframe'],
                trade['signal_id'],
                trade['vpm_score'],
                trade['exit_reason']
            ))
        
        return backtest_id
    
    def _empty_result(self) -> Dict:
        """Boş sonuç döndür"""
        return {
            'strategy_name': 'Simple_VPM_Backtest',
            'timeframe': 'ALL',
            'days_back': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'avg_pnl_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'exit_reasons': {},
            'trades': []
        }


def run_simple_backtest(
    days_back: int = 7,
    intervals: Optional[List[str]] = None,
    min_vpm_score: float = 0.0,
    signal_filter: Optional[str] = None
):
    """
    Convenience function - basit backtest çalıştır
    
    Usage:
        results = run_simple_backtest(
            days_back=7,
            intervals=['5m', '15m'],
            min_vpm_score=70.0
        )
    """
    backtest = SimpleBacktest()
    return backtest.run_backtest(
        days_back=days_back,
        intervals=intervals,
        min_vpm_score=min_vpm_score,
        signal_filter=signal_filter
    )
