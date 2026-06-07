#!/usr/bin/env python3
"""
Aktif Sinyal Takipçisi - Sadece aktif sinyalleri olan semboller için optimized canlı veri takibi

Bu modül:
1. Aktif sinyalleri veritabanından çeker
2. Sadece bu semboller için WebSocket bağlantısı kurar  
3. Real-time fiyat güncellemeleri alır
4. Sinyal durumlarını otomatik günceller (hit/miss/expired)
5. Performans metriklerini hesaplar
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from binance.client import Client
from binance.streams import BinanceSocketManager
import asyncpg

from utils.logger import get_logger
from config import Config
from database.connection_manager import get_connection_manager

logger = get_logger(__name__)

class ActiveSignalTracker:
    """
    Aktif sinyaller için optimize edilmiş canlı veri takipçisi
    """
    
    def __init__(self) -> None:
        self.active_signals: Dict[str, List[Dict]] = {}  # symbol -> [signal_data]
        self.current_prices: Dict[str, float] = {}  # symbol -> current_price
        self.websocket_manager: Optional[BinanceSocketManager] = None
        self.client: Optional[Client] = None
        self.is_running = False
        self.update_interval = 30  # Sinyal durumu güncelleme aralığı (saniye)
        
    async def initialize(self) -> None:
        """Binance client ve WebSocket manager'ı başlat"""
        try:
            self.client = Client()
            self.websocket_manager = BinanceSocketManager(self.client)
            logger.info("ActiveSignalTracker başarıyla başlatıldı")
        except Exception as e:
            logger.error(f"ActiveSignalTracker başlatma hatası: {e}")
            raise
    
    async def load_active_signals(self) -> Dict[str, List[Dict]]:
        """Veritabanından aktif sinyalleri yükle"""
        try:
            async with get_connection_manager().get_connection() as conn:
                query = '''
                SELECT id, symbol, signal_type, timestamp, price, pullback_level,
                       strength, indicators, rsi as rsi_value, status, 
                       vpms_score, vpm_confirmed, interval
                FROM signals 
                WHERE status = 'active'
                ORDER BY timestamp DESC
                '''
                
                results = await conn.fetch(query)
                
                # Sembollere göre grupla
                signals_by_symbol: Dict[str, List[Dict]] = {}
                for row in results:
                    symbol = row['symbol']
                    if symbol not in signals_by_symbol:
                        signals_by_symbol[symbol] = []
                    
                    signal_data = {
                        'id': row['id'],
                        'symbol': symbol,
                        'signal_type': row['signal_type'],
                        'timestamp': row['timestamp'],
                        'entry_price': float(row['price']),
                        'pullback_level': float(row['pullback_level']) if row['pullback_level'] else None,
                        'strength': row['strength'],
                        'indicators': row['indicators'],
                        'rsi_value': float(row['rsi_value']) if row['rsi_value'] else None,
                        'vpms_score': float(row['vpms_score']) if row['vpms_score'] else None,
                        'vpm_confirmed': row['vpm_confirmed'],
                        'interval': row['interval'],
                        'current_price': None,
                        'price_change_pct': 0.0,
                        'unrealized_pnl': 0.0,
                        'bars_since_signal': 0
                    }
                    signals_by_symbol[symbol].append(signal_data)
                
                self.active_signals = signals_by_symbol
                
                logger.info(f"Aktif sinyaller yüklendi: {len(results)} sinyal, {len(signals_by_symbol)} sembol")
                return signals_by_symbol
                
        except Exception as e:
            logger.error(f"Aktif sinyal yükleme hatası: {e}")
            return {}
    
    async def start_price_tracking(self) -> None:
        """Sadece aktif sinyal sembolları için fiyat takibi başlat"""
        if not self.active_signals:
            await self.load_active_signals()
        
        symbols = list(self.active_signals.keys())
        if not symbols:
            logger.warning("Takip edilecek aktif sinyal bulunamadı")
            return
        
        logger.info(f"Fiyat takibi başlatılıyor: {len(symbols)} sembol")
        
        try:
            # Sembol listesini küçük harfe çevir (Binance WebSocket formatı)
            stream_symbols = [symbol.lower() for symbol in symbols]
            
            # Combined stream oluştur
            streams = [f"{symbol}@ticker" for symbol in stream_symbols]
            
            # WebSocket bağlantısı kur
            if self.websocket_manager is None:
                raise RuntimeError("WebSocket manager başlatılmamış!")
            socket = self.websocket_manager.multiplex_socket(streams)
            
            self.is_running = True
            
            async with socket as stream:
                logger.info(f"WebSocket bağlantısı kuruldu: {len(streams)} stream")
                
                while self.is_running:
                    try:
                        data = await stream.recv()
                        await self._process_price_update(data)
                    except Exception as e:
                        logger.error(f"Fiyat güncelleme işleme hatası: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            logger.error(f"WebSocket bağlantı hatası: {e}")
            self.is_running = False
    
    async def _process_price_update(self, data: dict) -> None:
        """Gelen fiyat güncellemesini işle"""
        try:
            if 'data' in data:
                ticker_data = data['data']
                symbol = ticker_data['s']  # BTCUSDT formatında
                current_price = float(ticker_data['c'])
                
                # Fiyatı güncelle
                self.current_prices[symbol] = current_price
                
                # Bu sembolün sinyallerini güncelle
                if symbol in self.active_signals:
                    for signal in self.active_signals[symbol]:
                        signal['current_price'] = current_price
                        
                        # Fiyat değişim yüzdesini hesapla
                        entry_price = signal['entry_price']
                        price_change = ((current_price - entry_price) / entry_price) * 100
                        
                        # Sinyal yönüne göre PnL hesapla
                        if signal['signal_type'] == 'Long':
                            signal['unrealized_pnl'] = price_change
                        else:  # Short
                            signal['unrealized_pnl'] = -price_change
                        
                        signal['price_change_pct'] = price_change
                
        except Exception as e:
            logger.error(f"Fiyat güncelleme işleme hatası: {e}")
    
    async def update_signal_status(self) -> None:
        """Sinyal durumlarını kontrol et ve batch olarak güncelle."""
        try:
            # status -> [id] mapping — tüm sinyalleri önce belleğe topla
            pending: Dict[str, List[int]] = {}

            for signals in self.active_signals.values():
                for signal in signals:
                    new_status = self._evaluate_signal_status(signal)
                    if new_status:
                        pending.setdefault(new_status, []).append(signal['id'])

            if pending:
                await self._batch_update_status_in_db(pending)

        except Exception as e:
            logger.error(f"Sinyal durum güncelleme hatası: {e}")

    def _evaluate_signal_status(self, signal: Dict) -> Optional[str]:
        """Tek bir sinyalin yeni durumunu hesapla (DB çağrısı yok)."""
        try:
            current_price = signal.get('current_price')
            if not current_price:
                return None

            hit_threshold = 2.0
            miss_threshold = 1.0
            signal_type = signal['signal_type']
            price_change_pct = signal.get('price_change_pct', 0.0)

            if signal_type == 'Long':
                if price_change_pct >= hit_threshold:
                    return 'hit'
                if price_change_pct <= -miss_threshold:
                    return 'miss'
            else:  # Short
                if price_change_pct <= -hit_threshold:
                    return 'hit'
                if price_change_pct >= miss_threshold:
                    return 'miss'

            signal_age = datetime.now() - signal['timestamp']
            if signal_age > timedelta(hours=24):
                return 'expired'

            return None
        except Exception as e:
            logger.error(f"Sinyal koşul değerlendirme hatası: {e}")
            return None

    async def _batch_update_status_in_db(self, pending: Dict[str, List[int]]) -> None:
        """Durumu değişen sinyalleri status'e göre gruplandırarak tek sorguda güncelle."""
        try:
            async with get_connection_manager().get_connection() as conn:
                for new_status, ids in pending.items():
                    await conn.execute(
                        "UPDATE signals SET status = $1, updated_at = NOW() WHERE id = ANY($2::int[])",
                        new_status, ids
                    )
                    logger.info(f"{len(ids)} sinyal '{new_status}' olarak güncellendi")
        except Exception as e:
            logger.error(f"Batch sinyal durum güncelleme hatası: {e}")
    
    def get_active_signals_summary(self) -> Dict:
        """Aktif sinyallerin özetini döndür"""
        total_signals = sum(len(signals) for signals in self.active_signals.values())
        
        # PnL istatistikleri
        pnl_values = []
        for signals in self.active_signals.values():
            for signal in signals:
                if signal.get('unrealized_pnl') is not None:
                    pnl_values.append(signal['unrealized_pnl'])
        
        summary = {
            'total_signals': total_signals,
            'total_symbols': len(self.active_signals),
            'avg_pnl': sum(pnl_values) / len(pnl_values) if pnl_values else 0,
            'positive_signals': len([p for p in pnl_values if p > 0]),
            'negative_signals': len([p for p in pnl_values if p < 0]),
            'last_update': datetime.now()
        }
        
        return summary
    
    async def stop(self) -> None:
        """Takipçiyi durdur"""
        self.is_running = False
        if self.client:
            await self.client.close_connection()
        logger.info("ActiveSignalTracker durduruldu")

# Global instance
active_signal_tracker = ActiveSignalTracker()
