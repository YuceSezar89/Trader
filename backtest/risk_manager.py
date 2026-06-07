"""
Risk Yönetimi Sistemi - Position sizing, stop loss, take profit hesaplamaları
"""

from decimal import Decimal
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk yönetimi sistemi
    - Position sizing
    - Stop loss hesaplama
    - Take profit hesaplama
    - Risk kontrolü
    """
    
    def __init__(self):
        # Varsayılan risk parametreleri
        self.max_position_size_percentage = 5.0  # Portföyün %5'i
        self.stop_loss_percentage = 2.0          # %2 stop loss
        self.take_profit_percentage = 5.0        # %5 take profit
        self.max_daily_loss_percentage = 10.0    # Günlük max kayıp %10
        self.max_open_positions = 5              # Maksimum açık pozisyon sayısı
        
        # Risk takibi
        self.daily_pnl = Decimal('0')
        self.open_positions_count = 0
    
    def calculate_position_size(
        self,
        account_balance: Decimal,
        entry_price: float,
        risk_percentage: Optional[float] = None
    ) -> Decimal:
        """
        Pozisyon büyüklüğü hesapla
        
        Args:
            account_balance: Hesap bakiyesi
            entry_price: Giriş fiyatı
            risk_percentage: Risk yüzdesi (opsiyonel)
        
        Returns:
            Decimal: Pozisyon büyüklüğü (quantity)
        """
        try:
            # Risk yüzdesini belirle
            risk_pct = risk_percentage or self.max_position_size_percentage
            
            # Maksimum pozisyon değeri
            max_position_value = account_balance * Decimal(str(risk_pct / 100))
            
            # Quantity hesapla
            quantity = max_position_value / Decimal(str(entry_price))
            
            # Minimum quantity kontrolü (0.001)
            min_quantity = Decimal('0.001')
            if quantity < min_quantity:
                return Decimal('0')
            
            logger.debug(f"Position size: {quantity} (Risk: {risk_pct}%, Value: {max_position_value})")
            return quantity
            
        except Exception as e:
            logger.error(f"Position size hesaplama hatası: {e}")
            return Decimal('0')
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> Decimal:
        """
        Stop loss fiyatı hesapla
        
        Args:
            entry_price: Giriş fiyatı
            side: İşlem yönü ('BUY' veya 'SELL')
        
        Returns:
            Decimal: Stop loss fiyatı
        """
        try:
            entry_decimal = Decimal(str(entry_price))
            stop_loss_ratio = Decimal(str(self.stop_loss_percentage / 100))
            
            if side == 'BUY':
                # Long pozisyon: giriş fiyatının altında stop loss
                stop_loss = entry_decimal * (Decimal('1') - stop_loss_ratio)
            else:  # SELL
                # Short pozisyon: giriş fiyatının üstünde stop loss
                stop_loss = entry_decimal * (Decimal('1') + stop_loss_ratio)
            
            logger.debug(f"Stop loss ({side}): {stop_loss} (Entry: {entry_price})")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Stop loss hesaplama hatası: {e}")
            return Decimal(str(entry_price))
    
    def calculate_take_profit(self, entry_price: float, side: str) -> Decimal:
        """
        Take profit fiyatı hesapla
        
        Args:
            entry_price: Giriş fiyatı
            side: İşlem yönü ('BUY' veya 'SELL')
        
        Returns:
            Decimal: Take profit fiyatı
        """
        try:
            entry_decimal = Decimal(str(entry_price))
            take_profit_ratio = Decimal(str(self.take_profit_percentage / 100))
            
            if side == 'BUY':
                # Long pozisyon: giriş fiyatının üstünde take profit
                take_profit = entry_decimal * (Decimal('1') + take_profit_ratio)
            else:  # SELL
                # Short pozisyon: giriş fiyatının altında take profit
                take_profit = entry_decimal * (Decimal('1') - take_profit_ratio)
            
            logger.debug(f"Take profit ({side}): {take_profit} (Entry: {entry_price})")
            return take_profit
            
        except Exception as e:
            logger.error(f"Take profit hesaplama hatası: {e}")
            return Decimal(str(entry_price))
    
    def check_risk_limits(
        self,
        account_balance: Decimal,
        proposed_trade_value: Decimal
    ) -> Dict[str, bool]:
        """
        Risk limitlerini kontrol et
        
        Args:
            account_balance: Hesap bakiyesi
            proposed_trade_value: Önerilen işlem değeri
        
        Returns:
            Dict: Risk kontrol sonuçları
        """
        checks = {
            'position_size_ok': True,
            'daily_loss_ok': True,
            'max_positions_ok': True,
            'overall_risk_ok': True
        }
        
        try:
            # Pozisyon büyüklüğü kontrolü
            max_position_value = account_balance * Decimal(str(self.max_position_size_percentage / 100))
            if proposed_trade_value > max_position_value:
                checks['position_size_ok'] = False
                logger.warning(f"Pozisyon büyüklüğü limiti aşıldı: {proposed_trade_value} > {max_position_value}")
            
            # Günlük kayıp kontrolü
            max_daily_loss = account_balance * Decimal(str(self.max_daily_loss_percentage / 100))
            if abs(self.daily_pnl) > max_daily_loss:
                checks['daily_loss_ok'] = False
                logger.warning(f"Günlük kayıp limiti aşıldı: {abs(self.daily_pnl)} > {max_daily_loss}")
            
            # Maksimum pozisyon sayısı kontrolü
            if self.open_positions_count >= self.max_open_positions:
                checks['max_positions_ok'] = False
                logger.warning(f"Maksimum pozisyon sayısı aşıldı: {self.open_positions_count} >= {self.max_open_positions}")
            
            # Genel risk kontrolü
            checks['overall_risk_ok'] = all([
                checks['position_size_ok'],
                checks['daily_loss_ok'],
                checks['max_positions_ok']
            ])
            
        except Exception as e:
            logger.error(f"Risk kontrol hatası: {e}")
            checks['overall_risk_ok'] = False
        
        return checks
    
    def update_daily_pnl(self, pnl: Decimal):
        """Günlük P&L güncelle"""
        self.daily_pnl += pnl
        logger.debug(f"Günlük P&L güncellendi: {self.daily_pnl}")
    
    def update_position_count(self, change: int):
        """Açık pozisyon sayısını güncelle"""
        self.open_positions_count += change
        self.open_positions_count = max(0, self.open_positions_count)  # Negatif olamaz
        logger.debug(f"Açık pozisyon sayısı: {self.open_positions_count}")
    
    def reset_daily_stats(self):
        """Günlük istatistikleri sıfırla"""
        self.daily_pnl = Decimal('0')
        logger.info("Günlük istatistikler sıfırlandı")
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: str
    ) -> float:
        """
        Risk/Reward oranını hesapla
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            take_profit: Take profit fiyatı
            side: İşlem yönü
        
        Returns:
            float: Risk/Reward oranı
        """
        try:
            if side == 'BUY':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SELL
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                return 0.0
            
            ratio = reward / risk
            logger.debug(f"Risk/Reward oranı: {ratio:.2f} (Risk: {risk}, Reward: {reward})")
            return ratio
            
        except Exception as e:
            logger.error(f"Risk/Reward hesaplama hatası: {e}")
            return 0.0
    
    def get_risk_summary(self, account_balance: Decimal) -> Dict:
        """
        Risk özetini al
        
        Args:
            account_balance: Hesap bakiyesi
        
        Returns:
            Dict: Risk özeti
        """
        max_daily_loss = account_balance * Decimal(str(self.max_daily_loss_percentage / 100))
        daily_loss_used = (abs(self.daily_pnl) / max_daily_loss) * 100 if max_daily_loss > 0 else 0
        
        return {
            'account_balance': float(account_balance),
            'daily_pnl': float(self.daily_pnl),
            'daily_loss_limit': float(max_daily_loss),
            'daily_loss_used_percentage': float(daily_loss_used),
            'open_positions': self.open_positions_count,
            'max_positions': self.max_open_positions,
            'position_slots_used_percentage': (self.open_positions_count / self.max_open_positions) * 100,
            'risk_parameters': {
                'max_position_size_percentage': self.max_position_size_percentage,
                'stop_loss_percentage': self.stop_loss_percentage,
                'take_profit_percentage': self.take_profit_percentage,
                'max_daily_loss_percentage': self.max_daily_loss_percentage
            }
        }
    
    def update_risk_parameters(self, **kwargs):
        """Risk parametrelerini güncelle"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"Risk parametresi güncellendi: {param} = {value}")
            else:
                logger.warning(f"Bilinmeyen risk parametresi: {param}")


class AdvancedRiskManager(RiskManager):
    """
    Gelişmiş risk yönetimi - Volatilite bazlı position sizing
    """
    
    def __init__(self):
        super().__init__()
        self.volatility_adjustment = True
        self.correlation_adjustment = True
    
    def calculate_volatility_adjusted_position_size(
        self,
        account_balance: Decimal,
        entry_price: float,
        volatility: float,
        target_volatility: float = 0.02  # %2 hedef volatilite
    ) -> Decimal:
        """
        Volatiliteye göre ayarlanmış pozisyon büyüklüğü
        
        Args:
            account_balance: Hesap bakiyesi
            entry_price: Giriş fiyatı
            volatility: Mevcut volatilite (günlük)
            target_volatility: Hedef volatilite
        
        Returns:
            Decimal: Ayarlanmış pozisyon büyüklüğü
        """
        try:
            # Temel pozisyon büyüklüğü
            base_size = self.calculate_position_size(account_balance, entry_price)
            
            if not self.volatility_adjustment or volatility <= 0:
                return base_size
            
            # Volatilite ayarlaması
            volatility_multiplier = target_volatility / volatility
            volatility_multiplier = max(0.1, min(2.0, volatility_multiplier))  # 0.1x - 2.0x arası sınırla
            
            adjusted_size = base_size * Decimal(str(volatility_multiplier))
            
            logger.debug(f"Volatilite ayarlı pozisyon: {adjusted_size} (Çarpan: {volatility_multiplier:.2f})")
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Volatilite ayarlı position size hatası: {e}")
            return self.calculate_position_size(account_balance, entry_price)
    
    def calculate_correlation_adjusted_exposure(
        self,
        current_positions: Dict,
        new_symbol: str,
        correlation_matrix: Dict
    ) -> float:
        """
        Korelasyon bazlı risk exposure hesaplama
        
        Args:
            current_positions: Mevcut pozisyonlar
            new_symbol: Yeni sembol
            correlation_matrix: Korelasyon matrisi
        
        Returns:
            float: Ayarlanmış exposure çarpanı
        """
        try:
            if not self.correlation_adjustment or not current_positions:
                return 1.0
            
            total_correlation_risk = 0.0
            
            for symbol, position in current_positions.items():
                if symbol == new_symbol:
                    continue
                
                # Korelasyon değerini al
                correlation = correlation_matrix.get(f"{symbol}_{new_symbol}", 0.0)
                position_weight = float(position.get('weight', 0.0))
                
                # Korelasyon riski hesapla
                correlation_risk = abs(correlation) * position_weight
                total_correlation_risk += correlation_risk
            
            # Exposure çarpanını hesapla (yüksek korelasyon = düşük exposure)
            exposure_multiplier = max(0.2, 1.0 - total_correlation_risk)
            
            logger.debug(f"Korelasyon ayarlı exposure: {exposure_multiplier:.2f}")
            return exposure_multiplier
            
        except Exception as e:
            logger.error(f"Korelasyon ayarlı exposure hatası: {e}")
            return 1.0
