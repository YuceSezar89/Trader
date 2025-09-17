# Windsurf Veritabanı Eklentisi Bağlantı Bilgileri

## 🔌 Bağlantı Ayarları

### PgBouncer Üzerinden (Önerilen)
```
Host: localhost
Port: 6432
Database: trader_panel
Username: yusuf
Password: (boş bırakın - trust authentication)
SSL Mode: disable
```

### Direkt PostgreSQL
```
Host: localhost
Port: 5432
Database: trader_panel
Username: yusuf
Password: (boş bırakın - trust authentication)
SSL Mode: disable
```

## 📊 Mevcut Tablolar

### 1. `signals` Tablosu
- **Kayıt Sayısı:** 2
- **Açıklama:** Trading sinyalleri
- **Ana Sütunlar:**
  - `symbol` (VARCHAR) - Sembol adı (örn: BTCUSDT)
  - `signal_type` (VARCHAR) - Sinyal türü (Long/Short)
  - `timestamp` (TIMESTAMP) - Sinyal zamanı
  - `price` (NUMERIC) - Sinyal fiyatı
  - `strength` (NUMERIC) - Sinyal gücü
  - `interval` (VARCHAR) - Zaman dilimi
  - `rsi_value` (NUMERIC) - RSI değeri
  - `macd_value` (NUMERIC) - MACD değeri
  - `volume` (NUMERIC) - Hacim

### 2. `price_data` Tablosu
- **Kayıt Sayısı:** 0 (henüz veri yok)
- **Açıklama:** OHLCV fiyat verileri ve teknik indikatörler
- **Ana Sütunlar:**
  - `symbol` (VARCHAR) - Sembol adı
  - `timestamp` (TIMESTAMP) - Veri zamanı
  - `open_price` (NUMERIC) - Açılış fiyatı
  - `high_price` (NUMERIC) - En yüksek fiyat
  - `low_price` (NUMERIC) - En düşük fiyat
  - `close_price` (NUMERIC) - Kapanış fiyatı
  - `volume` (NUMERIC) - Hacim
  - `rsi_9` (NUMERIC) - 9 periyot RSI
  - `rsi_14` (NUMERIC) - 14 periyot RSI
  - `ma_20` (NUMERIC) - 20 periyot MA
  - `ma_50` (NUMERIC) - 50 periyot MA
  - `ma_200` (NUMERIC) - 200 periyot MA
  - `macd` (NUMERIC) - MACD değeri
  - `macd_signal` (NUMERIC) - MACD sinyal hattı

## 🔍 Örnek Sorgular

### Son Sinyalleri Görüntüle
```sql
SELECT 
    symbol,
    signal_type,
    timestamp,
    price,
    strength
FROM signals 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Belirli Sembol İçin Sinyaller
```sql
SELECT * FROM signals 
WHERE symbol = 'BTCUSDT' 
ORDER BY timestamp DESC;
```

### Son 24 Saatteki Sinyaller
```sql
SELECT * FROM signals 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
```

### Sinyal Türlerine Göre Dağılım
```sql
SELECT 
    signal_type,
    COUNT(*) as count
FROM signals 
GROUP BY signal_type;
```

## ⚙️ Bağlantı Test Komutu

Terminal'de test etmek için:
```bash
# PgBouncer üzerinden
psql -h localhost -p 6432 -U yusuf -d trader_panel -c "SELECT COUNT(*) FROM signals;"

# Direkt PostgreSQL
psql -h localhost -p 5432 -U yusuf -d trader_panel -c "SELECT COUNT(*) FROM signals;"
```

## 📝 Notlar

1. **Authentication:** Trust authentication kullanılıyor, şifre gerekmez
2. **PgBouncer:** Connection pooling için kullanılıyor (port 6432)
3. **SSL:** Yerel bağlantı olduğu için SSL gerekmiyor
4. **Encoding:** UTF-8
5. **Timezone:** Europe/Istanbul (UTC+3)

## 🚀 Mevcut Veriler

```
Son Sinyaller:
- 2025-09-10 23:43:54 | BTCUSDT | Long | $45000.0
- 2025-09-10 23:29:35 | BTCUSDT | Long | $50000.0
```
