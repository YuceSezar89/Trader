# TRader-Panel-ASYNC — Mimari Özeti ve Sinyal Akışı

Bu belge, proje içindeki ana modülleri ve sinyal üretim/zenginleştirme/veri kaydı akışını kısa ve pratik şekilde özetler.

## Ana Modüller

- **`live_data_manager.py`**
  - Görev: Tarihsel/veri senkronizasyonu, canlı veri akışı, in-memory DataFrame yönetimi, sinyal tetikleme.
  - Önemli metotlar:
    - `LiveDataManager._sync_symbol_data(symbol)`: Eksik kline’ları Binance’ten alır, DB’ye yazar.
    - `LiveDataManager._update_and_process_symbol(symbol, kline_data)`: Yeni barı DF’e ekler, indikatörleri günceller, Redis’e yazar ve sinyal işleme görevini tetikler.
    - `LiveDataManager._process_signal_for_symbol(symbol)`: İlgili sembol için sinyal üretim/zenginleştirme sürecini çağırır.

- **`indicators/core.py`**
  - Görev: RSI, MACD, ADX, ATR vb. teknik indikatör hesapları.
  - `add_all_indicators(df)`: Verilen DataFrame üzerine gerekli tüm indikatör kolonlarını ekler.

- **`signals/signal_engine.py`**
  - Görev: Teknik kurallara dayalı sinyal üretimi (async).
  - Örnekler: RSI crossover, MA200 crossover/level.
  - Çıktı: Standartlaştırılmış sinyal sözlükleri (`signal_type`, `signal_time`, `price`, `strength`, `pullback_level`, ilgili indikatör değerleri).

- **`signals/signal_processor.py`**
  - Görev: `signal_engine` ile üretilen sinyalleri finansal metriklerle (alpha, beta, Sharpe, Sortino vb.) zenginleştirir ve DB’ye yazar.
  - `process_and_enrich_signals(symbol, df, ref_df, interval)`: Metrikleri hesaplar, sinyalleri zenginleştirir, `database.crud.create_signal` ile kaydeder (UPSERT).

- **`database/models.py`**
  - `PriceData`: OHLCV + temel indikatör alanları, birincil anahtar `(symbol, timestamp)`.
  - `Signal`: Sinyal detayları; benzersiz kısıt `(symbol, signal_time, signal_type, interval)`.

- **`database/crud.py`**
  - `bulk_insert_price_data(symbol, df)`: Kline toplu UPSERT.
  - `create_signal(signal_dict)`: Tekil sinyal UPSERT.
  - `get_last_timestamp(symbol)`: Son kline timestamp.

## Sinyal Akışı (Uçtan Uca)

1. **Eksik veri senkronizasyonu** (`LiveDataManager._sync_symbol_data`)
   - `get_last_timestamp(symbol)` ile son kayıt çekilir.
   - Binance’ten eksik kline’lar alınır ve `bulk_insert_price_data` ile DB’ye UPSERT yapılır.

2. **Canlı bar güncelleme** (`LiveDataManager._update_and_process_symbol`)
   - Yeni kapalı bar in-memory DF’e eklenir, 1000 satıra kadar tutulur.
   - DF üzerine `add_all_indicators` ile indikatörler güncellenir.
   - Güncel DF Redis’e yazılır (hızlı erişim için).
   - Sinyal işleme async görev olarak tetiklenir (`_process_signal_for_symbol`).

3. **Sinyal üretimi** (`signals/signal_engine.py`)
   - En güncel kapalı barlar kullanılarak aktif kural(lar) hesaplanır (örn. RSI/MA200 crossover).
   - Her kural için standart sinyal sözlüğü üretilir.

4. **Zenginleştirme ve kaydetme** (`signals/signal_processor.py`)
   - Finansal metrikler: alpha, beta, Sharpe, Sortino, bilgi oranı vb. hesaplanır (gerekirse referans seri `ref_df`).
   - Sinyal sözlüğü bu metriklerle zenginleştirilir.
   - `create_signal` ile `Signal` tablosuna UPSERT yapılır (tekrar eden sinyal aynı anahtarla güncellenir).

## Veritabanına Kaydedilen Başlıca Alanlar

- **`PriceData`** (örnek): `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, seçili indikatör kolonları (örn. `rsi`, `macd`, `adx`, `atr`, hareketli ortalamalar).
- **`Signal`** (örnek):
  - Kimlik: `symbol`, `signal_time`, `signal_type` (Long/Short vb.), `interval`.
  - Piyasa: `price`, `pullback_level`, `strength`.
  - İndikatörler: `rsi`, `macd`, `adx`, `atr`, `momentum`, vb.
  - Finansal metrikler: `alpha`, `beta`, `sharpe`, `sortino`, `information_ratio` vb.
  - Analiz alanları (opsiyonel): performans izleme için ek kolonlar.

## Notlar ve Genişletme Noktaları

- **Zamanlama**: Kısmi (henüz kapanmamış) barlar sinyal için kullanılmaz; kapanmış bar güvenilirlik sağlar.
- **Dayanıklılık**: `_sync_symbol_data` için segmentli fetch + retry/backoff iyileştirmesi önerilir.
- **Yeni sinyaller**: `SignalEngine` içine yeni yöntem ekle, `calculate_all_signals`’a kaydet; `process_and_enrich_signals` zenginleştirmeyi otomatik uygular.
- **UPSERT semantiği**: Tekrarlanan sinyal anahtarında güncelleme yapılır, böylece idempotent işlem akışı korunur.

## Hızlı İzleme

- Veri -> `PriceData`
- İndikatör -> `indicators/core.py`
- Sinyal -> `signals/signal_engine.py`
- Zenginleştirme + Kayıt -> `signals/signal_processor.py` + `database/crud.py`
- Orkestrasyon -> `live_data_manager.py`
