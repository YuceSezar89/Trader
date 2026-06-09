# Mimari Yol Haritası

> Hedef: DB=kaynak, Redis=cache, PgBouncer=bağlantı katmanı. Binance IP ban sorununu kökten çöz.

---

## Faz 1 — Altyapı (Sistemi çalışır hale getir)

- [x] **1. PgBouncer**: `pool_mode = session` → `transaction`, `default_pool_size = 25` → `50`
- [x] **2. Redis bağlantı havuzu**: text `100` → `500`, binary `50` → `200`
- [x] **3. Redis TTL**: buffer süreleriyle eşitle (1m→86400s, 5m→259200s, 1h→604800s, 4h→1209600s, 1d→2592000s)
- [ ] **4. IP ban kontrolü**: Binance ban kalktı mı doğrula, servis başlat, hata yok mu kontrol et

---

## Faz 2 — Startup Mimarisi (Kök neden fix)

- [ ] **5. 1m veriyi DB'den yükle**: `_load_symbol_all_timeframes` → 1m için Binance yerine PostgreSQL
- [ ] **6. 5m/15m/1h aggregator'dan üret**: startup'ta `_aggregate_and_cache_mtf()` çağır (0 Binance isteği)
- [ ] **7. Binance'ten sadece 4h + 1d + 1m gap**: eksik barları tespit et, sadece onları çek
- [ ] **8. Batch delay**: MTF batch `3s` → `5s` (rate limit güvenliği)
- [ ] **9. sync_historical arka plana al**: startup'tan çıkar, `asyncio.create_task` ile 5dk sonra çalıştır, semaphore `20` → `2`

---

## Faz 3 — Veri Kalıcılığı

- [ ] **10. 4h + 1d DB'ye yaz**: WebSocket'ten kapanış gelince `bulk_insert_price_data` ile kaydet
- [ ] **11. Startup'ta 4h/1d DB'den yükle**: Faz 2 tamamlandıktan sonra, 4h/1d de Binance'e gerek kalmaz

---

## Faz 4 — Performans

- [ ] **12. İndikatör incremental hesaplama**: her 1m kapanışında full recalc → son `200 + buffer` bar ile hesapla

---

## Tamamlananlar

_(her adım bitince buraya taşı)_

---

**Son güncelleme:** 2026-06-09
