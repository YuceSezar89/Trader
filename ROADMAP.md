# Mimari Yol Haritası

> Hedef: DB=kaynak, Redis=cache, PgBouncer=bağlantı katmanı. Binance IP ban sorununu kökten çöz.

---

## Faz 1 — Altyapı (Sistemi çalışır hale getir)

- [x] **1. PgBouncer**: `pool_mode = session` → `transaction`, `default_pool_size = 25` → `50`
- [x] **2. Redis bağlantı havuzu**: text `100` → `500`, binary `50` → `200`
- [x] **3. Redis TTL**: buffer süreleriyle eşitle (1m→86400s, 5m→259200s, 1h→604800s, 4h→1209600s, 1d→2592000s)
- [x] **4. IP ban kontrolü**: Binance ban kalktı mı doğrula, servis başlat, hata yok mu kontrol et

---

## Faz 2 — Startup Mimarisi (Kök neden fix)

- [x] **5. 1m veriyi DB'den yükle**: `_load_symbol_all_timeframes` → 1m için Binance yerine PostgreSQL
- [x] **6. 5m/15m aggregator'dan üret**: 1m buffer'dan aggregate (0 Binance isteği); 1h Binance'te kalır (MA200 için 250 bar lazım)
- [x] **7. Cache-first strateji**: 1h/4h/1d için önce Redis cache bak, cache miss varsa Binance'e git (restart'ta sıfırdan çekmez)
- [x] **8. Batch delay**: MTF batch `3s` → `5s` (10×3TF×2weight=60/batch → 720 weight/dk, limit 1200)
- [x] **9. sync_historical arka plana al**: WebSocket açıldıktan 30s sonra başlar, semaphore `20` → `2`; batch delay cache hit'te 0.5s, Binance'te 5s

---

## Faz 3 — Veri Kalıcılığı

- [x] **10. 4h + 1d DB'ye yaz**: WebSocket kapanışında `bulk_insert_price_data` ile yaz (Redis düşse kayıp olmaz)
- [x] **11. Startup'ta 4h/1d DB'den yükle**: Redis miss → DB → Binance zinciri; DB dolunca Binance'e gerek kalmaz

---

## Faz 4 — Performans

- [ ] **12. İndikatör incremental hesaplama**: ertelendi — 250 bar vectorized hesaplama zaten ~20ms, kritik değil

---

## Tamamlananlar

_(her adım bitince buraya taşı)_

---

**Son güncelleme:** 2026-06-09
