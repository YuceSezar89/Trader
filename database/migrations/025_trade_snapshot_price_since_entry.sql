-- ============================================================
-- Migration 025: trade_snapshots'a price_since_entry_pct eklendi
--
-- Hoca-kaynaklı VPMV referanslarında (bkz. sohbet, 19-20 Tem 2026) P bileşeni
-- since-signal kümülatif hesaplanıyor (Supertrend flip'te sıfırlanan pullback %).
-- Bizim sistemimizde "sinyal" = işlemin kendi giriş anı — Supertrend'e gerek yok.
-- price_since_entry_pct = (fiyat - entry_price) / entry_price × 100, yön-ayarlı
-- (Long'da pozitif=kâr yönü, Short'ta pozitif=kâr yönü — Hoca'nın net_price
-- konvansiyonuyla aynı). Mevcut price_score (rolling-window, bar-bar) YERİNE
-- değil, YANINDA — iki farklı "fiyat" tanımı ayrı ayrı izleniyor.
-- ============================================================

ALTER TABLE trade_snapshots ADD COLUMN IF NOT EXISTS price_since_entry_pct DOUBLE PRECISION;
