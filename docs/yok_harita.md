# V-P-M Onaylı Sinyal Yol Haritası

Bu doküman, basit sinyallerin (AL/SAT) hacim (V), fiyat yüzdesi (P) ve momentum (M) ile onaylanması ve aynı sinyali verenler arasından seçim/sıralama yapılması için kısa yol haritasını içerir.

## Hedef
- Basit sinyali, üçlü onay (V-P-M) ile filtrele.
- Aynı sinyali verenler arasında V-P-M skoru ile sıralayıp en iyileri seç.

## Ölçümler (aynı bar veya kısa lookback)
- Fiyat % (P): (close/close[-1] - 1) × 100 veya ROC_N
- Hacim (V): volume z-skoru (tercih) veya hacim % değişimi
- Momentum (M): RSI değişimi (RSI_delta) veya MACD histogram değişimi (MACD_hist_delta)

## Onay Kuralları
- Basit AND: V ve P ve M eşiği geçti → ONAYLI
- 2/3 Kuralı (önerilen başlangıç): 3 metrkten en az 2’si eşiği geçti → ONAYLI
- Ağırlıklı Skor (sıralama için):
  - score = 0.4·P + 0.3·V + 0.3·M (AL için pozitif yön, SAT için işaretler ters)

## Başlangıç Eşikleri (öneri)
- P: |% değişim| ≥ 0.3
- V: volume_z ≥ 1.0
- M: RSI_delta ≥ +2 (AL) / ≤ −2 (SAT) veya MACD_hist_delta ≥ +0.5 (AL) / ≤ −0.5 (SAT)
- Onay modu: 2/3
- Sıralama: skor desc, eşitlikte `scaled_avg_normalized` desc

## Entegrasyon Noktaları
- config.py: eşikler, ağırlıklar, onay modu, lookback
- signals/signal_processor.py (process_and_enrich_signals):
  - P/V/M metriklerini hesapla (lag/rolling ile sızıntısız)
  - Onay kararını ver (confirmed) ve `vpms_score` üret
- app.py (panel):
  - “Sadece onaylı sinyaller” filtresi
  - Varsayılan sıralama: `vpms_score` desc → `scaled_avg_normalized` desc

## Seçim/Portföy Kuralları
- Aynı sinyali verenler → `vpms_score` yüksek olanlar öncelikli
- Eşitlik kırıcılar: `scaled_avg_normalized` (kalite), hacim, ATR/spread (risk)

## Opsiyoneller
- MTF onay: bir üst zaman diliminde momentum yönü uyumluysa skora küçük ağırlık ekle
- Winsorize/outlier kırpma: uç değerlerde daha stabil skor

## Gelecek: ML ile Onay (özet)
- Hedef: sinyal yönüyle uyumlu N bar sonrası getiri > eşik → 1/0
- Özellikler: P/V/M + teknik ve finansal metrikler (`utils/financial_metrics.py`)
- Modeller: LogReg, RF, XGBoost; zaman serisi CV; canlıda `ml_score`/`ml_confirmed`

## Notlar
- Sızıntı önleme: tüm metrikler geçmiş veriye bakmalı (shift/rolling)
- Semboller arası adalet için z-score tercih edilebilir

## Mimari Akış ve Bileşenler
- **WebSocket veri toplama (`live_data_manager.py`)**: Binance Futures akışı. Çoklu timeframe desteklenir (örn. 1m/5m/15m/1h).
- **Redis cache**: Son N bar ve son metrik/snapshot; key şeması `symbol:tf:*`.
- **Metrik üretimi (`utils/financial_metrics.py`)**: P/V/M ve kalite metrikleri (Sharpe, Treynor, Information, scaled_avg_normalized).
- **Sinyal motoru (`signals/signal_engine.py`)**: RSI/MA200/C20MX sinyalleri.
- **Onay ve zenginleştirme (`signals/signal_processor.py`)**: V-P-M onayı, `vpms_score`, ek alanlar.
- **DB (`database/`)**: Kalıcı kayıt; panel sorguları buradan.
- **Panel (`app.py`)**: Filtre/sıralama, “sadece onaylı” görünümü.

## Tetikleyiciler
- **Bar kapanışı**: WS → Redis’e yeni bar; processor V-P-M onayı yapar ve DB’ye yazar.
- **Manuel tarama**: Panelden tetik; aynı akış çalışır.

## MTF Entegrasyonu (özet)
- Veri: Tüm TF’ler paralel WS; Redis’te `symbol:tf` saklama.
- Onay: Alt TF sinyal + üst TF momentum uyumu (opsiyonel ağırlık ekle).
- Panel/DB: TF alanı mevcut; `mtf_score` alanı opsiyonel olarak eklenebilir.
