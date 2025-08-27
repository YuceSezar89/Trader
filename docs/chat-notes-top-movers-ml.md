# Top Movers + ML Sohbet Özeti

Tarih: 2025-08-24 12:06 (+03:00)
Proje: TRader-Panel-ASYNC

## Kısa Özet
- Amaç: Fiyat, hacim ve momentumun yüzde değişimlerine bakarak aynı bar anında en çok ayrışan (top/bottom movers) sembolleri bulmak ve bu seçimi ML ile iyileştirerek sinyal başarısını artırmak.
- Yaklaşım: Önce kuralsal bir baseline (cross-sectional z-score ile Top-K sıralama), sonra ML modeli (XGBoost/RandomForest) ile r_diff tahmini ve Top-K seçim kalitesi artışı.

## Top Movers Mantığı (Ne yapılıyor?)
- Aynı zaman diliminde, aynı timeframe’de, tüm semboller için fiyat/hacim/momentum yüzde değişimleri hesaplanır.
- Cross-sectional z-score veya rank ile pozitif/negatif en çok hareket edenler seçilir.
- Kurallarla doğrudan filtrelenebilir veya ML ile özelliklerden bir skor öğrenilir.
- Hedef: Daha iyi giriş seçimi, false-positive’leri eleme, Top-K kalitesini artırma.

## ML Ne Katar?
- Ağırlık öğrenme: Fiyat vs hacim vs momentum etkileri veriden öğrenilir.
- Bağlam farkındalığı: Rejim, volatilite, likiditeye göre dinamik kararlar.
- Sıralama kalitesi: Cross-sectional regresyon ile r_diff tahmini; precision@K ve avg r_diff@K artışı.
- Genelleme: Outlier/pump anlarında daha dengeli seçim.

## Kritik Noktalar (Kaçırılmamalı)
- Kohort ve senkron: Aynı bar, aynı sinyal tipi/yön/timeframe ile kıyaslama.
- Normalize/ölçekleme: Salt % değişim yerine z-score/ATR normalize edilmesi.
- Likidite/spread filtreleri: Düşük hacimli enstrümanların yanıltıcı etkisi.
- Outlier yönetimi: Winsorization/clip ile kuyrukları bastırma.
- Rejim bağımlılığı: Boğa/ayı/yan piyasa; rejim feature’ı faydalı.
- Hedef tanımı: Kohort ortalamasına göre gelecek H mum getirisi farkı (r_diff) önerilir.

## Kod İncelemesi: `utils/financial_metrics.py`
- Zaman/indeks: UTC→Europe/Istanbul dönüşümü; `date` yoksa guard-path ile metrikler 0’a set ediliyor.
- Alpha/Beta: `ref_df` varsa rolling beta, TradingView tarzı alpha (ret2 - retb2*beta)*100.
- Yüzdesel değişimler ve türevleri:
  - Fiyat: `percent_price_diff` → `normalized_price_diff` → `smoothed_price_diff`
  - Hacim: `percent_volume_diff` → `normalized_volume_diff` → `smoothed_volume`
  - Volatilite (high-low): `percent_volatility_diff` → `normalized_volatility_diff` → `smoothed_volatility`
  - Momentum (L=14): `percent_momentum_diff` → `normalized_momentum_diff` → `smoothed_momentum`
  - RSI (L=14): `percent_rsi_diff` → `normalized_rsi_diff` → `smoothed_rsi`
- Kümülatif gün içi referans: `cumulative_change` ve normalize/smooth varyantları.
- Risk/performans metrikleri: Sharpe, Sortino, Calmar, Omega, (opsiyonel) Treynor, Information Ratio; kompozit skorlar (`composite_ratio`, `scaled_avg_normalized`).
- Basit sinyal izlekleri: `buy_crossover`/`sell_crossunder`, durum sayaçları (`buy_mum_sayisi`, `sell_mum_sayisi`).

### Eksik Olanlar (Top Movers + ML için)
- Cross-sectional karşılaştırma (kohort bazında z-score/rank üretimi).
- Outlier/winsorize yardımı.
- Likidite ve trade edilebilirlik filtreleri (örn. USD hacim eşiği).
- Rejim feature’ları (MA200 konumu, ATR persentil, trend/gürültü oranı).
- Tek noktada "feature vektörü" üretip ML boru hattına besleyen builder.

## Önerilen Yardımcı Modül
- Konum: `utils/cross_section.py` (öneri)
- Fonksiyonlar:
  - Cross-sectional z-score/rank hesaplayıcı (aynı timestamp’te tüm semboller için).
  - Winsorization/clip yardımcıları (p=0.01/0.02 varsayılan).
  - Likidite filtreleri (son N bar USD hacim ortalaması vb.).
  - `mover_score` kompoziti (ör. price 0.5, volume 0.2, momentum 0.2, rsi 0.1 ağırlık).

## POC Planı (Kuralsal Baseline → ML)
1) Metrikler (POC): `smoothed_price_diff`, `smoothed_volume`, `smoothed_momentum`, `smoothed_rsi`.
2) Kohort z-score ve `mover_score` (eşit veya önerilen ağırlıklar: 0.5/0.2/0.2/0.1).
3) Filtreler: Likidite eşiği + winsorize.
4) Top-K seçimi: Pozitif ve negatif için ayrı K (örn. 5 ve 5).
5) Gözlem/Doğrulama: Panelde listele, temel istatistikler.
6) ML’e geçiş: r_diff regresyonu ile Top-K sıralamasını model skoru ile yap.

## ML Entegrasyonu (Hafıza ile uyumlu plan)
- Hedef (target): Kohort ortalamasına göre gelecek H mum getirisi farkı (r_diff) veya sınıflandırma.
- Özellikler: Yukarıdaki metrikler + rejim/likidite + composite skorlar + multi-timeframe özetleri.
- Model: XGBoost/RandomForest baseline; metrikler: precision@K, avg r_diff@K, PR-AUC.
- Entegrasyon: `signals/signal_processor.py` içinde canlı inference ve sıralama.
- Veri hattı: `live_data_manager.py` (WS) + Redis cache; DB’den sinyal zamanlı örnek üretimi.

## Karar/Netleştirme Gerektirenler
- Kohort kapsamı: Hangi semboller/timeframe?
- Ağırlıklar: Eşit mi, yoksa 0.5/0.2/0.2/0.1 mi?
- Likidite eşiği: Son 50 bar ort. USD hacim > ?
- Outlier politikası: Winsor p=0.01/0.02 makul mu?
- Top-K: Pozitif/negatif için K değerleri?

## Sonraki Adımlar (Öneri)
- `utils/cross_section.py` modülünü oluştur (
  z-score, winsorize, likidite, mover_score).
- Kuralsal baseline’ı çalıştır ve panelde "Top Movers (Kohort)" olarak göster.
- Sonuçları gözlemle; ardından ML eğitim veri hazırlama scriptine geç.

---
Bu dosya, sohbetimizin özlü bir derlemesidir. Detaylara dönmek istediğinde başlıklar üzerinden hızlıca göz atabilirsin.
