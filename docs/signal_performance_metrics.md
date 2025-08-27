# Sinyal Performans Metrikleri ve Yaşam Döngüsü (Özet)

Bu doküman iki ana soruyu pratik şekilde cevaplamak için tasarlandı:

- Sinyal yönünde fiyat hareketi geldi mi? Ne kadar hızlı geldi?
- Aynı anda aynı sinyali veren coinler arasında hangisi ayrıştı?

## 1) Yön Teyidi (Confirmation)

- "Yön imzası": Long=+1, Short=-1. Tüm getirileri bu işaretle çarparız ki tek eksende değerlendirilebilsin.
  - ret_1_dir = sign * ret_1bar
  - ret_3_dir, ret_5_dir, ret_10_dir
- Teyit: ret_N_dir > 0 ise sinyal yönünde hareket var.
- Time-to-confirm: Kümülatif yönlü getiri (cum_ret_dir) ilk kez > 0 olduğu bar sayısı.

Örnek (Long, giriş 100):
- 1. bar kapanışı 101 → ret_1 = +1% → ret_1_dir = +1% → teyit var
- 3. bar kümülatif 102.5 → cum_ret_3 ≈ +2.5% → ret_3_dir ≈ +2.5%
- time_to_confirm = 1 bar

## 2) Ayrışma (Cohort Ranking)

- Cohort: Aynı zaman penceresinde (aynı bar veya ±1 bar), aynı sinyal tipi (Long/Short), aynı interval.
- Karşılaştırma metrikleri: ret_3_dir, ret_5_dir, vol_adj_ret_3/5
- Çıktılar:
  - rank_N: Cohort içinde sıralama (örn. 2/10)
  - pct_N: Percentile (örn. %90 → üst %10’luk dilim)
  - z_N: Z-score (olağandışı hareket ölçümü)

Örnek:
- Aynı dakikada 10 coin Long sinyal verdi. Sizin coin ret_5_dir=+2.2% → rank 2/10, pct ≈ %90.

## 3) Risk-Ayarlı Büyüklük ve Yolculuk Kalitesi

- Volatilite ayarlı getiri:
  - vol_adj_ret_N = ret_N_dir / ATR_N (veya kısa ufuklarda realized vol)
- MFE/MAE (N bar ufukta):
  - MFE% (Maximum Favorable Excursion): p_dir * (max(high) - entry)/entry*100
  - MAE% (Maximum Adverse Excursion): p_dir * (min(low) - entry)/entry*100
- Time-to-target/stop: Belirlenen hedef/stop’e kaç barda ulaşıldı (opsiyonel eşiklerle)

Örnek (Long, giriş 100; 5 bar): En yüksek 103 → MFE +3%, en düşük 99 → MAE -1%.

## 4) Confluence Skoru (P-V-M)

- P (Price), V (Volume), M (Momentum) bileşenlerini normalize edip ağırlıklandır:
  - score = wP*norm(ret_dir) + wV*norm(volume_delta_dir) + wM*norm(momentum_delta_dir)
  - Ağırlıklar `Config.VPM['WEIGHTS']` ile hizalanır. İstenirse MTF bonusu eklenir (`Config.VPM['MTF']`).

## 5) Benchmark-Nötr (Opsiyonel)

- BTCUSDT’e göre rölatif etki (alpha):
  - alpha_N = ret_N_dir - beta * btc_ret_N
  - beta: yakın geçmişte kovaryans/variance ile ölçülür.

## 6) Rejim ve MTF (Opsiyonel)

- Rejim etiketleri: trending/ranging ve vol düşük/orta/yüksek (MA200 eğimi, ADX, realized vol).
- MTF uyum: üst TF yön ve momentum ile hizalanma (1, 0, -1 veya sürekli skor).

## 7) Yaşam Döngüsü Kuralları (Lifecycle)

- Her sinyal için N-bar değerlendirme penceresi başlat (örn. 5 bar) ve metrikleri güncelle.
- Erken bitirme koşulları:
  - reversal: ters sinyal gelirse mevcut sinyali reversal ile kapat (time_to_reversal, reversal_signal_id).
  - target/stop: eşik tetiklenirse kapat (time_to_target/stop).
  - timeout: horizon dolarsa kapat.
- Aynı yönde yeni sinyal:
  - Supersede: Eski sinyali `superseded_by` ile kapat, yeni pencere aç (basit ve anlaşılır).
  - (Alternatif) Chain: Zincir mantığıyla ilişkilendir, ama raporlama karmaşıklaşır.

## 8) Uygulama – Minimal Başlangıç Seti

- Yeni/perf kolonları (Signal tablosu):
  - perf_ret_1_dir, perf_ret_3_dir, perf_ret_5_dir
  - perf_vol_adj_ret_3, perf_mfe_5, perf_mae_5, perf_time_to_confirm
  - perf_rank_5, perf_pct_5, (ops.) perf_pvm_score
- Hesaplayıcı (signal_performance_tracker.py):
  - Yön imzası + ret_N_dir + time_to_confirm
  - vol_adj_ret_3
  - MFE/MAE_5
  - Cohort rank/pct (aynı time window + sinyal tipi + interval)

## 9) Panelde Gösterim Önerisi

- Sinyal satırı: ✓ teyit (1 bar), ret_3_dir: +1.8%, vol_adj_ret_3: 0.9, MFE/MAE(5): +3.0%/-1.0%
- Cohort: rank 2/10 (pct %90)
- Status/Reason: active/completed; reversal/target/stop/timeout/superseded

---
Bu minimal set, “yön teyidi geldi mi ve ne kadar hızlı geldi?” ile “aynı sinyali verenler içinde ayrışma” sorularını doğrudan yanıtlar. İleride alpha/beta, rejim ve MTF uyum metrikleri eklenerek zenginleştirilebilir.
