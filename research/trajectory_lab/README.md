Trajectory Lab is an exploratory research tool.
It does not validate hypotheses.
It does not generate production features.
Its only purpose is to shorten the time required to discover useful hypotheses.
Every hypothesis generated here must be validated independently by Pattern Lab before being considered for production.

---

# Trajectory Lab

Pattern Lab'ın yanına eklenen ikinci araştırma katmanı. Pattern Lab bir
hipotezin **doğru olup olmadığını** (kronolojik split, placebo, look-ahead
kontrolü, outlier analizi, OOS doğrulaması) test eder — bu metodoloji
**değişmiyor**. Trajectory Lab ise bir hipotezin **neden** doğru ya da
yanlış olabileceğini, metriklerin zaman içindeki DAVRANIŞINA bakarak
keşfeder — tek bir snapshot değer (ör. "EVOL=18") değil, o değere nasıl
gelindiği (ör. "70→55→40→25→18, ivmelenerek düşüyor").

## Akış

```
Gözlem → Görselleştirme → Davranış keşfi → Hipotez → Pattern Lab doğrulaması → Canlı sistem
```

## Tasarım ilkeleri (v1)

- **Sadece görselleştirme/keşif aracı.** Kümeleme, motif madenciliği, HMM
  faz-segmentasyonu gibi algoritmalar v1 kapsamında DEĞİL — önce
  araştırmacının gözünü güçlendiren araçlar (overlay, ortalama+bant,
  çoklu-metrik sinyal hikâyesi), sonra (gerekirse) algoritma.
- **Production koduna bağımlılık yok.** `metrics.py`'deki her fonksiyon,
  ilgili production formülünün (EVOL/VPMV/VP Score/CVD Slope)
  bağımsız bir kopyasıdır — `indicators/core.py`, `utils/vpmv.py`,
  `signals/market_context.py` gibi canlı modülleri İMPORT ETMEZ. Neden:
  (a) production formülü değişirse geçmiş araştırma sessizce
  etkilenmesin, (b) araştırma kodu canlı sinyal mantığına asla
  bağımlı olmasın.
- **Pattern Lab'ın metodolojisi değişmiyor.** Trajectory Lab sadece
  hipotez üretir; her bulgu, canlıya gitmeden önce Pattern Lab'ın kendi
  kapılarından (placebo, split-period, look-ahead vb.) geçmek zorunda.
- **Provider ≠ corpus kapsamı.** `metrics.py`'de TÜM adaylar (Price Return,
  EVOL, VPMV, CVD Slope, VP Score) eşit muamele görür — hangisinin
  "değersiz" olduğuna dair önyargılı bir eleme yapılmaz. `config.py`'de
  hangi metriklerin varsayılan corpus'a dahil olacağı ayrı bir karardır
  (bkz. aşağıda), tek satırla değiştirilebilir.
- **Feature değil, dimension (4 Ağustos'tan itibaren).** Pattern Lab'ın
  amacı büyüyen bir indikatör koleksiyonu değil, piyasa durumunu (market
  state) az sayıda bağımsız EKSENLE tanımlayan bir motor kurmaktır —
  "Market State Engine". `EVOL Strength × EVOL Persistence` örneği
  (aşağıya bkz.) bunu somutlaştırdı: ikisi ayrı iki "özellik" değil, aynı
  EVOL davranışının iki ekseni. Bu yüzden bu dokümanda artık mümkün
  olduğunca **dimension/boyut** terimi kullanılıyor — "metrik" hâlâ
  `metrics.py`'deki HAM formülü/provider'ı işaret eder (ör. EVOL), ama bir
  metrikten birden fazla bağımsız boyut türeyebilir (ör. EVOL → Strength
  + Persistence). "Feature Acceptance Pipeline" ismi (kurulu, çapraz
  referanslı bir süreç adı) değişmedi, ama içeriği artık boyut adaylarını
  değerlendiriyor.

## v1 çekirdek metrikleri

Aktif (varsayılan corpus): **Price Return, EVOL, VPMV, CVD Slope, CVD Level, z_score**.
Hazır-bekleyen (provider olarak var, corpus'ta kapalı): **VP Score** — EVOL
ile aynı ham girdiyi (fiyat+hacim) paylaşıyor ve CVD Slope ile aynı
alış/satış hacim vekilinden türüyor; şimdilik teorik gerekçeyle değil,
gerçek overlay/mean-band bulguları CVD Slope'tan farklı davrandığını
gösterirse etkinleştirilecek.

VPMV'nin alt bileşenleri (vol/mom/vlt/price_score) ve devisso_score/delta
v1'de corpus'a DAHİL DEĞİL — ilki sadece `viz.plot_signal_story()` ile
istek üzerine (drill-down) hesaplanır, ikincisi EVOL ile aynı aileden
("Δfiyat / osilatör" oranı) olduğu için ayrıca eklenmedi.

### Accepted (Promoted) — production corpus'unda active=True

4 Ağustos'tan itibaren kabul kriteri değişti (bkz. `RESEARCH_SOP.md`
Stage 1/Stage 2 ayrımı): "istatistiksel olarak anlamlı mı?" yetmiyor,
"production'a girecek kadar DEĞERLİ mi?" (ΔAUC ≥ 0.003 pratik eşiği)
sorusu da sorulmalı. Bu eşiği geçenler:

| Tier | Boyut | Cohen's d | Mutual Information | ΔAUC (mevcut sete katkı) | Zamanlama |
|---|---|---|---|---|---|
| **1** | EVOL Strength (seviye, EVOL(t=4)) | 1.90 | 0.284 | (temel boyut) | post-sinyal (t=+4) |
| **1** | EVOL Persistence (`evol_age`, Strength'in AYNI ekseni — 2D okunur, tek başına doğrusal kural değil) | 0.430 | 0.112 | +0.015 ile +0.024 arası (3 indikatörde) | post-sinyal (t=+4 civarı) |
| **2** | VPMV | 0.97 | 0.149 | (temel boyut) | post-sinyal (t=0) |
| **3** | CVD Level | 0.385 | 0.076 | +0.0053 ≥ 0.003 ✓ | post-sinyal (t=+5) |

**5 Ağustos:** temel taşlar (EVOL Strength+Persistence, VPMV, CVD Level)
artık SABİT kabul ediliyor — kullanıcı kararı: *"Artık yeni metrik
peşinde koşmuyoruz. Her seferinde tek bir piyasa fikrini test edeceğiz...
Her yeni fikir, ancak gerçek işlem performansına katkı gösterirse
sisteme girecek."* Bundan sonraki odak: eksik kalan piyasa davranışlarını
tek tek doğrulamak + hangi piyasa ortamında çalıştıklarını anlamak
(Stage 3 disiplini) — yeni boyut avcılığı değil.

### Research Accepted — HENÜZ Production DEĞİL (Stage 2 + stability geçti, config.py'de active=False)

7 Ağustos, "Sinyal Mumu Anatomisi" çalışmasından (bkz. `CONTEXT_LAB_STATUS.md`):

| Boyut | Tanım | Cohen's d | ΔAUC (baseline=EVOL+VPMV+CVD Level) | Bootstrap %95 CI | Zaman-içi stabilite |
|---|---|---|---|---|---|
| **close_pos** | Sinyal barının kapanışının kendi range'i içindeki konumu, `(close-low)/(high-low)*100`, HAM (percentile-rank/normalize edilmedi) | **1.02** (evol=1.90'dan sonra 2., vpmv=0.996'dan güçlü) | **+0.0079** | [+0.0058, +0.0103] | **7/7 dilim pozitif, TÜMÜ CI sıfırı dışlıyor** (body_size_pct'in 1/7 belirsiz dilimine kıyasla daha temiz) |

Aile-bazlı doğrulama (HA_Cross_Long/RSI_Cross_Long/Supertrend_Long, hem
Stage 2 hem 8-dilimlik stability testinde AYRI AYRI): ΔAUC 3 ailede de
+0.010 ile +0.012 arası, 7 dilim × 3 aile = 21 hücrenin HEPSİ pozitif,
istisnasız. `stage2_pctrank.py`/`stage2_anatomy.py`'nin AYNI standart
bataryası (walk-forward + 1000-permütasyon placebo + bootstrap CI).

`buy_pct` (gerçek taker buy/sell oranı) tek başına orta-güçlü göründü
(d=0.66, ham corr(y)≈0.30) ama close_pos'u kontrol edince ΔAUC'si
+0.0004'e (CI sıfırı içeriyor) düştü — **redundant, kapandı, Research
Accepted DEĞİL.**

**Karar (kullanıcı, 7 Ağustos): Research Accepted — bundan sonraki
araştırmalarda close_pos baseline/Accepted feature olarak kullanılabilir**
(örn. yeni bir adayın incremental katkısı test edilirken
evol+vpmv+cvd_level+close_pos baseline'ına eklenebilir). **Production
Accepted İLAN EDİLMEDİ** — config.py'de `active=True` yapılmadı, canlı
corpus'a girmedi. Sıradaki hedef eşik/strateji optimizasyonu DEĞİL:
close_pos'un hangi piyasa rejimlerinde/hangi tamamlayıcı koşullarla
confluence oluşturduğunu araştırmak.

**Kullanıma alma zinciri durumu (8 Ağustos, bkz. RESEARCH_SOP.md
"Kullanıma Alma Zinciri" — Research Accepted ≠ Production Accepted):**

```
RESEARCH:    Research Accepted ✅
MECHANISM:   Under investigation 🟡 (VPMV/volatilite etkileşimi
             keşfedildi, VPMV decomposition sürüyor)
CONFLUENCE:  Tested, Context-dependent 🟡 (VPMV ile ters-U/monoton
             asimetrik etkileşim, walk-forward'da büyük ölçüde
             tekrarlandı — ama "Supported" değil, mekanizma netleşmedi)
STRATEGY:    Not translated ❌
DEPLOYMENT:  Not paper tested ❌ → Production Accepted ❌
```

config.py'ye dokunulmaması ŞU AN DOĞRU — Research Accepted, Mechanism
ve Confluence aşamaları tamamlanıp Strategy'ye somut bir öneriyle
geçilmeden Production'a atlama YOK (Anayasa'yla aynı ağırlıkta kural).

### Accepted set gerçekten birbirini tamamlıyor mu? (4 Ağustos leave-one-out doğrulaması)

Yeni boyut aramak yerine, zaten kabul edilmiş 3 boyutun (EVOL Strength/
VPMV/CVD Level) birbirinin gerçekten tekrarı olup olmadığı ayrıca test edildi
— "Volatility Stability ailesi"nde adaylara uyguladığımız YÖNTEMİN
(korelasyon + permutation importance + leave-one-out) aynısı, bu kez
Accepted set'in kendisine.

**Özellikler arası korelasyon/MI düşük** (evol-vpmv=0.225, evol-cvd_level
=0.104, vpmv-cvd_level=0.146; MI'lar 0.16-0.20 aralığında) — yüksek
örtüşme yok.

**Leave-one-out (walk-forward, tam set AUC=0.9347):**

| Çıkarılan | Kalan AUC | Düşüş | Eşiğe (0.003) göre |
|---|---|---|---|
| EVOL | 0.7616 | **+0.1732** | 58x eşik — vazgeçilmez |
| VPMV | 0.8980 | **+0.0368** | 12x eşik — açıkça gerekli |
| CVD Level | 0.9313 | **+0.0034** | eşiğin hemen üstünde |

3 corpus'ta AYRI AYRI da doğrulandı — CVD Level'in çıkarılma-düşüşü
**her 3 corpus'ta da** 0.003'ü geçiyor (HA_Cross +0.0040, RSI_Cross
+0.0052, Supertrend +0.0047) — yani gerekliliği agregatta değil, her
indikatörde ayrı ayrı da tutuyor.

**Sonuç: redundancy yok, üçü de gerçek/bağımsız bilgi taşıyor.** EVOL
ezici ağırlıkta ana sürücü, VPMV güçlü ikinci sinyal, CVD Level küçük
ama her corpus'ta tutarlı üçüncü katman — hiçbirini çıkarmak mantıklı
olmaz.

### EVOL Strength × EVOL Persistence — EVOL'ün iki boyutu (4 Ağustos)

Aynı gün fark edilen bir soru: EVOL'ün bugüne kadar kullanılan tek
temsili (**EVOL Strength**, kod adı `level4` — divergence-peak bar'ındaki
ham seviye) mi en bilgi yoğun boyut, yoksa CVD Slope→Level hikâyesindeki
gibi burada da "yanlış pencereye mi bakıyoruz"? EVOL'ü statik bir
gösterge değil, zamansal bir süreç olarak ele alıp 7 temsil sistematik
karşılaştırıldı — yeni bir metrik/provider yazılmadan, hepsi zaten
corpus'taki `evol` serisinden türetildi:

**1. Temsil karşılaştırması (kafa kafaya, VPMV+CVD_Level+z_score_pre sabit):**

| Temsil | Tanım | Tek başına AUC |
|---|---|---|
| **EVOL Strength** (`level4`, mevcut şampiyon) | EVOL(t=4) | **0.9385** |
| accel | (EVOL(8)-EVOL(4)) - (EVOL(4)-EVOL(0)) | 0.9024 |
| slope_04 | EVOL(4)-EVOL(0) | 0.8888 |
| slope_48 | EVOL(8)-EVOL(4) | 0.8600 |
| **EVOL Persistence** (`evol_age`) | EVOL(4)'ün 50 etrafında kaç bardır aynı tarafta | 0.8484 |
| phase_peak | [-10,+10] penceresinde EVOL'ün tepe yaptığı bar | 0.8295 |
| shape_mono | EVOL(0..8)'in doğrusal rampayla korelasyonu | 0.7756 |

Sonuç: CVD hikâyesi TEKRARLANMADI — EVOL Strength açık ara en güçlü tek
temsil, hiçbir alternatif onu geçemedi.

**2. Ama hepsini EVOL Strength'e EKLEYİNCE** (yerine değil, yanına)
walk-forward AUC **0.9385 → 0.9591** (+0.0206, eşiğin 7 katı) — yani bu
temsiller şampiyonun YERİNE değil YANINA konunca gerçek ek bilgi taşıyor.

**3. Redundancy/leave-one-out** (6 ekstra temsil arasında): eğim/ivme
üçlüsü (slope_04, slope_48, accel) birbirinin neredeyse tam tekrarı
(matematiksel olarak bağlantılılar) — gerçek katkıları yok. **EVOL
Persistence tek başına diğer 5'in toplamından fazla değer taşıyor**
(leave-one-out düşüşü +0.0078, permutation importance +0.0220/%12.4 gain
— en yakın rakibi phase_peak'in 3 katından fazla).

**4. EVOL Persistence'ın tam Feature Acceptance bataryası** (taban:
mevcut Accepted set EVOL Strength+VPMV+CVD Level):

| Metrik | EVOL Persistence (`evol_age`) | (CVD Level ile kıyas) |
|---|---|---|
| Cohen's d | **0.430** | 0.385 |
| MI | **0.112** | 0.076 |
| Walk-forward AUC | 0.9348 → **0.9534** (+0.0186, std 0.0047→0.0022) | +0.0053 |
| Permutation importance | **+0.0327** (%14.4 gain) | +0.0076 (%2.7) |
| Placebo p-değeri | 0.0000 (null std=0.0004, ~39-sigma) | 0.0000 |
| Bootstrap %95 CI | **[+0.0134, +0.0183]** | [+0.004,+0.007] |
| 3 indikatörde ayrı | +0.0214 / +0.0217 / +0.0241 (hepsi 7x eşik üstü) | ~+0.004 |

CVD Level'in birkaç katı büyüklüğünde, her ölçütte ve her indikatörde
kararlı — Stage 1+2'yi fazlasıyla geçti.

**5. Stage 3 — Mekanizma (EVOL Persistence=6'daki %28→%96 sıçraması):**
Göz testinde win-rate-vs-persistence eğrisi düz DEĞİLDİ — persistence=5'te
%28, persistence=6'da aniden %96.1. Araştırma (indikatör/sembol/hafta
dağılımı + kod mantığının elle doğrulanması + EVOL Strength'in
persistence'a göre kontrolü) bunun bir **artefakt olmadığını**, EVOL
Strength ile EVOL Persistence arasındaki ETKİLEŞİMİN bir sonucu
olduğunu gösterdi: persistence=6 sinyallerinin %99.4'ünde EVOL
Strength≥50 (ort=79.7, YÜKSEK), persistence=4/5'te bu oran sadece
%26-27 (ort=34, DÜŞÜK). Zayıf EVOL genelde 4-5 bardan uzun "hayatta
kalmıyor" (kısa ömürlü), güçlü EVOL genelde 6+ bar kalıcı oluyor
(momentum) — Persistence'ın TEK BAŞINA eğrisi bu kompozisyon
kaymasından dolayı düz değil, ama Strength ile birlikte okununca tam
mantıklı. **Bu, "iki ayrı özellik" değil "aynı davranışın iki ekseni"
çerçevesinin doğduğu an.**

**6. Stage 3 — Karar Haritası** (EVOL Strength × EVOL Persistence, 5×6
hücre, win rate/PF/ortalama getiri/n;
`research/trajectory_reports/evol_age_decision_map.png`):

| Win Rate (%) | Persistence 1-2 | 3-4 | 5-6 | 7-8 | 9-10 | 11-15 |
|---|---|---|---|---|---|---|
| Strength 0-20 | 21.4 | 3.7 | 0.6 | 4.9 | 2.7 | 4.9 |
| Strength 20-40 | 34.8 | 9.8 | 2.4 | 11.2 | 15.5 | 12.9 |
| Strength 40-60 | 48.7 | 37.7 | 67.7 | 69.7 | 58.4 | 57.4 |
| Strength 60-80 | 57.9 | 81.8 | 96.1 | 92.2 | 90.0 | 89.7 |
| Strength 80-100 | 64.9 | 89.8 | **99.1** | 97.5 | 97.4 | 96.4 |

Sınırlar temiz: **Strength<40**'ta Persistence artışı win rate'i
DÜŞÜRÜYOR (kaçın/erken çık bölgesi), **Strength≥60**'ta win rate'i
YÜKSELTİYOR (tut/onayla bölgesi) — PF de aynı deseni doğruluyor
(Strength 0-20/Persistence 1-2: PF=0.29, Strength 80-100/Persistence
5-6: PF=21.7, n=6197).

**7. Stage 4 — Production Rule (5 Ağustos, kullanıcı onayladı, Tier 1'e
EVOL Strength ile birlikte kabul edildi):**

1. **Ne ölçüyor?** EVOL Strength: sinyalden 4 bar sonra hacim-verimliliği
   seviyesi. EVOL Persistence: bu seviyenin 50 orta noktasına göre kaç
   bardır aynı tarafta kaldığı (momentum devamlılığı).
2. **Ne zaman güvenilir?** SADECE birlikte okunduğunda — tek başına
   Persistence (AUC 0.85) Strength'in (AUC 0.94) yanında zayıf kalıyor,
   ama ikisi birlikte PF'yi 0.0'dan 78'e kadar ayırıyor (n=6197,
   gürültü değil).
3. **Bot bunu nasıl kullanacak?** (Pozisyon zaten açık, sinyalden 4 bar
   sonra — bu bir EXIT/hold kararı, giriş filtresi değil):
   - **Strength<40** (persistence fark etmez) → PF 0.0-0.6 → **hemen
     çık** (mevcut `ha_cross_evol_exit`'in EVOL<25 kuralından daha
     erken/geniş bir tetikleyici)
   - **Strength 40-60** → PF 0.7-2.3 → **temkinli tut**, risk artırma
   - **Strength≥60 VE Persistence≥3** → PF 3.7-78 → **tut/güven artır**
     (trailing gevşet, scale-in düşünülebilir)
4. **Hangi durumda kullanılmamalı?** Persistence=15 hücreleri censored
   (pencere sınırı, "en az 15" demek — kesin değer değil). Sadece Long +
   HA_Cross/RSI_Cross/Supertrend'de test edildi — Short'a veya başka
   indikatörlere DOĞRUDAN uygulanmamalı, önce oralarda da doğrulanmalı.

**Durum:** Stage 1/2/3/4 TAMAMLANDI. EVOL Strength+Persistence, tek bir
2D boyut çifti olarak Tier 1'e kabul edildi. Yukarıdaki Production Rule,
Pattern Lab'a taşınacak somut öneri — Pattern Lab kendi metodolojisiyle
(kronolojik split, placebo, look-ahead kontrolü) ayrıca doğrulayacak.

### Research Archive — istatistiksel olarak gerçek, ama pratik eşiği geçemedi

Stage 1'in TÜMÜNÜ (placebo p<0.05, bootstrap CI sıfırı dışlıyor, 3/3
indikatörde tutarlı) geçen, ama Stage 2'nin ΔAUC≥0.003 pratik eşiğinde
kalan boyut adayları. **Silinmez, "başarısız" sayılmaz** — sadece henüz
production'a değmeyecek kadar küçük. İleride Pattern Lab'ın gerçek $ P&L
sonucuyla ya da başka Archive üyeleriyle kombine edilerek yeniden
değerlendirilebilir.

| Aday | Kategori | Cohen's d | MI | ΔAUC | Neden Archive (Accepted değil) |
|---|---|---|---|---|---|
| `z_score` (giriş-öncesi ort., t∈[-10,-1]) | **Entry** (tek giriş-tarafı örneği, EVOL/VPMV/CVD Level'in aksine sinyal ÖNCESİNDE bilinir) | -0.067 | 0.054 | +0.0015 < 0.003 | Placebo p=0.0000, bootstrap CI=[+0.0004,+0.0025] — istatistiksel olarak gerçek ama etkisi çok küçük |
| `regime_age` (volatility rejimi kaç bardır aynı, en basit kural: >70 high/<30 low/normal) | Entry | 0.037 | 0.0067 | +0.0012 (solo) | Placebo p=0.0000, bootstrap CI=[+0.0002,+0.0020] — "Volatility Stability ailesi"nin (bkz. aşağı) en az redundant üyesi, ama yine de eşiğin altında |

### Neden bu eşik (ΔAUC≥0.003)?

Kullanıcının 4 Ağustos gerekçesi: "EVOL/VPMV ciddi sıçratıyor, CVD Level
gerçekten yeni bilgi getiriyor — ama regime_age +0.001 katıyor. Bu gerçek
ve istatistiksel olarak doğru, ama trading tarafında soru şu: bu +0.001
bana gerçekten daha fazla para kazandıracak mı? Bunu henüz bilmiyoruz."
Amaç artık boyut SAYISINI değil, boyut KALİTESİNİ artırmak — p<0.05
olan ama AUC'yi +0.001 artıran boyutlarla sistemi doldurmamak.

## Neden CVD Slope reddedildi, CVD Level neden kabul edildi?

2 Ağustos'ta HA_Cross/RSI_Cross/Supertrend Long+Short corpus'larında **CVD
Slope**'un divergence-peak'i (winner_mean - loser_mean tepe noktası) global
toplamda t=9'da çok tutarlı görünüyordu (6/6 corpus). Ama haftalık ve
sembol-grubu split'lerinde bu bulgu **dağıldı** — büyük örneklemli haftalar
bile t=4/5/9 arasında savruluyordu. Soru şuydu: *CVD verisi gerçekten
bilgisiz mi, yoksa yanlış temsil biçimi mi test ediliyor?*

CVD Slope, ham kümülatif order-flow'un (`(2*bv-volume).cumsum()`) **farkını**
alıp oran'a çeviriyor — EVOL/VPMV gibi 0-100 bantlı, durum/seviye tipi
oscillator'lardan yapısal olarak farklı bir nesne (zaten diff alınmış,
gürültüye eğilimli bir momentum ölçüsü). Aynı ham veriden, farkı almadan,
aynı rolling min-max normalizasyonuyla (VP Score'daki gibi) bir **seviye**
temsili (**CVD Level**, `metrics.py::cvd_level_series`) üretilip AYNI test
bataryasından geçirildi — tek manipüle edilen değişken temsil biçimiydi,
ham girdi birebir aynıydı (kontrollü deney).

Sonuç: CVD Level, haftalık split'te 20 dilimin 18'inde t=4-6 bandında sıkı
kümelendi, sembol-grubu split'inde büyük örneklemli (alt-coin) grupta 3/3
tutarlı çıktı — CVD Slope'un düşürüldüğü testleri geçti. **Sonuç: sorun
CVD verisinde değil, türev/oran temsilindeydi.**

Bu ayrım gelecekte önemli: yeni bir metrik provider'ı ilk denemede
"dağınık" görünürse, hemen elenmemeli — önce "yanlış temsil biçimi mi
test ediliyor?" sorusu sorulmalı (bkz. aşağıdaki Feature Acceptance
Pipeline'ın 2. adımı).

## Feature Acceptance Pipeline

(İsim tarihsel olarak korundu — bkz. yukarıdaki "Tasarım ilkeleri";
değerlendirdiği şey artık "feature" değil **boyut** adayları.) Bu bölüm,
bir boyut adayının KABUL/RED kararına giden istatistiksel bataryayı
anlatır. Bir adayın buraya girmeye DEĞER olup olmadığına (hipotez,
provenance/kirlilik kontrolü, warmup kontrolü, göz testiyle erken eleme)
karar veren süreç ayrı bir dokümanda: **`RESEARCH_SOP.md`** (v0.3 —
Stage 1/2/3, 3-4 Ağustos'un derslerinden yazıldı).

CVD Level'in 3 Ağustos'ta geçtiği süreç, artık Trajectory Lab'a yeni bir
boyut eklemenin **standart kabul süreci**. Bir aday, aktif corpus'a
(`config.py`'de `active=True`) girmeden önce SIRAYLA:

1. **Overlay + Mean-Band** (göz testi) — SADECE "bu tamamen saçma mı"
   (rastgele gürültü) sorusunun ucuz, hızlı elemesi. **"Küçük görünüyor"
   göz testiyle ELENEMEZ** — z_score (Tier 4) bunun kanıtı: mean-band
   grafiğinde sinyal-öncesi fark gözle "önemsiz" görünüyordu, ama tam
   batarya (walk-forward+placebo+bootstrap) gerçek olduğunu gösterdi.
   Küçüklük/büyüklük kararı SADECE adım 6'daki (Cohen's d, placebo,
   bootstrap) ölçümlerle verilir.
2. **Divergence eğrisi** (`viz.compute_divergence`) — winner/loser
   ayrışmasının tepe noktası var mı, nerede.
3. **Haftalık (zaman) split** — tepe noktası farklı kalenderik dönemlerde
   tutuyor mu, yoksa tek bir dönemin/rejimin artefaktı mı.
4. **Sembol-grubu split** (majör/alt) — bulgu tek bir sembol grubuna mı
   özgü.
5. **Bootstrap %95 CI + Cohen's d** — etki büyüklüğü ve güven aralığı.
6. **Incremental value testi** — aday, MEVCUT boyut setine (o an aktif
   olan diğer boyutlara) GERÇEKTEN yeni bilgi katıyor mu, yoksa zaten
   var olanın tekrarı mı:
   - Mutual Information + korelasyon matrisi + redundancy (residual
     korelasyonu)
   - Kademeli classifier (LightGBM): mevcut set → mevcut set + aday, AUC/
     Precision/Recall/F1/Brier
   - **Walk-forward validation** (kronolojik, expanding window) — katkı
     farklı piyasa rejimlerinde de korunuyor mu
   - **1000-permütasyon placebo testi** — adayın katkısı, adayı
     karıştırıp (shuffle) yeniden ölçüldüğünde şans eseri elde
     edilebilecek bir büyüklük mü, yoksa gerçek mi (ampirik p-değeri)
   - Bootstrap %95 CI (ΔAUC) — katkı sıfırı dışlıyor mu
   - **Her indikatörde AYRI AYRI tekrar** — havuzlanmış sonuç tek bir
     indikatörün artefaktı olabilir, bu adım onu ekarte eder

**Karar kuralı:** permütasyon testi p<0.05 VE bootstrap CI sıfırı
dışlıyor VE katkı yönü tüm indikatörlerde/rejimlerde tutuyorsa →
`active=True`. Aksi halde aday `metrics.py`'de provider olarak kalır
(silinmez — bkz. "Provider ≠ corpus kapsamı" ilkesi), ama corpus'a dahil
edilmez.

Bu süreç, bir boyutun kabul edilme nedeninin "teorik olarak mantıklı
gelmesi" ya da "tek başına anlamlı çıkması" değil, **mevcut sete
gerçekten yeni bilgi katması** olmasını garanti eder.

## Reddedilen Boyutlar

Aynı hipotezin tekrar tekrar denenmesini önlemek için — kabul edilenler
kadar reddedilenler de kayıt altında.

| Aday | Tarih | Kategori | Red nedeni |
|---|---|---|---|
| `recent_win_rate` (PnL-tabanlı) | 3-4 Ağu | Entry (aday) | **Provenance kirliliği** (RESEARCH_SOP adım 1) — `PaperTrade.pnl_usd`'den türüyor, bu da SL/TP/trailing/timeout çıkış mekaniğinden geçmiş sonucu yansıtıyor. İstatistiksel testler bunu yakalayamaz (kirli girdiyi "anlamlı" gösterebilir), bu yüzden hiç test edilmeden reddedildi. |
| `recent_clean_win_rate` (aynı fikir, `signal_performance.return_t5_pct` ile temiz yeniden tanım — sembol+indikatör+yön'ün SON 20 sinyaldeki winner/loser oranı, PaperTrade'e hiç dokunmadan) | 4 Ağu | Entry | Provenance temizdi, Feature Acceptance Pipeline'ın TAMAMINDAN geçti ama **istatistiksel olarak reddedildi**: Cohen's d=0.114, MI=0.0000, redundancy≈0.03 (neredeyse sıfır), 1000-permütasyon placebo **p=0.079** (p<0.05 eşiğini geçemedi), bootstrap %95 CI=**[-0.0004,+0.0007]** (sıfırı içeriyor). "Bu sembol+strateji şu an sıcak mı soğuk mu" fikrinin — kirlilik sorunu düzeltilmiş hâliyle bile — gerçek bir sinyal taşımadığı sonucuna varıldı. |
| `regime_persistence` (son 30 barın, t=-1'deki rejimle aynı olan oranı) | 4 Ağu | Entry | **Redundant** — "Volatility Stability ailesi" analizinde `regime_age` ile r=0.799 (neredeyse aynı bilgi), tam-aile modelinde permutation importance=+0.0011 (en düşüklerden), leave-one-out çıkarılınca AUC düşüşü sadece +0.0006. `regime_age` aynı bilgiyi daha güçlü taşıyor, ikisini birden tutmanın faydası yok. |
| `vol_pct_pre_std` / "volatility_std" (t∈[-30,-1] volatility_pct serisinin std'si) | 4 Ağu | Entry | **Redundant** — aynı ailede en zayıf üye: tam-aile modelinde permutation importance=+0.0005 (en düşük), leave-one-out çıkarılınca AUC düşüşü sadece +0.0003, 3 corpus'ta tek-başına katkısı tutarsız (+0.0001/+0.0005/+0.0001 — neredeyse sıfır). NOT: bu, 10-bar pencereli (t∈[-10,-1]) ilk test edilen versiyondan FARKLI — 10-bar versiyonu tek başına Stage 1'i geçmişti (ΔAUC=+0.0019), ama aileyle adil (30-bar) karşılaştırıldığında en redundant üye çıktı; pencere uzunluğu bu boyut için kritik. |

## "Volatility Stability" ailesi — tek temsilci seçimi (4 Ağustos)

`vol_pct_pre_std`, `regime_age`, `regime_persistence` — üçü de aynı ham
kaynaktan (`volatility_pct`) türeyen, aynı hipotezi ("sinyal-öncesi
volatilite rejimi ne kadar istikrarlı") farklı açılardan ölçen adaylardı.
Amaç en güçlü üçünü bulmak değil, **aynı bilgiyi en az tekrar eden TEK
temsilciyi** seçmekti (bkz. RESEARCH_SOP.md Stage 2, madde 2 — redundancy
artık sadece BASE'e göre değil, AİLE İÇİNDE de ölçülüyor).

Yöntem: aile-içi korelasyon + BASE+tüm-aile modelinde permutation
importance + leave-one-out (birini çıkarınca AUC ne kadar düşüyor) + her
corpus'ta tek-başına tutarlılık. Sonuç: `regime_age` ↔ `regime_persistence`
arası r=0.799 (neredeyse aynı bilgi); `regime_age` her ölçütte (permutation
importance, leave-one-out, 3-corpus tutarlılığı) ailenin en güçlü/en az
redundant üyesi çıktı. **Sadece `regime_age` hayatta kaldı** (Research
Archive'e, Stage 2'nin pratik eşiğini geçemediği için) — diğer ikisi
Rejected/Redundant.

## Dosyalar

- `config.py` — pencere boyutu, aktif/kapalı metrik listesi, çıktı yolları.
- `metrics.py` — 5 provider (bağımsız, production'a bağımlı değil).
- `corpus_builder.py` — gerçek kapanmış sinyaller (indicators+direction) için
  DB'den pencere-boyu kline çekip trajectory corpus'unu (uzun/tidy format
  parquet) üretir. Outcome etiketi `signal_performance` tablosundaki temiz
  sabit-ufuk hedeften (return_t5_pct) gelir — Pattern Lab ile AYNI ölçüt.
- `viz.py` — 3 çekirdek görselleştirme: overlay, ortalama+güven bandı,
  çoklu-metrik sinyal hikâyesi.
- `explore.py` — komut satırı giriş noktası.

## Corpus şeması (uzun/tidy format)

```
signal_id | symbol | t0 | outcome | t_offset | metric | value
```

`outcome`: `signal_performance.return_t5_pct` eşiklenip winner/loser/neutral.
