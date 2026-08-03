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

| Tier | Metrik | Cohen's d | Mutual Information | ΔAUC (mevcut sete katkı) | Zamanlama |
|---|---|---|---|---|---|
| **1** | EVOL | 1.90 | 0.284 | (temel özellik) | post-sinyal (t=+4) |
| **2** | VPMV | 0.97 | 0.149 | (temel özellik) | post-sinyal (t=0) |
| **3** | CVD Level | 0.385 | 0.076 | +0.0053 ≥ 0.003 ✓ | post-sinyal (t=+5) |

### Accepted set gerçekten birbirini tamamlıyor mu? (4 Ağustos leave-one-out doğrulaması)

Yeni feature aramak yerine, zaten kabul edilmiş 3 özelliğin (EVOL/VPMV/
CVD Level) birbirinin gerçekten tekrarı olup olmadığı ayrıca test edildi
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

### Research Archive — istatistiksel olarak gerçek, ama pratik eşiği geçemedi

Stage 1'in TÜMÜNÜ (placebo p<0.05, bootstrap CI sıfırı dışlıyor, 3/3
indikatörde tutarlı) geçen, ama Stage 2'nin ΔAUC≥0.003 pratik eşiğinde
kalan adaylar. **Silinmez, "başarısız" sayılmaz** — sadece henüz
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
Amaç artık özellik SAYISINI değil, özellik KALİTESİNİ artırmak — p<0.05
olan ama AUC'yi +0.001 artıran özelliklerle sistemi doldurmamak.

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

Bu bölüm, bir adayın KABUL/RED kararına giden istatistiksel bataryayı
anlatır. Bir adayın buraya girmeye DEĞER olup olmadığına (hipotez,
provenance/kirlilik kontrolü, warmup kontrolü, göz testiyle erken eleme)
karar veren süreç ayrı bir dokümanda: **`RESEARCH_SOP.md`** (v0.1, taslak
— 3 Ağustos'un derslerinden yazıldı, sonraki 10-15 adayda test edilecek).

CVD Level'in 3 Ağustos'ta geçtiği süreç, artık Trajectory Lab'a yeni bir
metrik/feature eklemenin **standart kabul süreci**. Bir aday, aktif
corpus'a (`config.py`'de `active=True`) girmeden önce SIRAYLA:

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
6. **Incremental value testi** — aday, MEVCUT feature set'e (o an aktif
   olan diğer metriklere) GERÇEKTEN yeni bilgi katıyor mu, yoksa zaten
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

Bu süreç, bir feature'ın kabul edilme nedeninin "teorik olarak mantıklı
gelmesi" ya da "tek başına anlamlı çıkması" değil, **mevcut sete
gerçekten yeni bilgi katması** olmasını garanti eder.

## Reddedilen Özellikler

Aynı hipotezin tekrar tekrar denenmesini önlemek için — kabul edilenler
kadar reddedilenler de kayıt altında.

| Aday | Tarih | Kategori | Red nedeni |
|---|---|---|---|
| `recent_win_rate` (PnL-tabanlı) | 3-4 Ağu | Entry (aday) | **Provenance kirliliği** (RESEARCH_SOP adım 1) — `PaperTrade.pnl_usd`'den türüyor, bu da SL/TP/trailing/timeout çıkış mekaniğinden geçmiş sonucu yansıtıyor. İstatistiksel testler bunu yakalayamaz (kirli girdiyi "anlamlı" gösterebilir), bu yüzden hiç test edilmeden reddedildi. |
| `recent_clean_win_rate` (aynı fikir, `signal_performance.return_t5_pct` ile temiz yeniden tanım — sembol+indikatör+yön'ün SON 20 sinyaldeki winner/loser oranı, PaperTrade'e hiç dokunmadan) | 4 Ağu | Entry | Provenance temizdi, Feature Acceptance Pipeline'ın TAMAMINDAN geçti ama **istatistiksel olarak reddedildi**: Cohen's d=0.114, MI=0.0000, redundancy≈0.03 (neredeyse sıfır), 1000-permütasyon placebo **p=0.079** (p<0.05 eşiğini geçemedi), bootstrap %95 CI=**[-0.0004,+0.0007]** (sıfırı içeriyor). "Bu sembol+strateji şu an sıcak mı soğuk mu" fikrinin — kirlilik sorunu düzeltilmiş hâliyle bile — gerçek bir sinyal taşımadığı sonucuna varıldı. |
| `regime_persistence` (son 30 barın, t=-1'deki rejimle aynı olan oranı) | 4 Ağu | Entry | **Redundant** — "Volatility Stability ailesi" analizinde `regime_age` ile r=0.799 (neredeyse aynı bilgi), tam-aile modelinde permutation importance=+0.0011 (en düşüklerden), leave-one-out çıkarılınca AUC düşüşü sadece +0.0006. `regime_age` aynı bilgiyi daha güçlü taşıyor, ikisini birden tutmanın faydası yok. |
| `vol_pct_pre_std` / "volatility_std" (t∈[-30,-1] volatility_pct serisinin std'si) | 4 Ağu | Entry | **Redundant** — aynı ailede en zayıf üye: tam-aile modelinde permutation importance=+0.0005 (en düşük), leave-one-out çıkarılınca AUC düşüşü sadece +0.0003, 3 corpus'ta tek-başına katkısı tutarsız (+0.0001/+0.0005/+0.0001 — neredeyse sıfır). NOT: bu, 10-bar pencereli (t∈[-10,-1]) ilk test edilen versiyondan FARKLI — 10-bar versiyonu tek başına Stage 1'i geçmişti (ΔAUC=+0.0019), ama aileyle adil (30-bar) karşılaştırıldığında en redundant üye çıktı; pencere uzunluğu bu özellik için kritik. |

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
