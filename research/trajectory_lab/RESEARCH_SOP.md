# Trajectory Lab — Research SOP v0.4

## Context Manifestosu (5 Ağustos — pusula, üç cümle)

Context Lab'ın amacı yeni indikatör bulmak değildir. Context Lab'ın
amacı piyasanın davranışlarını tanımlamak ve bu davranışları en iyi
temsil eden boyutu bulmaktır. Production'a yalnızca davranışı açıklayan
ve karar kalitesini anlamlı biçimde artıran boyutlar girer.

Her Context Lab çalışması TEK bir soruyla kapanır: **"Bu keşif botun
kararını gerçekten değiştirecek mi?"** Cevap "hayır" ise — istatistiksel
olarak güzel olsa bile — laboratuvar notu (Research Archive) olarak
kalır. Cevap "evet" ise Stage 4'te bir Production Rule'a dönüşür. Amaç
laboratuvara gömülmek değil, botun kararını gerçekten değiştirmek.

## Anayasa (5 Ağustos — üç madde, her şeyin üstünde)

1. **Önce soru belirlenir.** Hangi piyasa davranışı/bağlamı ölçülmek
   isteniyor — somut, yazılı bir cümleyle (aday formül değil).
2. **Sonra o soruyu ölçebilecek adaylar yarışır.** Birden fazla temsilci
   aynı disiplinle (Stage 1) test edilir — "Volatility Stability ailesi"
   ve "EVOL 7 temsil" presedanları gibi.
3. **Kazanan temsilci sisteme girer, diğerleri elenir.** Aynı soruyu
   soran birden fazla temsilci asla birlikte tutulmaz (redundancy).

Bu üç madde ihlal edilirse (ör. önce bir formül yazıp sonra "bu neyi
ölçüyor" diye sorulursa) çalışma GEÇERSİZ sayılır, baştan başlanır.

**Durum:** v0.4 (5 Ağustos) — numaralandırma BİRLEŞTİRİLDİ. v0.2/v0.3'ün
"Stage 1 (Discovery)/Stage 2 (Promotion)/Stage 3 (Mekanizma)" üçlüsü,
Context Lab için tasarlanan daha ince taneli **Stage -1→0→1→2→3→4**
şemasına haritalandı — artık TEK bir numaralandırma, hem sinyal
boyutları (EVOL/VPMV/CVD) hem context boyutları (Trend/Likidite/...)
için geçerli. En önemli ekleme: **Stage 4 (Production Rule)** —
laboratuvarın çıktısı artık bir metrik değil, botun kullanacağı somut
bir KURAL. Kullanıcının kendi sözleriyle: *"Bugün çıkan sonuç: 'EVOL
seviyesi tek başına yeterli değil, kalıcılığıyla birlikte okunmalı.' Bu
bir feature değil, bu bir piyasa kuralı."*

## Kullanıma Alma Zinciri — Research Accepted ≠ Production Accepted
(8 Ağustos — YENİ temel ilke, Anayasa ile aynı ağırlıkta)

Şimdiye kadarki disiplin "bulguyu keşfetme/doğrulama" tarafını güçlü
kurdu (Stage -1→2), ama doğrulanmış bir bulgunun trading sisteminde
NASIL kullanılabilir hale geleceği ayrı, açık bir aşama olarak
tanımlanmamıştı. Bu boşluk artık kapatılıyor:

> **Observation → Hypothesis → Pattern Validation → Research Accepted
> → Mechanism → Confluence → Strategy → Paper Trading → Production
> Accepted**

Her aşama FARKLI bir soruya cevap verir — birini atlayıp bir sonrakine
geçmek YOK, özellikle **Research Accepted'tan Production Accepted'a
DOĞRUDAN GEÇİŞ YOK**:

| Aşama | Soru | Mevcut Stage numarasıyla eşleşme |
|---|---|---|
| Observation → Hypothesis → Pattern Validation | "Gerçekten bir şey var mı?" | Stage -1, 0, 1a/1b |
| **Research Accepted** | (yukarıdakinin sonucu — istatistiksel olarak gerçek ve tekrarlanabilir) | Stage 2 |
| **Mechanism** | "Bu şey NEDEN/NE ZAMAN çalışıyor, ne zaman çalışmıyor?" | Stage 3 |
| **Confluence** (YENİ, ayrı ve zorunlu kontrol noktası) | "Başka Accepted market-state özellikleriyle BİRLİKTE nasıl davranıyor?" | Stage 3'ün içinde ima ediliyordu, artık AYRI aşama |
| **Strategy** | "Bu bilgiyi açık bir trading kararına dönüştürünce GERÇEKTEN faydalı mı?" | Stage 4 (Production Rule) — ama henüz PRODUCTION değil |
| **Paper Trading** (YENİ, zorunlu kontrol noktası) | "Gerçek zamanlı ortamda beklediğimiz gibi davranıyor mu?" | Stage 4 ile Pattern Lab arası — önceden formalize edilmemişti |
| **Production Accepted** | "Artık canlı sisteme güvenli şekilde alınabilir mi?" | Pattern Lab doğrulaması + config.py active=True |

**Research Accepted olmak şu ANLAMLARA GELMEZ:** config.py'ye eklenmek,
production'da kullanılmaya başlanmak, ya da threshold/entry/exit kuralı
üretmek. Threshold/strategy optimizasyonu SADECE Strategy aşamasının
işidir — Research/Mechanism/Confluence aşamalarında bir bulguyu hemen
`if close_pos > X: BUY` gibi bir kurala çevirmek YASAK (Anayasa'nın 3
maddesiyle aynı ağırlıkta bir kural).

**STATUS.md sınıflandırma etiketleri (8 Ağustos'tan itibaren HER bulgu
için kullanılacak):**

```
RESEARCH:    Tested / Research Accepted / Rejected
MECHANISM:   Under investigation / Mechanism supported /
             Mechanism rejected / Context-dependent
CONFLUENCE:  Not tested / Tested / Supported / Redundant /
             Context-dependent
STRATEGY:    Not translated / Candidate / Validated / Rejected
DEPLOYMENT:  Not paper tested / Paper tested / Production Accepted
```

**Büyük resim (8 Ağustos, kullanıcı — labın asıl hedefinin hatırlatması):**
hedef "en güçlü tek indikatörü bulmak" değil, **grafiği açan iyi bir
trader'ın yaptığı şeyi sistematikleştirmek: önce marketin ne durumda
olduğunu anlamak, sonra o duruma uygun stratejiyi seçmek.** `close_pos`,
`VPMV`, `CVD`, `body_size_pct` gibi boyutlar tek başına "al/sat
indikatörü" değil — market state'in farklı boyutlarını açıklayan
parçalar:

```
MARKET STATE
     │
     ├── Price behavior (close_pos, body_size_pct)
     ├── Energy / momentum (VPMV)
     ├── Volume / participation (CVD)
     ├── Volatility (volatility_pct)
     └── Temporal behavior (sinyal-zamanı dizileri)
            │
            ↓
      MARKET BEHAVIOR
            │
       ┌────┴────┐
       ↓         ↓
 CONTINUATION  FAILURE
       │         │
       ↓         ↓
  CONFLUENCE   AVOID/EXIT
       │
       ↓
    STRATEGY
```

**close_pos'un 8 Ağustos itibarıyla durumu (örnek uygulama):**
RESEARCH=Research Accepted | MECHANISM=Under investigation
(VPMV/volatilite etkileşimi keşfedildi, decomposition sürüyor) |
CONFLUENCE=Tested, Context-dependent (VPMV ile ters-U/monoton
asimetrik etkileşim, walk-forward'da büyük ölçüde tekrarlandı — ama
"Supported" değil, hâlâ mekanizma netleşmedi) | STRATEGY=Not translated
| DEPLOYMENT=Not paper tested. config.py'ye DOKUNULMADI, bu DOĞRU.

## Confluence Candidate Test Protokolü (9 Ağustos — mom_s0'ın temel
eksen adayı çıkmasından sonra formalize edildi)

**Temel prensip:** *"Bir değişken outcome ile ilişkili diye Confluence
adayı DEĞİLDİR. Önce mevcut EN GÜÇLÜ temel eksenin (şu an: mom_s0)
taşıdığı bilgiyi kontrol etmek gerekir."* Bu, VPMV/volatility/decay'in
hepsinin "ham etkide güçlü, momentum kontrolünde kaybolan" desenine
düşmesinden çıkarılan disiplin.

Yeni bir adayın Confluence'a aday olabilmesi için 5 adım (SIRAYLA):

1. **Kendi ayrımı var mı?** — Cohen's d / korelasyon, ham (bağlamsız).
2. **Temporal olarak stabil mi?** — walk-forward (expanding window,
   8 dilim), yön + bootstrap CI tutarlılığı.
3. **Mevcut temel eksenle (mom_s0) redundancy'si ne kadar?** — iki
   yönlü kapalı-form partial correlation.
4. **KRİTİK KARAR NOKTASI:** mom_s0 kontrol edildiğinde adayın katkısı
   KALIYOR mu?
   - Katkı KAYBOLUYORSA → **yeni bilgi DEĞİL**, redundant, kayıtta kalır
     ama Confluence adayı OLMAZ.
   - Katkı KORUNUYORSA → **gerçek Confluence adayı**, 5. adıma geç.
5. **Katkı birden fazla sinyal ailesinde VE zaman diliminde
   tekrarlanıyor mu?** — ancak bu adımdan sonra Confluence olarak
   işaretlenir.

**Aday önceliklendirme kuralı:** elde zaten olduğu için değil,
**GERÇEKTEN FARKLI bir market davranış boyutunu temsil ettiği için**
öncelik ver. price/close_pos/VPMV/decay'in büyük bölümünün AYNI
momentum ekseninde toplandığı zaten gösterildi — aranan şey "aynı
eksenin yeni bir versiyonu" değil, **yeni bir bilgi boyutu.**

**Negatif sonuç da GEÇERLİ bir araştırma sonucudur:** *"Bu örneklemde
mom_s0'ın üzerine anlamlı bağımsız bilgi ekleyen aday bulunamadı"* —
bu, Strategy tasarımını SADELEŞTİREN bir bulgudur, başarısızlık değil.

**KESİN: mom_s0 DAHİL hiçbir aday "hikâyeye uyduğu için" korunmaz —
hepsi çürütülmeye AÇIK kalır.** threshold/production/Strategy/paper
trading bu protokolün hiçbir adımında YOK — sadece bilgi boyutlarının
ayıklanması.

## Eski → yeni Stage haritası

| Eski (v0.2/v0.3) | Yeni (v0.4) |
|---|---|
| Stage 1, adım 0 (Hipotez bildirimi) | **Stage -1** (Hipotez) |
| — (yoktu, örtük) | **Stage 0** (Araştırma Sorusu — hipotez ölçülebilir TEK soruya indirgenir) |
| Stage 1, adım 1-2 (provenance/warmup GATE) + adım 5 (göz testi) | **Stage 1a** (Ön Eleme) |
| — (yoktu, tek adaylar test ediliyordu) | **Stage 1b** (Temsilci Yarışı — birden fazla aday varsa) |
| Stage 1, adım 6 + Stage 2 (tüm madde) | **Stage 2** (Kazananın Doğrulanması) |
| Stage 3 (madde 1-2: mekanizma+etkileşim) | **Stage 3** (Mekanizma) |
| Stage 3 (madde 3: karara çevirme) | **Stage 4** (Production Rule — YENİ, ayrıştırıldı) |

## STAGE -1 — Hipotez

Deney BAŞLAMADAN yazılır, sonuca göre değiştirilmez (hindsight bias'a
karşı). Düz dille bir piyasa davranışı iddiası: *"Trend varsa momentum
daha iyi çalışır." "Likidite düşükse edge azalır."* Beklenen mekanizma +
beklenen yön + başarı kriteri de burada yazılır.

## STAGE 0 — Araştırma Sorusu

Hipotez, ölçülebilir TEK bir soruya indirgenir: *"Trendi en doğru nasıl
ölçebiliriz?"* Bu adım henüz formül İÇERMEZ — sadece neyin ölçüleceğini
tarif eder (Anayasa madde 1'in somut uygulaması).

## Aile Başlangıç Tablosu (YENİ, 5 Ağustos — Stage 0 ile Stage 1a arasında ZORUNLU)

Stage 1a başlamadan önce her aile bu tabloyla açılır:

| Alan | İçerik |
|---|---|
| Davranış | Stage -1 hipotezinin tek ifadelik özeti |
| Soru | Stage 0 sorusu |
| Yarışmacılar | Aday temsilciler listesi |
| Kazanacak şey | Bu ailenin üreteceği nihai boyutun adı |

Her yarışmacı için **tek satırlık bir gerekçe zorunlu** ("Neden X? →
...") — gerekçesiz aday yarışa giremez (Anayasa madde 1'in ikinci
uygulaması: sadece soru değil, "bu aday neden bu soruyu ölçüyor" da
yazılı olmalı).

### Trend Gücü ailesi başlangıç tablosu (ilk örnek)

| Alan | İçerik |
|---|---|
| Davranış | Yönlü hareketin GÜCÜ (istatistiksel) |
| Soru | Yönlü piyasanın gücü en doğru nasıl temsil edilir? |
| Yarışmacılar | ADX, EMA Slope, Up Close Ratio |
| Kazanacak şey | Trend Gücü Dimension |

- **Neden ADX?** → Trend gücü ölçüyor.
- **Neden EMA Slope?** → Yön + hız birlikte.
- **Neden Up Close Ratio?** → En basit yön tutarlılığı ölçüsü.

**ADX / EMA Slope / Up Close Ratio — GENEL BAŞARISIZ TEMSİL (5 Ağustos):**
Sinyal-öncesi (t∈[-10,-1] ort. VE tek nokta t=-1, iki operasyonelleştirme
de aynı sonucu verdi) 3 sinyal ailesinin ÜÇÜNDE de test edildi (önyargısız
— HA_Cross'taki zayıflığın redundancy'den kaynaklanmadığı ÖNCE doğrulandı:
HA_Cross'un tetikleme kodu trend filtresi içermiyor, adayların
popülasyonda GENİŞ varyansı var, yani "zaten aynı bilgiyi taşıyor"
açıklaması elendi). Sonuç: 9 kombinasyonun (3 aday×3 indikatör) HEPSİNDE
|Cohen's d|<0.05, MI<0.006 — koşullu bir desen YOK (işaretler bazen
değişiyor ama büyüklükler sıfır-gürültüsü seviyesinde). Çapraz-Aile
Doğrulama Kuralı'na göre **Genel Başarısız Temsil** — **Trend Gücü
ailesi bu 3 temsilciyle KAPANDI** (provider olarak `metrics.py`'de
kalıyor, `active=False`).

**Not (5 Ağustos düzeltmesi):** `hh_hl_series` başlangıçta bu ailenin 4.
adayı olarak eklenmişti, ama kullanıcı doğru bir ayrım yaptı — HH/HL
"trend GÜCÜ" değil "piyasa YAPISI" ölçüyor, kavramsal olarak farklı bir
soru. Bu yüzden Trend Gücü ailesinden ÇIKARILDI ve **Market Structure**
ailesine (bkz. yukarı, aile #2) taşındı — Trend Gücü'nün kapanmasıyla
"karışık" bir statüde kalmadı, kendi ayrı ailesinde bekliyor. İlk
implementasyonu (`hh_hl_series`, ardışık 10-bar blok max/min kıyası)
klasik Dow-teorisi swing-pivot tanımı DEĞİL, kaba bir vekil — trend
olmadan sadece volatilite artışıyla bile tetiklenebilir. Market
Structure ailesinin Stage -1 hipotezi yazılınca, gerçek swing-pivot
tabanlı bir versiyonla birlikte ele alınabilir.

**Not (6 Ağustos — üst çerçeve düzeltmesi):** Kullanıcı "sinyaller
trende erken arıyoruz" sezgisiyle bu ailenin sorusunu sorguladı —
ADX/EMA Slope/Up Close Ratio'nun test ettiği şey ("mevcut trend gücü ne
kadar yüksek?") olgun bir trend varsayıyordu, ama sinyaller muhtemelen
trendin DOĞUŞ anını yakalıyor, olgunluğunu değil. Tartışma ilerlerken
daha temel bir sorun ortaya çıktı: "Trend Gücü" ve "Market Structure"
gibi aile adlarının KENDİSİ sorunlu — HH/HL, BOS, ADX gibi kavramların
evrensel bir tanımı yok (Pine Script'e, pencereye, timeframe'e göre
değişir), bu isimlerle başlamak davranışı değil implementasyonu test
etmek olur. Bu yüzden Trend Gücü ve Market Structure aile adları aktif
kategori olarak TERK EDİLDİ — bkz. aşağıda "Davranış Ailesi #1 —
Hareketin Doğuşu" ve onun nötr alt-soruları. Bu bölümdeki kapanış kaydı
(ADX/EMA Slope/Up Close Ratio testi) tarihsel bir veri noktası olarak
kalıyor, sonuçları geçerli.

## STAGE 1a — Ön Eleme (ucuz, hızlı)

Her aday (tek aday da olsa, birden fazla aday da olsa) yarışa girmeden
önce:

1. **Provenance/kirlilik kontrolü (GATE)** — execution-bağımlı mı
   (realized PnL, TP/SL/trailing) yoksa temiz mi (OHLCV,
   `return_t5_pct`)? Kirliyse → **yarışa hiç ALINMAZ**, ya reddedilir ya
   da temiz hedefle yeniden tanımlanır (`recent_win_rate` →
   `recent_clean_win_rate` presedanı). İstatistik hiç çalıştırılmadan
   burada elenebilir.
2. **Veri yeterliliği / warmup kontrolü (GATE)** — formülün rolling/
   lookback ihtiyacı `WARMUP_BARS+WINDOW_PRE` içinde karşılanıyor mu?
3. **Kategori etiketi** — Entry / Exit / *(ileride)* Risk-Position Mgmt.
4. **Corpus build** (gerekliyse)
5. **Görsel inceleme (overlay+mean-band)** — SADECE "bu açıkça gürültü
   mü?" sorusu. Büyüklük kararı BURADA verilmez (z_score dersi — göz
   testinde önemsiz görünen bir fark istatistiksel olarak gerçek
   çıkabilir).

Bu adımları geçemeyen aday **yarıştan çıkar**, diğer adaylar (varsa)
devam eder.

### Çapraz-Aile Doğrulama Kuralı (YENİ, 5 Ağustos — context adayları için)

Context adayları (sinyal boyutlarının aksine) tanım gereği HA_Cross/
RSI_Cross/Supertrend'den BAĞIMSIZ olmalı — bu yüzden bir aday, **TEK bir
sinyal ailesindeki** sonuçla nihai olarak elenmez ya da kabul edilmez.
Kural:

- **Bir indikatörde zayıf çıkarsa** → önce MEKANİZMA sorulur: bu zayıflık
  o indikatöre ÖZGÜ olabilir mi (ör. redundancy — indikatörün kendi
  tetikleme mantığı zaten aynı bilgiyi içeriyor mu)? Makul bir gerekçe
  YOKSA (ör. "Trend adayları HA_Cross popülasyonunda geniş varyansa
  sahip, redundant değil" — 5 Ağustos presedanı) → diğer ≥1 indikatörde
  de test edilir, önyargısız (aynı sonucu DOĞRULAMAK için değil,
  genellenebilirliği SINAMAK için).
- **≥2 sinyal ailesinde de başarısızsa** → **"Genel Başarısız Temsil"**
  statüsü — aileden çıkar (Rejected değil, ayrı bir kategori: temsil
  fikri denendi, iş görmedi).
- **Bir ailede çalışıp diğerinde çalışmazsa** → **"Koşullu Temsil"**
  statüsü — bu ayrı ve DEĞERLİ bir sonuç, basitçe elenmez. Hangi koşulda
  çalıştığı Stage 3'te (mekanizma) araştırılır — bu, sistemin zaten
  HA_Cross/RSI_Cross/Supertrend üzerine kurulu olması nedeniyle
  Context Lab'ın kalitesini artıracak bir ayrım.

## STAGE 1b — Temsilci Yarışı (birden fazla aday varsa)

Sadece AYNI soruyu ölçen adaylar birbiriyle yarışır (ör. Trend için
Supertrend/EMA/ADX). Bu KARŞILAŞTIRMALI bir eleme — Stage 2'nin pahalı
tam bataryası (1000-permütasyon placebo, bootstrap) HENÜZ uygulanmaz,
sadece adaylar birbirine göre sıralanır:

1. **Aile-içi korelasyon matrisi.**
2. **Hepsini BİRDEN mevcut Accepted sete ekleyip permutation
   importance** — her aday, DİĞERLERİ VARKEN ne kadar marjinal katkı
   sağlıyor.
3. **Leave-one-out** — her adayı tek tek çıkarıp AUC'nin ne kadar
   düştüğü.
4. **Her indikatörde/hafta/sembol-grubunda tutarlılık.**

Çıktı: bir ÖNCELİK SIRASI (1., 2., 3. ...) — tek bir "kazanan" değil,
sıralı bir liste (Stage 2'de kazanan düşerse sıradaki denenir).

Presedanlar: "Volatility Stability ailesi" (3 aday → `regime_age`
kazandı), "EVOL 7 temsil" (7 aday → `evol_age`/Persistence kazandı).

## STAGE 2 — Kazananın Doğrulanması

Stage 1b'nin 1. sıradaki adayı, **MEVCUT Accepted sete** (bu set
BÜYÜYEN bir şeydir — her yeni kabul edilen boyut bir sonrakinin tabanına
eklenir; ör. Likidite ailesi test edilirken taban artık EVOL+VPMV+CVD
Level+Trend-kazananını içerir) karşı TAM Feature Acceptance Pipeline
bataryasından geçirilir:

1. Cohen's d, Mutual Information, korelasyon matrisi, redundancy
   (residual korelasyonu).
2. Walk-forward validation (kronolojik, expanding window).
3. **1000-permütasyon placebo testi** (ampirik p-değeri).
4. **Bootstrap %95 CI** (ΔAUC sıfırı dışlamalı).
5. **Pratik katkı eşiği: ΔAUC ≥ 0.003** (sabit/kutsal değil, Pattern
   Lab'ın gerçek $ P&L sonucuyla kalibre edilebilir).
6. **Her indikatörde AYRI AYRI tekrar.**

**Kazanan tüm kapıları geçerse** → `active=True` (Accepted/Promoted),
Stage 3'e geç. **Geçemezse** → Stage 1b sıralamasındaki BİR SONRAKİ
aday ile Stage 2 TEKRARLANIR. **Sıradaki hiçbir aday geçemezse** → aile
"Soru doğru, mevcut temsilcilerin hiçbiri yeterli değil" statüsünde
KAPANIR — bu, "hipotez yanlış" ile KARIŞTIRILMAMALI, ayrı bir sonuç
kategorisidir (gelecekte yeni bir temsilci adayı çıkarsa yeniden
açılabilir). **Stage 1a/1b'nin herhangi bir kapısında düşen VEYA
redundancy'de elenen** → **Rejected** (nedeniyle: Provenance /
İstatistiksel / Yanlış Temsil / Redundant). **Stage 2'yi geçip sadece
pratik eşikte (madde 5) takılan** → **Research Archive**.

## STAGE 3 — Mekanizma

Stage 2'yi geçen bir aday, iki soru cevaplanmadan ilerleyemez:

1. **Neden çalışıyor?** — monotonluk kontrolü (değer arttıkça outcome
   düzgün mü değişiyor, yoksa beklenmedik bir kırılma/sıçrama var mı),
   winner/loser dağılım karşılaştırması. Beklenmedik bir kırılma
   bulunursa → **STOP, önce onu araştır** (`evol_age=6` sıçraması
   presedanı — araştırılmadan "sonuç" diye kabul edilmedi).
2. **Ne zaman çalışıyor?** — adayın etkisi MEVCUT kabul edilmiş diğer
   boyutlarla etkileşiyor mu (2D bucket/win-rate tablosu ile kontrol);
   hafta/sembol-grubu split'lerinde yön/büyüklük tutarlı mı.

Çıktı: bir karar haritası (win rate/PF/ortalama getiri/n içeren 2D ya da
çok boyutlu tablo/heatmap) — LightGBM'in kara-kutu olarak öğrendiği
ilişkinin insan tarafından denetlenebilir forma çevrilmiş hâli.

## STAGE 4 — Production Rule (YENİ, 5 Ağustos)

Laboratuvarın SON ürünü — ve Manifesto'nun kapanış sorusunun ("Bu keşif
botun kararını gerçekten değiştirecek mi?") cevabı "evet" olduğunda
girilen aşama. Stage 3'ün karar haritası burada 4 satırlık, sabit bir
formata sıkıştırılır — her kabul edilen boyut (ya da boyut ÇİFTİ, ör.
EVOL Strength×Persistence) için:

1. **Ne ölçüyor?**
2. **Ne zaman güvenilir?**
3. **Bot bunu nasıl kullanacak?**
4. **Hangi durumda kullanılmamalı?**

Bu format olmadan bir boyut Pattern Lab'a/production'a TAŞINMAZ —
`evol_age` örneği bunun neden şart olduğunu kanıtladı: istatistiksel
olarak çok güçlü (ΔAUC=+0.015-0.024) bir boyut bile, mekanizması
anlaşılıp somut bir kurala çevrilmeden naif bir eşiğe (ör. "age≥N ise
gir") dönüştürülürse YANLIŞ olabilir — çünkü etkisi EVOL Strength'e
göre YÖN DEĞİŞTİRİYORDU. Stage 4'ü geçen bir boyut, somut bir üretim-
kuralı ÖNERİSİYLE Pattern Lab'a taşınmaya hazırdır (Pattern Lab kendi
metodolojisiyle — kronolojik split, placebo, look-ahead kontrolü — bu
öneriyi ayrıca doğrular).

*Örnek (EVOL Strength×Persistence, tam metin için bkz. README):* Ne
ölçüyor → hacim-verimliliği seviyesi + kalıcılığı. Ne zaman güvenilir →
SADECE birlikte okunduğunda (tek başına yetersiz). Bot nasıl kullanacak
→ Strength<40: hemen çık / 40-60: temkinli tut / ≥60 VE Persistence≥3:
tut+güven artır. Ne zaman kullanılmamalı → Persistence=15 (censored) /
Short'ta veya HA_Cross/RSI_Cross/Supertrend dışında henüz doğrulanmadı.

## Research Archive

Stage 2'yi geçip sadece pratik eşikte (ΔAUC≥0.003) takılan boyutlar
burada kalır — silinmez, "başarısız" sayılmaz, sadece HENÜZ production'a
girecek kadar değerli değildir. Gelecekte: (a) Pattern Lab'da gerçek $
P&L ile test edilip pratik değeri kanıtlanırsa Accepted'e terfi edebilir,
(b) başka Archive üyeleriyle KOMBİNE edilip birlikte eşiği geçip
geçmediği denenebilir (henüz yapılmadı).

## Context Lab (5 Ağustos — ayrı bir lab değil, Trajectory Lab içinde bir çalışma teması)

Temel taşlar (EVOL Strength+Persistence, VPMV, CVD Level) kilitlendikten
sonraki hedef: **edge'in NEDEN değiştiğini açıklayan piyasa bağlamını
(context) bulmak** — yeni boyut avcılığı değil. Alpha (piyasaya göre
fazladan getiri) bu fazın context'lerle AÇIKLANACAK sonuç değişkeni,
kendisi bir context adayı DEĞİL — en sona, context aileleri kurulduktan
sonra tekrar ele alınacak.

**Context'in tanımı:** sinyalin İÇİNE doğduğu piyasa ortamı — hangi
indikatörün (HA_Cross/RSI_Cross/Supertrend) tetiklendiğini BİLMEDEN,
sadece "bu an ve bu sembolde piyasa nasıldı?" sorusuna cevap
verebilmelidir. Sinyal boyutları (EVOL/VPMV/CVD) sinyalin KENDİ
trajectory'sinden türer; context türemez, ortamı tarif eder.

**Davranış'ın tanımı (6 Ağustos):** Piyasanın sinyal anındaki dinamik
durumunu tarif eden, geleceği değil mevcut evreyi tanımlayan
gözlemlenebilir özellik. Bu tanım her yeni aday için bir kabul testi
sağlar — RSI ivmesi → davranış (sinyal anının dinamik durumu). Funding
oranı → davranış DEĞİL, çevre (Katılım Sorusu'na gider). Günün saati →
davranış DEĞİL, zaman (Zamansal Bağlam Sorusu'na gider). Supertrend
kırılımı → davranış (doğru temsilci olup olmadığı ayrı bir soru,
kategori doğru).

**Aile öncelik sırası ve Stage -1 hipotezleri** (5-6 Ağustos, kullanıcı
tarafından yazıldı — henüz araç/formül İÇERMİYOR, sadece davranış):

1. **Davranış Ailesi #1 — Hareketin Doğuşu** (6 Ağustos — "Trend Gücü"
   ve "Market Structure" adlarının YERİNE geçti, bkz. yukarıda "Aile
   Başlangıç Tablosu" bölümündeki 6 Ağustos notu). Stage -1 hipotezi:
   *"Sinyaller (HA_Cross/RSI_Cross/Supertrend) olgunlaşmış bir trendin
   içinde değil, yönlü hareketin DOĞUŞ anında tetikleniyor olabilir —
   mevcut göstergeler hareketin olgunluğunu değil, doğuşunu yansıtan
   davranışları taşıyor olabilir."* Stage 0 sorusu (winner/loser
   karşılaştırması olarak kurulmuş, sonucu varsaymıyor): **"Başarılı
   sinyaller ile başarısız sinyaller arasında, sinyal anında hangi
   piyasa davranışları sistematik olarak farklıdır?"**

   Bu soru, hazır kavramlarla (Trend/Market Structure/Volatilite)
   parçalanmadan, nötr alt-sorulara bölündü — her biri kendi Aile
   Başlangıç Tablosu'nu (Davranış/Soru/Yarışmacılar/Kazanacak şey)
   alacak, "Yarışmacılar" şu an bilerek BOŞ (Anayasa madde 1: önce soru,
   sonra aday):

   1. Fiyat önemli bir referans seviyesini aşıyor mu?
   2. Fiyat hızlanıyor mu?
   3. Katılım artıyor mu?
   4. Piyasa sıkışıyor mu yoksa genişliyor mu?
   5. Hareket süreklilik mi gösteriyor yoksa hemen sönüyor mu?

   Aday temsilciler (ADX ivmesi, RSI hızlanması, hacim patlaması,
   HH/HL, Supertrend kırılımı, açılış geri alımı, ...) bu 5 sorudan
   BİRİNE ya da BİRDEN FAZLASINA aday olabilir — aile adına göre önceden
   kutulanmazlar.
2. **Likidite** — *"Likidite düşük olduğunda sinyal kalitesi düşer."*
3. **Volatilite** — *"Volatilite tek başına iyi veya kötü değildir;
   hangi volatilite rejiminde olduğumuz sinyalin güvenilirliğini
   değiştirir."* (Not, 6 Ağustos: bu ailenin "rejim" sorusu ile yukarıdaki
   4. alt-soru "Piyasa sıkışıyor mu yoksa genişliyor mu?" arasındaki
   sınır henüz netleştirilmedi — statik rejim mi, dinamik geçiş mi
   sorusu ayrıca ele alınacak, şimdilik ikisi de açık.)
4. **Zaman** — *"Piyasanın gün içindeki davranışı homojen değildir.
   Bazı zaman dilimleri doğal edge üretir."*
5. **BTC Context** — *"Altcoin sinyalleri kendi başına
   değerlendirilmemeli; BTC'nin içinde bulunduğu rejim başarı
   olasılığını değiştirir."*
6. **Haber** — en sona bırakıldı (ölçmesi en zor); henüz hipotez
   yazılmadı.

Her aile Stage -1→0→1a→1b→2→3→4'ün TAMAMINDAN geçer.

**Final redundancy denetimi:** son aile (BTC Context) tamamlandığında,
TÜM context boyutları için EVOL/VPMV/CVD Level'de yaptığımız "Accepted
set gerçekten birbirini tamamlıyor mu?" denetiminin AYNISI (korelasyon +
leave-one-out) tekrarlanır — aileler arası gizli redundancy'yi (ör.
Trend Gücü ile BTC Context'in altcoin-BTC bağımlılığı yüzünden
örtüşmesi) yakalamak için.

## Tek-sayfa aile özeti (YENİ, 5 Ağustos)

Her TAMAMLANAN (Stage 4'ü geçmiş) boyut/aile için, `README.md`'nin
büyümesini önlemek amacıyla AYRI bir dosya:
`research/trajectory_lab/context_summaries/<aile_adı>.md`. İçerik
(sabit format):

- Hipotez (Stage -1)
- Araştırma sorusu (Stage 0)
- Yarışan temsilciler (Stage 1b)
- Kazanan temsilci
- Neden kazandı (Stage 1b/2 kanıtı, özet)
- Reddedilen temsilciler ve gerekçeleri
- Production'da nasıl kullanılacak (Stage 4'ün 4 maddesi)
- Açık kalan araştırma soruları

`README.md`/`RESEARCH_SOP.md` bu dosyalara SADECE link verir, içeriği
tekrarlamaz — 6 ay sonra "yüzlerce test arasında kaybolmamak" için.

## Şu anki sınıflandırma (5 Ağustos)

- **Accepted (Promoted), SABİT temel taşlar:** EVOL Strength (Tier 1),
  EVOL Persistence/`evol_age` (Tier 1, Strength'in AYNI ekseni —
  ΔAUC=+0.015/+0.024, CVD Level'in üstünde, ama TEK BAŞINA doğrusal eşik
  olarak kullanılamaz, Strength ile BİRLİKTE/2D okunur — Stage 4 kuralı
  README'de), VPMV (Tier 2), CVD Level (Tier 3). Yeni boyut avcılığı
  durdu, odak artık Context Lab: eksik piyasa davranışlarını tek tek
  doğrulamak + rejim-bağımlılığını anlamak; her yeni fikir ancak gerçek
  işlem performansına katkı gösterirse sisteme girecek.
- **Research Archive:** z_score, regime_age (ikisi de placebo/bootstrap
  geçti ama ΔAUC<0.003 — not: bu "regime_age" volatility_pct ailesinden,
  `evol_age` ile KARIŞTIRILMAMALI, farklı ham kaynaklardan türüyorlar)
- **Rejected — Redundant:** regime_persistence (regime_age ile r=0.799),
  volatility_std / `vol_pct_pre_std` (aile-içi en zayıf üye — bkz. README
  "Volatility Stability ailesi" analizi)
- **Rejected — Provenance:** `recent_win_rate` (PnL-tabanlı)
- **Rejected — İstatistiksel:** `recent_clean_win_rate` (placebo p=0.079)
- **Rejected — Yanlış Temsil:** CVD Slope (bkz. README — CVD Level'in
  kontrollü deneyiyle düzeltildi)

## Dokümantasyon kuralı

Her deney — Accepted/Archive/Rejected fark etmez — hipotez+tarih+sonuç+
kategori ile kayıt altına alınır. Sinyal boyutları (EVOL/VPMV/CVD)
README.md'de üç bölümde (Accepted Tier tablosu, Research Archive,
Reddedilen Boyutlar); Context Lab'ın her TAMAMLANAN ailesi ise ayrı bir
tek-sayfa özet dosyasında (`context_summaries/`, yukarıya bkz.).
