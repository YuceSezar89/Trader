# Trajectory Lab — Research SOP v0.2

**Durum:** v0.2 — 4 Ağustos'ta felsefe değişti: v0.1 tek soruyu sordu
("istatistiksel olarak anlamlı mı?"). v0.2 bunu ikiye ayırıyor —
**Stage 1 (Discovery)** hâlâ o soruyu sorar, ama artık yeterli değil.
**Stage 2 (Promotion)** yeni ve asıl soru: *"production'a girecek kadar
DEĞERLİ mi?"* p<0.05 olan ama AUC'ye +0.001 katan bir özellik istatistiksel
olarak gerçektir ama sistemi bu tür küçük özelliklerle doldurmak amaç
değil. Amaç artık özellik SAYISINI değil, özellik KALİTESİNİ artırmak.

## Bu doküman ile Feature Acceptance Pipeline'ın (README.md) farkı

- **Research SOP (bu dosya):** bir deneyin doğuşundan, Stage 1'i geçip
  Stage 2'ye (Promotion) girmeye değer olup olmadığına kadar olan süreç.
- **Feature Acceptance Pipeline (README.md):** Stage 1 ve Stage 2'nin
  istatistiksel bataryasının kendisi (walk-forward, split'ler, bootstrap,
  incremental value, redundancy).

## STAGE 1 — Discovery (serbest araştırma, değişmedi)

**0. Hipotez bildirimi** (deney BAŞLAMADAN yazılır, sonuca göre
değiştirilmez — hindsight bias'a karşı): Hipotez / beklenen mekanizma /
beklenen yön / başarı kriteri.

**1. Provenance/kirlilik kontrolü (GATE)** — execution-bağımlı mı
(realized PnL, TP/SL/trailing) yoksa temiz mi (OHLCV, `return_t5_pct`)?
Kirliyse → STOP veya temiz hedefle yeniden tanımla (`recent_win_rate` →
`recent_clean_win_rate` presedanı).

**2. Veri yeterliliği / warmup kontrolü (GATE)** — formülün rolling/
lookback ihtiyacı `WARMUP_BARS+WINDOW_PRE` içinde karşılanıyor mu?

**3. Kategori etiketi** — Entry / Exit / *(ileride)* Risk-Position Mgmt.

**4. Corpus build** (gerekliyse)

**5. Görsel inceleme (overlay+mean-band)** — SADECE "bu açıkça gürültü
mü?" sorusu. Büyüklük kararı BURADA verilmez (z_score dersi).

**6. Erken karar (Stop Conditions)** — gürültüyse/kirliyse/yetersizse
STOP; değilse Feature Acceptance Pipeline'ın istatistiksel bataryasına
geç (divergence, hafta/sembol split, Cohen's d, MI, korelasyon,
redundancy, walk-forward, 1000-permütasyon placebo, bootstrap CI, her
indikatörde ayrı doğrulama — bkz. README).

Stage 1'i geçen (placebo p<0.05 VE bootstrap CI sıfırı dışlıyor VE yön
tüm indikatörlerde tutuyor) bir aday artık **"istatistiksel olarak
gerçek"** — ama henüz **"production'a değer"** değil. Bu ayrım YENİ.

## STAGE 2 — Promotion (YENİ, 4 Ağustos)

Stage 1'i geçen bir aday, `config.py`'de `active=True` olmadan önce EK
olarak şu kapılardan geçmeli:

1. **Tekrar üretilebilirlik** — walk-forward'ın TÜM dilimlerinde VE her
   indikatörde AYRI AYRI aynı yönde (Stage 1'de zaten ölçülüyor, burada
   eşik: sapma/istisna YOK, hepsi tutarlı olmalı).
2. **Düşük redundancy** — sadece BASE'e göre değil, **mevcut KABUL
   EDİLMİŞ (Accepted) sete göre** — aday, ailesindeki diğer adaylarla
   (ör. volatility ailesi: std/age/persistence) leave-one-out ile
   karşılaştırılıp en az tekrar edeni seçilir (bkz. "Volatility Stability
   ailesi" presedanı — sadece `regime_age` hayatta kaldı).
3. **Split testleri** (hafta + sembol) geçilmeli.
4. **Placebo testi** geçilmeli (p<0.05).
5. **Bootstrap %95 CI** sıfırı dışlamalı.
6. **Pratik katkı eşiği (YENİ, asıl fark)** — istatistiksel anlamlılık
   YETMEZ. Somut eşik: **ΔAUC ≥ 0.003** (mevcut Accepted sete eklendiğinde
   — LightGBM walk-forward ortalaması). Bu sayı ilk kez 4 Ağustos'ta
   belirlendi, gerekirse (Pattern Lab'ın gerçek $ P&L sonuçlarıyla
   kalibre edilerek) güncellenebilir — sabit/kutsal bir sayı değil.

**Tüm 6 kapıyı geçen** → `active=True` (Accepted/Promoted).
**Stage 1'i geçip Stage 2'nin pratik-katkı eşiğinde (madde 6) takılan**
→ **Research Archive** (bkz. aşağı) — provider olarak kalır, `active=
False`, istatistiksel olarak doğrulanmış ama production'a alınmamış.
**Stage 1'in herhangi bir kapısında düşen VEYA redundancy'de (madde 2)
elenen** → **Rejected** (nedeniyle: Provenance / İstatistiksel / Yanlış
Temsil / Redundant).

## Research Archive

Küçük ama istatistiksel olarak gerçek katkılar burada kalır — silinmez,
"başarısız" sayılmaz, sadece HENÜZ production'a girecek kadar değerli
değildir. Gelecekte: (a) Pattern Lab'da gerçek $ P&L ile test edilip
pratik değeri kanıtlanırsa Accepted'e terfi edebilir, (b) başka Archive
üyeleriyle KOMBİNE edilip birlikte eşiği geçip geçmediği denenebilir
(henüz yapılmadı — tek tek küçük olan katkılar toplamda anlamlı olabilir,
bu ayrı bir araştırma sorusu).

## Şu anki sınıflandırma (4 Ağustos)

- **Accepted (Promoted):** EVOL (Tier 1), VPMV (Tier 2), CVD Level (Tier 3)
- **Research Archive:** z_score, regime_age (ikisi de placebo/bootstrap
  geçti ama ΔAUC<0.003)
- **Rejected — Redundant:** regime_persistence (regime_age ile r=0.799),
  volatility_std / `vol_pct_pre_std` (aile-içi en zayıf üye — bkz. README
  "Volatility Stability ailesi" analizi)
- **Rejected — Provenance:** `recent_win_rate` (PnL-tabanlı)
- **Rejected — İstatistiksel:** `recent_clean_win_rate` (placebo p=0.079)
- **Rejected — Yanlış Temsil:** CVD Slope (bkz. README — CVD Level'in
  kontrollü deneyiyle düzeltildi)

## Dokümantasyon kuralı

Her deney — Accepted/Archive/Rejected fark etmez — hipotez+tarih+sonuç+
kategori ile kayıt altına alınır. README.md'de üç bölüm: "Accepted"
(Tier tablosu), "Research Archive", "Reddedilen Özellikler".
