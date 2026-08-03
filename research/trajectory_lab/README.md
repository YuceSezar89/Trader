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

Aktif (varsayılan corpus): **Price Return, EVOL, VPMV, CVD Slope**.
Hazır-bekleyen (provider olarak var, corpus'ta kapalı): **VP Score** — EVOL
ile aynı ham girdiyi (fiyat+hacim) paylaşıyor ve CVD Slope ile aynı
alış/satış hacim vekilinden türüyor; şimdilik teorik gerekçeyle değil,
gerçek overlay/mean-band bulguları CVD Slope'tan farklı davrandığını
gösterirse etkinleştirilecek.

VPMV'nin alt bileşenleri (vol/mom/vlt/price_score) ve devisso_score/delta
v1'de corpus'a DAHİL DEĞİL — ilki sadece `viz.plot_signal_story()` ile
istek üzerine (drill-down) hesaplanır, ikincisi EVOL ile aynı aileden
("Δfiyat / osilatör" oranı) olduğu için ayrıca eklenmedi.

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
