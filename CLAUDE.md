# TRader Panel - Claude Çalışma Kuralları

## Dil

- Her zaman Türkçe konuş

## Çalışma Tarzı

- Her adımda ne yapacağımı söyle, onay bekle, sonra yap
- Bir şey yapmadan önce planı göster — onay gelince kodu yaz
- Sadece istenen şeyi yap, fazlasını yapma
- Cevaplar orta uzunlukta: kısa açıklama + sonuç

## Kod Kuralları

- Yorum satırı ekleme (gerekmedikçe)
- Refaktör istenmedikçe çevresini temizleme
- Her değişiklikten önce ilgili dosyayı oku

## Kod Kalite Kuralları

Her kod değişikliğinden sonra:
- pylint kontrolü yap
- autoflake ile import temizliği yap
- Duplicate kod varsa belirt

### Araçlar

- black        → otomatik kod formatlama
- isort        → import sıralaması
- mypy         → tip kontrolü (type hints)
- bandit       → güvenlik açıkları
- pylint       → kod kalite analizi
- autoflake    → kullanılmayan import temizliği

## Datetime Kuralı — KESİN

Projede datetime için tek kural:

```python
from datetime import datetime
datetime.now()   # ✓ tek kabul edilen form
```

**Yasak olanlar — bunları asla yazma:**
```python
datetime.now(ZoneInfo("Europe/Istanbul"))  # ✗ aware — DB reddeder
datetime.utcnow()                           # ✗ UTC ≠ lokal, karışıklık
datetime.now(timezone.utc)                  # ✗ aware — DB reddeder
```

**DB kolon tipi:** Her zaman `DateTime` (timezone yok). `PG_TIMESTAMP(timezone=True)` kullanma.

**Kural:** Herhangi bir dosyada `_IST`, `ZoneInfo`, `timezone.utc` görürsen datetime context'inde — sil, `datetime.now()` ile değiştir.

## Proje Bağlamı

- Bilgi grafiği: `graphify-out/` klasöründe mevcut (2026-06-09)
- Refaktör için `graphify query` ile graf sorgulanabilir
- God nodes: Config (61), RedisClient (56), LiveDataManager (~49), TVChart/DivergencePanel güncellendi
- Graf: 2151 node, 5472 edge, 134 community
