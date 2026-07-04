"""
Pattern Lab — vaka-kontrol çalışması sabitleri.

Tasarım kararlarının TEK adresi. Değişiklik = yeni çalışma; eski raporlar
meta.json'daki kopyasıyla kalır (sonuç değişirse nedeni izlenebilir).
"""

SEED = 42

# Pencereler
SELECTION_DAYS = 10      # vaka seçimi: son N günün getirisi
CORPUS_DAYS    = 14      # veri: seçimden 4 gün fazla (ısınma + 48h özellik payı)
BREAKOUT_FWD_H = 24      # t0 = en yüksek N saatlik ileri getirinin başlangıcı
FEAT_LONG_H    = 48      # uzak bağlam penceresi
FEAT_SHORT_H   = 6       # ateşleme penceresi

# Gruplar
N_CASES    = 20
N_CONTROLS = 20
MID_QUANTILE = (0.25, 0.75)   # kontrol adayları: getiri dağılımının orta dilimi
VOL_DECILES  = 10             # likidite eşleştirme ondalık sayısı

# Veri kalitesi
MIN_BAR_COVERAGE = 0.95   # beklenen 5m bar sayısının altına düşen sembol elenir

# Sıralama (ayrışma) pencereleri — 5m çözünürlükte, bar sayısı olarak
RANK_WINDOWS_BARS = {"30m": 6, "1h": 12, "4h": 48, "24h": 288}

# Hüküm eşikleri (Adım 6 — baştan yazılı)
EFFECT_THRESHOLD = 0.30   # |Cliff's delta| bu değerin üstü "belirgin"

BARS_PER_DAY_5M = 288

CORPUS_DIR  = "research/corpus"
REPORT_DIR  = "research/reports"
