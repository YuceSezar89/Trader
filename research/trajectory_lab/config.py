"""
Trajectory Lab — tasarım kararlarının TEK adresi (Pattern Lab'ın kendi
config.py'siyle aynı disiplin: değişiklik = yeni çalışma).
"""

SEED = 42

# Gözlem penceresi (sinyal barına göre, bar sayısı olarak)
WINDOW_PRE = 30  # t0'dan önce kaç bar
WINDOW_POST = 15  # t0'dan sonra kaç bar

# Isınma payı — EVOL/VPMV gibi metrikler kendi içinde rolling/rank
# pencereleri kullanıyor (ör. EVOL: RVOL(20) + EMA(7) + rank(100)); bu
# hesapların pencerenin İLK barında bile geçerli olması için WINDOW_PRE'nin
# öncesine ek geçmiş çekilir, sonra corpus'a sadece [-WINDOW_PRE, +WINDOW_POST]
# trim edilir (Pattern Lab'ın CORPUS_DAYS = SELECTION_DAYS + ısınma payı
# deseniyle aynı mantık).
WARMUP_BARS = 150

# metrics.py'deki TÜM provider'lar — hiçbiri "değersiz" diye özel muamele
# görmüyor, sadece hangisi varsayılan corpus'a dahil ("active") ayrı bir karar.
METRIC_PROVIDERS = {
    "price_return": {"active": True},
    "evol": {"active": True},
    "vpmv": {"active": True},
    "cvd_slope": {"active": True},
    # VP Score: EVOL ile aynı ham girdiyi (fiyat+hacim), CVD Slope ile aynı
    # alış/satış hacim vekilini paylaşıyor. Teorik gerekçeyle değil, gerçek
    # overlay/mean-band bulguları CVD Slope'tan ayrıştığını gösterirse
    # active=True yapılacak (bkz. README).
    "vp_score": {"active": False},
}

# Outcome etiketleme — Pattern Lab ile AYNI temiz hedef (signal_performance
# tablosu), yeni bir "başarı" tanımı icat edilmiyor.
OUTCOME_METRIC = "return_t5_pct"
WINNER_THRESHOLD = 0.5  # >= bu yüzde → winner
LOSER_THRESHOLD = -0.5  # <= bu yüzde → loser
# arası: neutral

CORPUS_DIR = "research/trajectory_corpus"
REPORT_DIR = "research/trajectory_reports"
