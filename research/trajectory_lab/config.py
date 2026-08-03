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
WARMUP_BARS = 220
# 3 Ağu: z_score (EMA200/STD200 tabanlı) eklenince 150'den 220'ye çıkarıldı —
# STD200'ün [-WINDOW_PRE, +WINDOW_POST] içinde geçerli (NaN olmayan) olması
# için en az 200 bar geçmişe ihtiyaç var. Diğer metrikleri etkilemez (daha
# fazla ısınma payı zarar vermez, sadece rolling pencerelerinin daha erken
# stabilize olmasını sağlar).

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
    # CVD Level (Tier 3, Accepted/Promoted): cvd_slope ile AYNI ham
    # kümülatif CVD'yi kullanır, farkı almadan seviye olarak normalize eder
    # (bkz. metrics.py::cvd_level_series docstring). Research SOP Stage 1
    # VE Stage 2'nin (bkz. RESEARCH_SOP.md) TÜMÜNÜ geçti: walk-forward +
    # 1000-permütasyon placebo (p=0.0000) + bootstrap %95 CI
    # (ΔAUC=[+0.004,+0.007]) + pratik katkı eşiği (ΔAUC=+0.0053 ≥ 0.003).
    "cvd_level": {"active": True},
    # z_score (Research Archive, 4 Ağu): Stage 1'i geçti (placebo p=0.0000,
    # bootstrap CI=[+0.0004,+0.0025] sıfırı dışlıyor, 3/3 indikatörde
    # tutarlı) AMA Stage 2'nin pratik katkı eşiğini (ΔAUC=+0.0015 < 0.003)
    # GEÇEMEDİ — istatistiksel olarak gerçek ama production'a değecek kadar
    # değerli değil (bkz. RESEARCH_SOP.md "Research Archive"). Bu yüzden
    # active=False'a geri alındı (önceden Tier 4/active=True idi, 4 Ağu'da
    # yeni Stage 2 eşiğiyle geri çekildi).
    "z_score": {"active": False},
    # volatility_pct (Research Archive'ın ham girdisi): signal_processor.py
    # ::volatility_regime'in bağımsız tam-seri kopyası (bkz.
    # metrics.py::volatility_pct_series). Kendisi değil, ondan türetilen 3
    # aday (vol_pct_pre_std, regime_age, regime_persistence — corpus_builder
    # dışı, ayrı script'lerde hesaplandı) test edildi: aile-içi redundancy
    # analizinde SADECE regime_age hayatta kaldı (Research Archive — Stage 2
    # pratik eşiğini geçemedi), diğer ikisi Rejected/Redundant (bkz. README).
    "volatility_pct": {"active": False},
}

# Outcome etiketleme — Pattern Lab ile AYNI temiz hedef (signal_performance
# tablosu), yeni bir "başarı" tanımı icat edilmiyor.
OUTCOME_METRIC = "return_t5_pct"
WINNER_THRESHOLD = 0.5  # >= bu yüzde → winner
LOSER_THRESHOLD = -0.5  # <= bu yüzde → loser
# arası: neutral

CORPUS_DIR = "research/trajectory_corpus"
REPORT_DIR = "research/trajectory_reports"
