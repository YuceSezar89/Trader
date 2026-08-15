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
    # Context Lab — Trend Gücü ailesi (5 Ağu): ADX/ema_slope/up_close_ratio
    # test edildi, 3 sinyal ailesinde de (HA_Cross/RSI_Cross/Supertrend)
    # |Cohen's d|<0.05 — Çapraz-Aile Doğrulama Kuralı'na göre "Genel
    # Başarısız Temsil", aile kapandı (bkz. RESEARCH_SOP.md). Provider
    # olarak kalıyor, active=False.
    "adx": {"active": False},
    "ema_slope": {"active": False},
    "up_close_ratio": {"active": False},
    # hh_hl: Trend Gücü ailesinden ÇIKARILDI (5 Ağu düzeltmesi) — "trend
    # gücü" değil "piyasa yapısı" ölçüyor, ayrı bir Market Structure
    # ailesinin adayı (BOS/Supertrend Break/Liquidity Sweep ile birlikte,
    # henüz Stage -1 hipotezi yazılmadı). İlk implementasyonu kaba bir
    # vekil (block max/min), gerçek swing-pivot değil.
    "hh_hl": {"active": False},
    # price_accel: Davranış Ailesi #1, sub-soru #2 ("Fiyat hızlanıyor mu?")
    # adayı (6 Ağu) — büyük-örneklem göz testi (n=65.673, HA_Cross_Long)
    # yapıldı: sinyal-öncesi (t<0) winner/loser TAMAMEN üst üste, ayrışma
    # SADECE t=0 sonrası (price_return/ema_slope/up_close_ratio ile AYNI
    # desen — 4. koşulsuz teyit). Stage 1a göz testi kapısında düştü,
    # koşulsuz haliyle. active=False (bkz. CONTEXT_LAB_STATUS.md).
    "price_accel": {"active": False},
    # price_vs_vwap: Davranış Ailesi #1, sub-soru #1 ("Fiyat önemli bir
    # referans seviyesini aşıyor mu?") adayı (6 Ağu) — büyük-örneklem göz
    # testi (n=65.685, HA_Cross_Long) yapıldı: sinyal-öncesi winner/loser
    # TAMAMEN üst üste (~VWAP'ın hafif altında, ikisi de), ayrışma SADECE
    # t=0 sonrası (ema_slope/up_close_ratio/price_return/price_accel ile
    # AYNI desen — 5. koşulsuz teyit). Stage 1a göz testinde düştü.
    # active=False (bkz. CONTEXT_LAB_STATUS.md).
    "price_vs_vwap": {"active": False},
    # "Enerji Birikimi" hipotezi adayları (7 Ağu) — büyük-örneklem göz
    # testi yapıldı (n=65.722, HA_Cross_Long): sinyal-öncesi winner/loser
    # TAMAMEN üst üste (6/7/8. koşulsuz teyit) — AMA her ikisi de t=-2/-1
    # civarında BİRLİKTE bir daralma gösteriyor (muhtemelen HA_Cross'un
    # kendi tetikleme mekanizmasının artefaktı, winner'a özgü değil).
    # Detay: CONTEXT_LAB_STATUS.md. active=False.
    "range_contraction": {"active": False},
    "body_contraction": {"active": False},
    "close_dispersion": {"active": False},
    # "Sıra dışılık" (sembol-içi göreceli) yönteminin ham girdileri (7 Ağu)
    # — büyük-örneklem testi yapıldı (HA_Cross_Long + RSI_Cross_Long,
    # 8 büyüklük), sonuçlar CONTEXT_LAB_STATUS.md'de. active=False.
    "body_size": {"active": True},
    "range_size": {"active": True},
    "volume_level": {"active": True},
    "rsi_raw": {"active": False},
    "atr_raw": {"active": False},
    "roc": {"active": True},
}

# Outcome etiketleme — Pattern Lab ile AYNI temiz hedef (signal_performance
# tablosu), yeni bir "başarı" tanımı icat edilmiyor.
OUTCOME_METRIC = "return_t5_pct"
OUTCOME_HORIZON_BARS = 5  # OUTCOME_METRIC'in ölçüldüğü bar — winner/loser kararı SADECE bu barda
WINNER_THRESHOLD = 0.5  # >= bu yüzde → winner
LOSER_THRESHOLD = -0.5  # <= bu yüzde → loser
# arası: neutral

CORPUS_DIR = "research/trajectory_corpus"
REPORT_DIR = "research/trajectory_reports"
