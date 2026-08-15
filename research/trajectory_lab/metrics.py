"""
Trajectory Lab — metrik provider'ları.

HİÇBİRİ production modüllerini (indicators/core.py, utils/vpmv.py,
signals/market_context.py, utils/preprocessing.py) IMPORT ETMEZ — her
fonksiyon ilgili production formülünün BAĞIMSIZ bir kopyasıdır, ama
üretim koduyla AYNI sonuca çıkacak şekilde yazılmıştır (bkz. README).
Fark: hepsi son barın değeri yerine TÜM PENCERE için seriyi döner (trajectory
şekli budur), rolling/rank pencerelerinin ısınma payı çağıran tarafın
(corpus_builder.py) sorumluluğundadır.

Her fonksiyon aynı imzayı paylaşır: (df: pd.DataFrame, **kwargs) -> pd.Series
— df en az open_time/close/volume (bazıları high/low da) içermeli.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# Normalizasyon yardımcıları — utils/preprocessing.py'nin bağımsız kopyası
# ============================================================


def _normalize_volume_0_100(series: pd.Series, window: int = 50) -> pd.Series:
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=series.index)
    log_series = np.log1p(np.maximum(series, 0))
    rolling_min = log_series.rolling(window, min_periods=1).min()
    rolling_max = log_series.rolling(window, min_periods=1).max()
    range_val = rolling_max - rolling_min
    normalized = np.where(range_val > 1e-8, ((log_series - rolling_min) / range_val) * 100, 50.0)
    return pd.Series(normalized, index=series.index).fillna(50.0)


def _normalize_momentum_0_100(series: pd.Series, k: float = 1.0, window: int = 100) -> pd.Series:
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=series.index)
    rolling_mean = series.rolling(window, min_periods=10).mean()
    rolling_std = series.rolling(window, min_periods=10).std()
    z_score = (series - rolling_mean) / (rolling_std + 1e-8)
    sigmoid = 100 / (1 + np.exp(-k * z_score))
    return sigmoid.fillna(50.0)


def _normalize_volatility_0_100(series: pd.Series, window: int = 200) -> pd.Series:
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=series.index)
    return (series.rolling(window, min_periods=10).rank(pct=True) * 100).fillna(50.0)


def _normalize_price_0_100(series: pd.Series, window: int = 50) -> pd.Series:
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=series.index)
    median = series.rolling(window, min_periods=1).median()
    q25 = series.rolling(window, min_periods=1).quantile(0.25)
    q75 = series.rolling(window, min_periods=1).quantile(0.75)
    iqr = q75 - q25
    normalized = ((series - median) / (iqr + 1e-8)) * 20 + 50
    return pd.Series(np.clip(normalized, 0, 100), index=series.index).fillna(50.0)


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """indicators/core.py::calculate_rsi'nin bağımsız kopyası (TV/Pine EMA tabanlı)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_values = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    return pd.Series(rsi_values, index=df.index)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """indicators/core.py::calculate_atr'nin bağımsız kopyası."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ============================================================
# 5 provider
# ============================================================


def price_return_series(df: pd.DataFrame) -> pd.Series:
    """Pencere başından itibaren kümülatif yüzde getiri — referans amaçlı,
    diğer metriklerin gerçekten fiyattan farklı bir şey söyleyip söylemediğini
    kontrol etmek için (bkz. README)."""
    base = df["close"].iloc[0]
    return ((df["close"] / base) - 1.0) * 100.0


def evol_series(
    df: pd.DataFrame, rvol_window: int = 20, smooth_span: int = 7, rank_window: int = 100
) -> pd.Series:
    """indicators/core.py::calculate_evol'un bağımsız, TAM SERİ kopyası.
    ΔFiyat% / RVOL, EMA ile yumuşatılıp her nokta için kendi geçmiş
    `rank_window`'una göre percentile-rank'e (0-100) çevrilir — production
    ile AYNI formül `(recent < current).sum() / len(recent)`, sadece
    her bar için tekrarlanır (production sadece son bar için hesaplar)."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    price_pct = close.pct_change() * 100.0
    rvol = volume / volume.rolling(rvol_window).mean()
    raw = price_pct / rvol.replace(0.0, np.nan)
    smoothed = raw.ewm(span=smooth_span, adjust=False).mean()
    smoothed = smoothed.replace([np.inf, -np.inf], np.nan)

    def _pct_rank(window: np.ndarray) -> float:
        current = window[-1]
        if np.isnan(current):
            return np.nan
        recent = window[~np.isnan(window)]
        if len(recent) < 20:
            return np.nan
        return float((recent < current).sum()) / len(recent) * 100.0

    return smoothed.rolling(rank_window, min_periods=20).apply(_pct_rank, raw=True)


def vpmv_series(df: pd.DataFrame, signal_type: str) -> pd.Series:
    """utils/vpmv.py::compute_series'in bağımsız kopyası — production'da
    ZATEN tam seri döndürüyordu (son barı almadan önce), burada birebir aynı."""
    side = 1.0 if signal_type == "Long" else -1.0
    has_dir = (
        "buy_volume" in df.columns
        and "sell_volume" in df.columns
        and df["buy_volume"].notna().any()
    )
    volume_side = (df["buy_volume"] if side > 0 else df["sell_volume"]) if has_dir else df["volume"]

    vol = _normalize_volume_0_100(volume_side)
    rsi = _rsi(df, period=14)
    mom = _normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)
    atr = _atr(df, period=14)
    vlt = _normalize_volatility_0_100(atr)
    prc = _normalize_price_0_100(df["close"].pct_change().fillna(0.0) * 100.0 * side)

    return (0.35 * vol + 0.35 * mom + 0.20 * vlt + 0.10 * prc).clip(0, 100)


def vpmv_components_series(df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    """vpmv_series'in AYNI iç mantığı — yeni formül YOK, sadece ağırlıklı
    tek skora indirgemeden ÖNCEKİ 4 bileşeni (vol/mom/vlt/prc) ayrı ayrı
    döndürür (8 Ağustos, VPMV decomposition — bkz. CONTEXT_LAB_STATUS.md,
    Mechanism aşaması). Ağırlıklar (0.35/0.35/0.20/0.10) BİLEREK
    uygulanmıyor — "hangi bileşen VPMV'ye en çok katkı yapıyor" sorusuna
    şimdilik girilmiyor, sadece "hangisi winner/loser'da farklı davranıyor"
    sorusuna bakılıyor (kullanıcı notu, 8 Ağustos)."""
    side = 1.0 if signal_type == "Long" else -1.0
    has_dir = (
        "buy_volume" in df.columns
        and "sell_volume" in df.columns
        and df["buy_volume"].notna().any()
    )
    volume_side = (df["buy_volume"] if side > 0 else df["sell_volume"]) if has_dir else df["volume"]

    vol = _normalize_volume_0_100(volume_side)
    rsi = _rsi(df, period=14)
    mom = _normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)
    atr = _atr(df, period=14)
    vlt = _normalize_volatility_0_100(atr)
    prc = _normalize_price_0_100(df["close"].pct_change().fillna(0.0) * 100.0 * side)

    return pd.DataFrame({"mom": mom, "vol": vol, "vlt": vlt, "prc": prc})


def cvd_slope_series(df: pd.DataFrame, slope_window: int = 10) -> pd.Series:
    """market_context.py::compute_cvd_slope'un bağımsız, TAM SERİ kopyası."""
    if "buy_volume" in df.columns and df["buy_volume"].notna().any():
        hl = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["buy_volume"].fillna(df["volume"] * (df["close"] - df["low"]) / hl)
    else:
        hl = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["volume"] * (df["close"] - df["low"]) / hl
    cvd = (2 * bv - df["volume"]).cumsum()
    avg_vol = df["volume"].rolling(slope_window).mean().clip(lower=1e-8)
    return cvd.diff().rolling(slope_window).mean() / avg_vol


def z_score_series(df: pd.DataFrame) -> pd.Series:
    """signal_processor.py::z_score_entry hesabının bağımsız, TAM SERİ
    kopyası — sembolün kendi fiyatının EMA200'den kaç std uzakta olduğu.
    Giriş-anı (statik) değeri z_score_entry olarak zaten kaydediliyor;
    burada amaç sinyal ÖNCESİ trajectory'sini (eğimini) görmek — production
    formülü, sadece son bar yerine tüm seri için. NOT: STD200'ün geçerli
    olması için en az 200 bar geçmiş gerekir (bkz. config.py WARMUP_BARS)."""
    close = df["close"].astype(float)
    ema200 = close.ewm(span=200, adjust=False).mean()
    std200 = close.rolling(200).std()
    return (close - ema200) / (std200 + 1e-12)


def volatility_pct_series(df: pd.DataFrame) -> pd.Series:
    """signal_processor.py::volatility_regime hesabının bağımsız, TAM SERİ
    kopyası — ATR(14)'ün 200-bar rolling percentile-rank'i (0-100). Production
    bunu >70 high / <30 low / normal kategorik etiketine indirgiyor; burada
    amaç sinyal-öncesi REJİM İSTİKRARINI (bu serinin t∈[-10,-1] std'si —
    düşük std=istikrarlı rejim) test etmek, ham seviyeyi değil. _atr/
    _normalize_volatility_0_100 zaten vpmv_series içinde kullanılan aynı
    helper'lar — yeni formül riski yok. NOT: 200-bar rolling ihtiyacı
    z_score ile aynı (bkz. config.py WARMUP_BARS=220)."""
    atr = _atr(df, period=14)
    return _normalize_volatility_0_100(atr)


def adx_series(df: pd.DataFrame, adxlen: int = 14, dilen: int = 14) -> pd.Series:
    """indicators/core.py::calculate_adx'in bağımsız, vektörize kopyası
    (Wilder RMA = ewm(alpha=1/period), production'ın Python-loop'lu
    wilder_rma'sıyla matematiksel olarak AYNI, sadece hızlı). Trend
    ailesi adayı — "trend GÜCÜNÜ" ölçer, yönü değil. Production kendi
    içinde len(df)>=28 ısınma payı istiyor, WARMUP_BARS=220 fazlasıyla
    yeterli."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1 / period, adjust=False).mean()

    truerange = _wilder_rma(tr, dilen)
    plus = 100 * _wilder_rma(plus_dm, dilen) / truerange.replace(0.0, np.nan)
    minus = 100 * _wilder_rma(minus_dm, dilen) / truerange.replace(0.0, np.nan)
    dx = (plus - minus).abs() / (plus + minus).replace(0.0, 1.0)
    return 100 * _wilder_rma(dx, adxlen)


def ema_slope_series(df: pd.DataFrame, span: int = 50, lookback: int = 10) -> pd.Series:
    """Trend ailesi adayı — EMA(50)'nin son `lookback` bardaki YÜZDE
    değişimi (yön + hız birlikte). Sembolün fiyat mertebesinden bağımsız
    olması için % olarak normalize edilir (ham nokta farkı değil)."""
    ema = df["close"].astype(float).ewm(span=span, adjust=False).mean()
    return (ema / ema.shift(lookback) - 1.0) * 100.0


def hh_hl_series(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Trend ailesi adayı — saf fiyat-yapısı (Dow teorisi): son `window`
    barın en yükseği bir önceki `window` barın en yükseğinden büyük mü
    (HH) VE en düşüğü bir önceki `window` barın en düşüğünden büyük mü
    (HL). Skor -1 (LH+LL, düşüş yapısı) ile +1 (HH+HL, yükseliş yapısı)
    arası, 0 karışık yapı."""
    curr_high = df["high"].rolling(window).max()
    prev_high = curr_high.shift(window)
    curr_low = df["low"].rolling(window).min()
    prev_low = curr_low.shift(window)
    hh = (curr_high > prev_high).astype(float)
    hl = (curr_low > prev_low).astype(float)
    return hh + hl - 1.0


def up_close_ratio_series(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Trend ailesi adayı — en basit yön tutarlılığı ölçüsü: son `window`
    barın yüzde kaçı bir önceki kapanıştan yüksek kapandı (0-100)."""
    up = (df["close"].diff() > 0).astype(float)
    return up.rolling(window).mean() * 100.0


def price_accel_series(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Davranış Ailesi #1, sub-soru #2 ("Fiyat hızlanıyor mu?") adayı —
    `price_return_series`'in (pozisyon/hız) AYIRT EDEMEDİĞİ şeyi ölçmeyi
    hedefler: sabit hızlı hareket (momentum sürüyor, B sınıfı) ile hızı
    ARTAN hareket (sıkışmadan kırılım, A sınıfı) arasındaki fark. İkinci
    türev: velocity(t) = t'den t-window'a % getiri (hız), price_accel(t)
    = velocity(t) - velocity(t-window) (hızın kendi değişimi). B'de ~0'a
    yakın, A'da pozitif ve büyüyen olması beklenir — henüz test edilmedi,
    Stage 1a öncesi aday."""
    velocity = (df["close"] / df["close"].shift(window) - 1.0) * 100.0
    return velocity - velocity.shift(window)


def price_vs_vwap_series(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """Davranış Ailesi #1, sub-soru #1 ("Fiyat önemli bir referans
    seviyesini aşıyor mu?") adayı — takvim sınırı (günlük/haftalık/aylık
    açılış) GEREKTİRMEYEN tek referans seviyesi: rolling/session VWAP.
    Günlük/haftalık açılış ve önceki gün/hafta H-L kasıtlı olarak
    DIŞARIDA tutuldu (6 Ağu kararı) — HA_Cross_Long sinyallerinin
    çoğu 5m interval'de, mevcut WARMUP_BARS=220 penceresi (~20.8 saat)
    günlük açılışı bile güvenilir kapsamıyor; hesaplanabilen sinyal
    alt-kümesiyle sınırlı bir test yanlı olurdu. lookback=100,
    WARMUP_BARS=220 içinde ISINMA payı bırakacak şekilde seçildi.

    price_vs_vwap(t) = (close(t) - VWAP(t)) / VWAP(t) * 100 — fiyat
    VWAP'ın kaç % üstünde/altında."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical_price * df["volume"]).rolling(lookback).sum()
    v = df["volume"].rolling(lookback).sum()
    vwap = pv / v.replace(0.0, np.nan)
    return (df["close"] - vwap) / vwap * 100.0


def body_size_series(df: pd.DataFrame) -> pd.Series:
    """"Sıra dışılık" (sembol-içi göreceli) yönteminin ham girdisi (7
    Ağu) — mum gövdesi büyüklüğü, |close-open|, HİÇBİR mühendislik yok.
    Kendi başına test edilmiyor; amaç bunun sembol-içi percentile-
    rank'ini almak (bkz. analiz scripti, CONTEXT_LAB_STATUS.md)."""
    return (df["close"] - df["open"]).abs()


def range_size_series(df: pd.DataFrame) -> pd.Series:
    """"Sıra dışılık" yönteminin ham girdisi — bar aralığı, high-low,
    ham (bkz. body_size_series docstring)."""
    return df["high"] - df["low"]


def volume_level_series(df: pd.DataFrame) -> pd.Series:
    """"Sıra dışılık" yönteminin ham girdisi — ham hacim, normalize
    edilmemiş (bkz. body_size_series docstring)."""
    return df["volume"]


def rsi_raw_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """"Sıra dışılık" yönteminin ham girdisi — RSI(14), ham, hiçbir ek
    işlem yok (bkz. body_size_series docstring)."""
    return _rsi(df, period=period)


def atr_raw_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """"Sıra dışılık" yönteminin ham girdisi — ATR(14), ham (bkz.
    body_size_series docstring)."""
    return _atr(df, period=period)


def roc_series(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """"Sıra dışılık" yönteminin ham girdisi — son `window` bardaki
    yüzde değişim (rate of change), ham (bkz. body_size_series
    docstring)."""
    return (df["close"] / df["close"].shift(window) - 1.0) * 100.0


def range_contraction_series(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """"Enerji Birikimi" hipotezi adayı (7 Ağu, objektif gözlem #5+#7:
    son N mumun high-low mesafesi daralıyor mu / fiyat bir aralığın
    dışına çıkamıyor mu) — son `window` barın (high-low) ortalaması,
    `window` bar önceki AYNI ortalamaya göre yüzde kaç küçülmüş. Pozitif
    = aralık daralıyor (sıkışma); negatif = genişliyor."""
    rng = df["high"] - df["low"]
    avg_now = rng.rolling(window).mean()
    avg_before = avg_now.shift(window)
    return (avg_before - avg_now) / avg_before.replace(0.0, np.nan) * 100.0


def body_contraction_series(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """"Enerji Birikimi" hipotezi adayı (7 Ağu, objektif gözlem #4: mum
    gövdesi küçülüyor mu) — `range_contraction_series` ile AYNI mantık,
    ham girdi (high-low) yerine gövde büyüklüğü (|close-open|). Pozitif
    = gövdeler küçülüyor."""
    body = (df["close"] - df["open"]).abs()
    avg_now = body.rolling(window).mean()
    avg_before = avg_now.shift(window)
    return (avg_before - avg_now) / avg_before.replace(0.0, np.nan) * 100.0


def close_dispersion_series(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """"Enerji Birikimi" hipotezi adayı (7 Ağu, objektif gözlem #2+#6:
    kapanışlar/açılışlar birbirine yakın bir bölgede kümeleniyor mu) —
    son `window` kapanışın değişim katsayısı (std/mean, %). DİĞER İKİ
    ADAYIN TERSİNE, burada DÜŞÜK değer sıkışmaya işaret eder (kapanışlar
    az saçılmış)."""
    std = df["close"].rolling(window).std()
    mean = df["close"].rolling(window).mean()
    return std / mean.replace(0.0, np.nan) * 100.0


def cvd_level_series(df: pd.DataFrame, lookback: int = 500) -> pd.Series:
    """cvd_slope_series ile BİREBİR AYNI ham kümülatif CVD'yi ((2*bv-volume)
    cumsum) kullanır — fark alıp orana çevirmek yerine, VP Score'daki `_norm`
    ile aynı rolling min-max normalizasyonuyla 0-100 SEVİYESİNE sıkıştırır.

    Amaç: CVD Slope'un divergence-peak testinde neden zaman/sembol split'te
    tutmadığını ayırt etmek — "CVD verisi bilgi taşımıyor" mu, yoksa "oran
    (türev) temsili yanlış nesneye yanlış test" mi? Tek manipüle edilen
    değişken temsil biçimi (seviye vs oran); ham order-flow girdisi
    cvd_slope_series ile birebir aynı — kontrollü deney.

    lookback=500, VP Score ile aynı temsil biçimini korumak ve yalnızca
    "seviye vs. oran" değişkenini izole etmek amacıyla seçildi. Bu çalışma,
    normalizasyon penceresini optimize etmeyi hedeflememektedir.
    """
    if "buy_volume" in df.columns and df["buy_volume"].notna().any():
        hl = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["buy_volume"].fillna(df["volume"] * (df["close"] - df["low"]) / hl)
    else:
        hl = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["volume"] * (df["close"] - df["low"]) / hl
    cvd = (2 * bv - df["volume"]).cumsum()
    lo = cvd.rolling(lookback, min_periods=1).min()
    hi = cvd.rolling(lookback, min_periods=1).max()
    return ((cvd - lo) / (hi - lo + 1e-10) * 100).fillna(50.0)


def vp_score_series(
    df: pd.DataFrame, lookback: int = 500, use_real_volume: bool = False
) -> pd.Series:
    """market_context.py::compute_vp_score'un bağımsız, TAM SERİ kopyası.
    Döner: buy_pos_avg - sell_neg_avg (0-100 skalada, +: alıcı baskın)."""
    price_change = df["close"].diff().fillna(0.0)
    cum_positive = price_change.clip(lower=0.0).cumsum()
    cum_negative = (-price_change).clip(lower=0.0).cumsum()
    total_move = (cum_positive + cum_negative).replace(0, np.nan)
    positive_pct = (cum_positive / total_move * 100).fillna(50.0)
    negative_pct = (cum_negative / total_move * 100).fillna(50.0)

    if use_real_volume:
        bv = df["buy_volume"].fillna(0.0)
        sv = df["sell_volume"].fillna(0.0)
    else:
        hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
        bv = df["volume"] * (df["close"] - df["low"]) / hl_range
        sv = df["volume"] * (df["high"] - df["close"]) / hl_range
    cum_buy = bv.cumsum()
    cum_sell = sv.cumsum()
    total_vol = (cum_buy + cum_sell).replace(0, np.nan)
    buy_pct = (cum_buy / total_vol * 100).fillna(50.0)
    sell_pct = (cum_sell / total_vol * 100).fillna(50.0)

    def _norm(s: pd.Series) -> pd.Series:
        lo = s.rolling(lookback, min_periods=1).min()
        hi = s.rolling(lookback, min_periods=1).max()
        return ((s - lo) / (hi - lo + 1e-10) * 100).fillna(50.0)

    buy_pos_avg = (_norm(buy_pct) + _norm(positive_pct)) / 2
    sell_neg_avg = (_norm(sell_pct) + _norm(negative_pct)) / 2
    return buy_pos_avg - sell_neg_avg


PROVIDERS = {
    "price_return": lambda df, signal_type: price_return_series(df),
    "evol": lambda df, signal_type: evol_series(df),
    "vpmv": lambda df, signal_type: vpmv_series(df, signal_type),
    "cvd_slope": lambda df, signal_type: cvd_slope_series(df),
    "cvd_level": lambda df, signal_type: cvd_level_series(df),
    "vp_score": lambda df, signal_type: vp_score_series(df),
    "z_score": lambda df, signal_type: z_score_series(df),
    "volatility_pct": lambda df, signal_type: volatility_pct_series(df),
    "adx": lambda df, signal_type: adx_series(df),
    "ema_slope": lambda df, signal_type: ema_slope_series(df),
    "hh_hl": lambda df, signal_type: hh_hl_series(df),
    "up_close_ratio": lambda df, signal_type: up_close_ratio_series(df),
    "price_accel": lambda df, signal_type: price_accel_series(df),
    "price_vs_vwap": lambda df, signal_type: price_vs_vwap_series(df),
    "range_contraction": lambda df, signal_type: range_contraction_series(df),
    "body_contraction": lambda df, signal_type: body_contraction_series(df),
    "close_dispersion": lambda df, signal_type: close_dispersion_series(df),
    "body_size": lambda df, signal_type: body_size_series(df),
    "range_size": lambda df, signal_type: range_size_series(df),
    "volume_level": lambda df, signal_type: volume_level_series(df),
    "rsi_raw": lambda df, signal_type: rsi_raw_series(df),
    "atr_raw": lambda df, signal_type: atr_raw_series(df),
    "roc": lambda df, signal_type: roc_series(df),
}
