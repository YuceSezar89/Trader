"""
signals/signal_processor.py::process_and_enrich_signals — orkestratör testleri.

Bu fonksiyon 528 satır, ~15 alt-hesaplamayı zincirliyor. Çoğu alt-fonksiyon
zaten ayrı test edildi (Faz 1-3: trailing/risk_manager, _compute_mtf_score,
_count_ha_ultra_confirm/_htf_ha_bullish, _compute_all_up, _compute_devisso_score,
_compute_fvg, _get_pt_flag/_get_ranking_snapshot/_rsi_cross_gate_passed) ve
signal_engine.py (31 test, tests/test_signal_engine.py). Bu dosyanın amacı
BUNLARI tekrar test etmek değil, orkestratörün KENDİSİNE özgü mantığı
doğrulamak: bar/interval kapıları, VPMV min-skor ve ratio filtreleri,
z-score sınırları, is_confluence hesabı, dry_run kısa devresi, enriched_signal
alan doldurma ve gerçek Signal DB yazımı.

Bilinçli sınırlar (kolaya kaçmak değil, test katmanı seçimi):
- `signal_engine.calculate_all_signals` monkeypatch'lenir — KENDİSİ ayrı ve
  gerçek verilerle (test_signal_engine.py) test edildi, burada orkestrasyonu
  izole etmek için kontrollü bir technical_signals sözlüğü enjekte edilir.
- `compute_components`/`compute_pre` (utils/vpmv.py) monkeypatch'lenir — bu
  modülün KENDİSİ henüz test edilmemiş (ayrı, bilinen bir boşluk, bu görevin
  kapsamı dışında); burada sadece VPMV min-skor/ratio KAPILARININ doğru
  çalıştığını kontrollü sayılarla doğruluyoruz.
- z_score_entry / regime_trend / volatility_regime / btc_z / devisso_score /
  VP score / market_structure / candle_pattern GERÇEK hesaplanır (gerçek OHLCV
  ile, mock YOK) — bunlar fonksiyonun kendi satır-içi pandas hesaplarıdır.
- Redis: RedisClient sahte istemciyle kontrol edilir (canlı "ranking:snapshot"/
  "settings:paper_trade_enabled"/"ticker:*" key'lerine dokunulmaz).
- Güvenlik: Config.PAPER["ENABLED_STRATEGIES"] testler boyunca [] zorlanır —
  bugünkü canlı config'e (sadece ta_kovalama_live) güvenilmiyor, gelecekte
  başka bir strateji eklenirse bile testler canlı paper-trade verisine asla
  yazmaz. signal_lifecycle_manager.process() GERÇEK çalışır (TEST sembolüyle
  izole, temizlenir) — bu fonksiyonun gerçek DB yazma yolu.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

import signals.signal_processor as sp
from config import Config
from database.engine import async_engine, get_session
from database.models import PaperTrade, Signal
from signals.signal_engine import signal_engine
from signals.signal_processor import (
    _PT_FLAG_CACHE,
    _RSI_CROSS_SNAPSHOT_CACHE,
    process_and_enrich_signals,
)
from utils.redis_client import RedisClient

pytestmark = pytest.mark.database

TEST_SYMBOL = "TESTUSDT"


class _FakeRedisClient:
    def __init__(self, responses=None):
        self._responses = responses or {}

    async def get(self, key):
        return self._responses.get(key)

    async def incr(self, key):
        return 1


async def _no_mtf_klines(symbol, tf, limit=None):
    return None


@pytest.fixture(autouse=True)
def _isolate_redis_and_config(monkeypatch):
    """Modül-seviyesi TTL cache'leri sıfırlar, Redis'i sahte istemciyle
    kontrol eder, canlı paper-trade stratejilerine yazılmasını KESİN olarak
    engeller (bkz. dosya docstring'i — güvenlik gerekçesi)."""
    _PT_FLAG_CACHE["value"] = "1"
    _PT_FLAG_CACHE["ts"] = 0.0
    _RSI_CROSS_SNAPSHOT_CACHE["data"] = None
    _RSI_CROSS_SNAPSHOT_CACHE["ts"] = 0.0
    monkeypatch.setattr(
        RedisClient,
        "get_client",
        lambda: _FakeRedisClient({"settings:paper_trade_enabled": "1"}),
    )
    monkeypatch.setattr(RedisClient, "get_mtf_klines", _no_mtf_klines)
    monkeypatch.setitem(Config.PAPER, "ENABLED_STRATEGIES", [])


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine_pool():
    yield
    await async_engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def clean_test_data():
    async def _cleanup():
        async with get_session() as session:
            await session.execute(delete(Signal).where(Signal.symbol == TEST_SYMBOL))
            await session.execute(delete(PaperTrade).where(PaperTrade.symbol == TEST_SYMBOL))
            await session.commit()

    await _cleanup()
    yield
    await _cleanup()


# ============================================================
# Yardımcılar
# ============================================================


def _build_price_df(n=220, seed=1, crash=None, step_ms=300_000):
    """Gerçek z_score_entry/regime/ADX/btc_z hesaplarını beslemek için
    yeterli (>=210 bar) sentetik OHLCV. `crash`: son barın kapanışını
    ema200'e göre ne kadar kaydıracağını kontrol eder (z-score testleri)."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 0.3, n))
    if crash is not None:
        prices[-1] = prices[-2] + crash
    return pd.DataFrame(
        {
            "open_time": np.arange(n) * step_ms,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1000.0),
        }
    )


def _tech_signal(signal_type="Long", indicators="RSI_Cross(9,24)", price=102.0, atr=1.5):
    return {
        "signal_type": signal_type,
        "timestamp": datetime.now(),
        "price": price,
        "pullback_level": None,
        "strength": 1,
        "indicators": indicators,
        "adx": None,
        "plus_di": None,
        "minus_di": None,
        "momentum": None,
        "rsi": 55.0,
        "macd": None,
        "atr": atr,
    }


def _patch_signal_engine(monkeypatch, technical_signals):
    async def _fake_calc(df, symbol="", interval="", signal_types=None):
        return technical_signals

    monkeypatch.setattr(signal_engine, "calculate_all_signals", _fake_calc)


def _patch_vpmv(monkeypatch, vol=80.0, mom=80.0, vlt=80.0, prc=80.0, pre_avg=50.0, vpmv_sig=70.0):
    """VPMV min-skor (B kapısı) ve ratio (VPMV pre/slope/ratio) kapılarını
    kontrollü sayılarla besler — compute_components/compute_pre'nin KENDİ
    formülü değil, orkestratörün bu sayılarla doğru karar verip vermediği
    test ediliyor."""

    def _fake_components(df, sig_type):
        return vol, mom, vlt, prc

    def _fake_pre(df, sig_type, volume_mode="real"):
        return pre_avg, 1.0, vpmv_sig

    monkeypatch.setattr(sp, "compute_components", _fake_components)
    monkeypatch.setattr(sp, "compute_pre", _fake_pre)


async def _get_test_signal():
    async with get_session() as session:
        result = await session.execute(select(Signal).where(Signal.symbol == TEST_SYMBOL))
        return result.scalars().first()


# ============================================================
# Guard/kapı testleri — signal_engine hiç çağrılmamalı
# ============================================================


@pytest.mark.asyncio
async def test_empty_df_skips_before_signal_engine(monkeypatch, clean_test_data):
    called = {"v": False}

    async def _fake_calc(*a, **kw):
        called["v"] = True
        return {}

    monkeypatch.setattr(signal_engine, "calculate_all_signals", _fake_calc)
    await process_and_enrich_signals(TEST_SYMBOL, pd.DataFrame(), _build_price_df(), "5m")
    assert called["v"] is False


@pytest.mark.asyncio
async def test_empty_ref_df_skips_before_signal_engine(monkeypatch, clean_test_data):
    called = {"v": False}

    async def _fake_calc(*a, **kw):
        called["v"] = True
        return {}

    monkeypatch.setattr(signal_engine, "calculate_all_signals", _fake_calc)
    await process_and_enrich_signals(TEST_SYMBOL, _build_price_df(), pd.DataFrame(), "5m")
    assert called["v"] is False


@pytest.mark.asyncio
async def test_insufficient_bars_skips_before_signal_engine(monkeypatch, clean_test_data):
    called = {"v": False}

    async def _fake_calc(*a, **kw):
        called["v"] = True
        return {}

    monkeypatch.setattr(signal_engine, "calculate_all_signals", _fake_calc)
    small_df = _build_price_df(n=10)  # 5m için min 100 bar gerekli
    await process_and_enrich_signals(TEST_SYMBOL, small_df, small_df, "5m")
    assert called["v"] is False


@pytest.mark.asyncio
async def test_wrong_interval_skips_before_signal_engine(monkeypatch, clean_test_data):
    called = {"v": False}

    async def _fake_calc(*a, **kw):
        called["v"] = True
        return {}

    monkeypatch.setattr(signal_engine, "calculate_all_signals", _fake_calc)
    df = _build_price_df()
    await process_and_enrich_signals(TEST_SYMBOL, df, df, "1h")  # sadece 5m/15m üretir
    assert called["v"] is False


@pytest.mark.asyncio
async def test_no_technical_signals_no_db_write(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {})
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    assert await _get_test_signal() is None


# ============================================================
# VPMV min-skor filtresi (B kapısı)
# ============================================================


@pytest.mark.asyncio
async def test_vpmv_below_min_score_skips_signal(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch, vol=20.0, mom=20.0, vlt=20.0, prc=20.0)  # skor=20 < MIN_SCORE(50)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    assert await _get_test_signal() is None


@pytest.mark.asyncio
async def test_vpmv_above_min_score_proceeds(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch, vol=80.0, mom=80.0, vlt=80.0, prc=80.0)  # skor=80 >= 50
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    sig = await _get_test_signal()
    assert sig is not None
    assert sig.vpms_score == pytest.approx(80.0, abs=0.5)


# ============================================================
# z-score filtresi (gerçek EMA200/STD200 hesabıyla)
# ============================================================


@pytest.mark.asyncio
async def test_long_skipped_when_zscore_below_min(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal(signal_type="Long")]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=-1)  # z ~ -2.13 < LONG_Z_MIN(-2.0)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    assert await _get_test_signal() is None


@pytest.mark.asyncio
async def test_long_passes_when_zscore_within_bounds(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal(signal_type="Long")]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=2)  # z ~ -0.13, LONG_Z_MIN(-2.0) üstünde
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    sig = await _get_test_signal()
    assert sig is not None
    assert sig.z_score_entry is not None and sig.z_score_entry > -2.0


# ============================================================
# VPMV ratio filtresi
# ============================================================


@pytest.mark.asyncio
async def test_vpmv_ratio_below_min_skips_signal(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch, pre_avg=50.0, vpmv_sig=50.0)  # ratio=1.0 < MIN_RATIO(1.3)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    assert await _get_test_signal() is None


@pytest.mark.asyncio
async def test_vpmv_ratio_above_min_proceeds(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch, pre_avg=50.0, vpmv_sig=70.0)  # ratio=1.4 >= 1.3
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})
    assert await _get_test_signal() is not None


# ============================================================
# is_confluence (HA_Cross + Long + 15m + HTF hizalanma)
# ============================================================


def _tf_buffer(ha_bull: bool, st_bull: bool):
    ha_o, ha_c = (100.0, 101.0) if ha_bull else (101.0, 100.0)
    return pd.DataFrame(
        {
            "ha_open": [ha_o],
            "ha_close": [ha_c],
            "st_direction": [-1.0 if st_bull else 1.0],
        }
    )


@pytest.mark.asyncio
async def test_is_confluence_true_when_htf_aligned(monkeypatch, clean_test_data):
    _patch_signal_engine(
        monkeypatch, {"ha_crossover": [_tech_signal(indicators="HA_Cross", signal_type="Long")]}
    )
    _patch_vpmv(monkeypatch)
    buffers = {tf: _tf_buffer(ha_bull=True, st_bull=True) for tf in ["4h", "6h", "8h", "12h", "1d"]}
    buffers["1h"] = _tf_buffer(ha_bull=True, st_bull=True)
    df = _build_price_df(crash=2, n=220)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(
        TEST_SYMBOL, df, ref_df, "15m", symbol_buffers=buffers
    )
    sig = await _get_test_signal()
    assert sig is not None
    assert sig.is_confluence is True


@pytest.mark.asyncio
async def test_is_confluence_false_when_htf_not_aligned(monkeypatch, clean_test_data):
    _patch_signal_engine(
        monkeypatch, {"ha_crossover": [_tech_signal(indicators="HA_Cross", signal_type="Long")]}
    )
    _patch_vpmv(monkeypatch)
    buffers = {tf: _tf_buffer(ha_bull=False, st_bull=False) for tf in ["4h", "6h", "8h", "12h", "1d"]}
    buffers["1h"] = _tf_buffer(ha_bull=False, st_bull=False)
    df = _build_price_df(crash=2, n=220)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(
        TEST_SYMBOL, df, ref_df, "15m", symbol_buffers=buffers
    )
    sig = await _get_test_signal()
    assert sig is not None
    assert sig.is_confluence is False


# ============================================================
# dry_run
# ============================================================


@pytest.mark.asyncio
async def test_dry_run_does_not_write_signal(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(
        TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={}, dry_run=True
    )
    assert await _get_test_signal() is None


@pytest.mark.asyncio
async def test_non_dry_run_writes_signal_for_comparison(monkeypatch, clean_test_data):
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)
    await process_and_enrich_signals(
        TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={}, dry_run=False
    )
    assert await _get_test_signal() is not None


# ============================================================
# Uçtan uca zenginleştirme + gerçek DB yazımı
# ============================================================


@pytest.mark.asyncio
async def test_happy_path_populates_enriched_fields_for_real(monkeypatch, clean_test_data):
    """VPMV dışındaki TÜM alanlar gerçek hesaplanır: z_score_entry, regime_trend,
    volatility_regime, btc_z_score/btc_trend, devisso_score, vp_score,
    market_structure, candle_pattern, fvg_tfs (Redis boş -> '-'), all_up (DB'de
    önceki sinyal yok -> None)."""
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal(signal_type="Long", price=105.0, atr=2.0)]})
    _patch_vpmv(monkeypatch, vol=80.0, mom=80.0, vlt=80.0, prc=80.0, pre_avg=50.0, vpmv_sig=70.0)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)

    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})

    sig = await _get_test_signal()
    assert sig is not None
    assert sig.symbol == TEST_SYMBOL
    assert sig.interval == "5m"
    assert sig.indicators == "RSI_Cross(9,24)"
    assert sig.signal_type == "Long"
    assert sig.open_price == 105.0
    assert sig.status == "active"
    assert sig.vpms_score == pytest.approx(80.0, abs=0.5)
    assert sig.z_score_entry is not None
    assert sig.regime_trend in ("trending", "ranging", "neutral")
    assert sig.volatility_regime in ("high", "low", "normal")
    assert sig.candle_pattern is not None
    assert sig.market_structure is not None
    assert sig.all_up is None  # bu sembol için önceki sinyal yok


@pytest.mark.asyncio
async def test_metrics_calculator_injection_sets_alpha_beta(monkeypatch, clean_test_data):
    """metrics_calculator enjekte edilirse gerçek calculate_metrics çağrılmaz,
    alpha/beta enjekte edilen değerlerden gelir."""
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)

    async def _fake_metrics(df_p, ref_df_p, interval):
        out = df_p.copy()
        out["alpha"] = 0.123
        out["beta"] = 0.987
        out["sharpe_ratio"] = 1.5
        out["sortino_ratio"] = 1.8
        out["calmar_ratio"] = 2.0
        out["information_ratio"] = 0.5
        return out

    await process_and_enrich_signals(
        TEST_SYMBOL,
        df,
        ref_df,
        "5m",
        symbol_buffers={},
        metrics_calculator=_fake_metrics,
    )

    sig = await _get_test_signal()
    assert sig is not None
    assert sig.alpha == pytest.approx(0.123)
    assert sig.beta == pytest.approx(0.987)


# ============================================================
# Güvenlik: canlı paper-trade fan-out'a HİÇBİR yazım yapılmamalı
# ============================================================


@pytest.mark.asyncio
async def test_no_paper_trade_written_even_with_valid_signal(monkeypatch, clean_test_data):
    """ENABLED_STRATEGIES=[] zorlaması sayesinde geçerli bir sinyal DB'ye
    yazılsa bile hiçbir PaperTradeManager (conf_100/ha_cross/rsi_15m/
    rsi_cross_live) yeni pozisyon açmamalı."""
    _patch_signal_engine(monkeypatch, {"rsi_crossover": [_tech_signal()]})
    _patch_vpmv(monkeypatch)
    df = _build_price_df(crash=2)
    ref_df = _build_price_df(seed=2)

    await process_and_enrich_signals(TEST_SYMBOL, df, ref_df, "5m", symbol_buffers={})

    assert await _get_test_signal() is not None  # sinyal yazıldı
    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.symbol == TEST_SYMBOL)
        )
        assert result.scalars().first() is None  # ama hiçbir paper trade açılmadı
