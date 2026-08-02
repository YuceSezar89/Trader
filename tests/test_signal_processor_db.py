"""
signals/signal_processor.py — DB/Redis'e bağlı yardımcı fonksiyonlar için
testler (_compute_all_up, _get_pt_flag, _get_ranking_snapshot,
_rsi_cross_gate_passed, _count_ha_ultra_confirm, _count_htf_ha_bullish,
_compute_mtf_score, _compute_fvg).

İzolasyon:
- _compute_all_up: gerçek DB'ye TEST% sembollerle yazar, sonunda temizler.
- _count_ha_ultra_confirm / _count_htf_ha_bullish / _compute_mtf_score:
  symbol_buffers parametresiyle çağrılır — Redis'e HİÇ dokunmaz (fonksiyonun
  kendi tasarımı bunu destekliyor).
- _get_pt_flag / _get_ranking_snapshot / _compute_fvg: RedisClient sahte bir
  istemciyle monkeypatch'lenir — gerçek "settings:paper_trade_enabled" /
  "ranking:snapshot" canlı key'lerine hiç dokunulmaz.
- _rsi_cross_gate_passed: modül-içi _RSI_CROSS_SNAPSHOT_CACHE doğrudan
  seed edilir (Redis'e hiç gidilmez).
"""

import json
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from database.engine import get_session
from signals.signal_processor import (
    _PT_FLAG_CACHE,
    _RSI_CROSS_SNAPSHOT_CACHE,
    _compute_all_up,
    _compute_fvg,
    _compute_mtf_score,
    _count_ha_ultra_confirm,
    _count_htf_ha_bullish,
    _get_pt_flag,
    _get_ranking_snapshot,
    _rsi_cross_gate_passed,
)
from tests.conftest import create_test_signal
from utils.redis_client import RedisClient

pytestmark = pytest.mark.database


class _FakeRedisClient:
    def __init__(self, responses=None, raise_exc=None):
        self._responses = responses or {}
        self._raise_exc = raise_exc

    async def get(self, key):
        if self._raise_exc:
            raise self._raise_exc
        return self._responses.get(key)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """_PT_FLAG_CACHE / _RSI_CROSS_SNAPSHOT_CACHE TTL'li modül-seviyesi
    önbellekler — testler arası sızıntıyı önlemek için her testten önce
    "bayat" hale getirilir (ts=0.0), böylece her test kendi Redis/cache
    senaryosunu kontrol eder."""
    _PT_FLAG_CACHE["value"] = "1"
    _PT_FLAG_CACHE["ts"] = 0.0
    _RSI_CROSS_SNAPSHOT_CACHE["data"] = None
    _RSI_CROSS_SNAPSHOT_CACHE["ts"] = 0.0
    yield


# --- _compute_all_up ---


@pytest.mark.asyncio
async def test_compute_all_up_returns_none_when_current_score_missing(clean_test_signals):
    result = await _compute_all_up("TESTUSDT", "5m", "RSI_Cross", "Long", None, 1.0, 1.0, 1.0)
    assert result is None


@pytest.mark.asyncio
async def test_compute_all_up_returns_none_without_prior_signal(clean_test_signals):
    result = await _compute_all_up("TESTUSDT", "5m", "RSI_Cross", "Long", 2.0, 2.0, 2.0, 2.0)
    assert result is None


@pytest.mark.asyncio
async def test_compute_all_up_true_when_all_scores_increased(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(
            session,
            indicators="RSI_Cross",
            opened_at=datetime.now() - timedelta(minutes=10),
            vol_score=1.0,
            mom_score=1.0,
            volat_score=1.0,
            price_score=1.0,
        )

    result = await _compute_all_up("TESTUSDT", "5m", "RSI_Cross", "Long", 2.0, 2.0, 2.0, 2.0)
    assert result is True


@pytest.mark.asyncio
async def test_compute_all_up_false_when_one_score_not_increased(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(
            session,
            indicators="RSI_Cross",
            opened_at=datetime.now() - timedelta(minutes=10),
            vol_score=1.0,
            mom_score=5.0,  # bu düşmeyecek/artmayacak
            volat_score=1.0,
            price_score=1.0,
        )

    result = await _compute_all_up("TESTUSDT", "5m", "RSI_Cross", "Long", 2.0, 2.0, 2.0, 2.0)
    assert result is False


@pytest.mark.asyncio
async def test_compute_all_up_none_when_prior_has_null_score(clean_test_signals):
    async with get_session() as session:
        await create_test_signal(
            session,
            indicators="RSI_Cross",
            opened_at=datetime.now() - timedelta(minutes=10),
            vol_score=1.0,
            mom_score=None,
            volat_score=1.0,
            price_score=1.0,
        )

    result = await _compute_all_up("TESTUSDT", "5m", "RSI_Cross", "Long", 2.0, 2.0, 2.0, 2.0)
    assert result is None


# --- _get_pt_flag ---


@pytest.mark.asyncio
async def test_get_pt_flag_reads_from_redis(monkeypatch):
    monkeypatch.setattr(
        RedisClient,
        "get_client",
        lambda: _FakeRedisClient(responses={"settings:paper_trade_enabled": "0"}),
    )
    flag = await _get_pt_flag()
    assert flag == "0"


@pytest.mark.asyncio
async def test_get_pt_flag_falls_back_to_cache_on_redis_error(monkeypatch):
    _PT_FLAG_CACHE["value"] = "1"
    _PT_FLAG_CACHE["ts"] = 0.0
    monkeypatch.setattr(
        RedisClient, "get_client", lambda: _FakeRedisClient(raise_exc=ConnectionError("down"))
    )
    flag = await _get_pt_flag()
    assert flag == "1"  # önceki cache korunuyor, çökmüyor


@pytest.mark.asyncio
async def test_get_pt_flag_uses_warm_cache_without_calling_redis(monkeypatch):
    _PT_FLAG_CACHE["value"] = "1"
    _PT_FLAG_CACHE["ts"] = time.monotonic()  # taze

    def _boom():
        raise AssertionError("Redis'e hiç gidilmemeliydi")

    monkeypatch.setattr(RedisClient, "get_client", _boom)
    flag = await _get_pt_flag()
    assert flag == "1"


# --- _get_ranking_snapshot ---


@pytest.mark.asyncio
async def test_get_ranking_snapshot_parses_redis_json(monkeypatch):
    rows = [{"symbol": "TESTUSDT", "rsi_cross_combined": 42.0}]
    monkeypatch.setattr(
        RedisClient,
        "get_client",
        lambda: _FakeRedisClient(responses={"ranking:snapshot": json.dumps(rows)}),
    )
    snapshot = await _get_ranking_snapshot()
    assert snapshot == {"TESTUSDT": {"symbol": "TESTUSDT", "rsi_cross_combined": 42.0}}


@pytest.mark.asyncio
async def test_get_ranking_snapshot_falls_back_to_cache_when_empty(monkeypatch):
    _RSI_CROSS_SNAPSHOT_CACHE["data"] = {"OLDUSDT": {"symbol": "OLDUSDT"}}
    _RSI_CROSS_SNAPSHOT_CACHE["ts"] = 0.0
    monkeypatch.setattr(
        RedisClient, "get_client", lambda: _FakeRedisClient(responses={"ranking:snapshot": None})
    )
    snapshot = await _get_ranking_snapshot()
    assert snapshot == {"OLDUSDT": {"symbol": "OLDUSDT"}}


@pytest.mark.asyncio
async def test_get_ranking_snapshot_falls_back_to_cache_on_error(monkeypatch):
    _RSI_CROSS_SNAPSHOT_CACHE["data"] = {"OLDUSDT": {"symbol": "OLDUSDT"}}
    _RSI_CROSS_SNAPSHOT_CACHE["ts"] = 0.0
    monkeypatch.setattr(
        RedisClient, "get_client", lambda: _FakeRedisClient(raise_exc=ConnectionError("down"))
    )
    snapshot = await _get_ranking_snapshot()
    assert snapshot == {"OLDUSDT": {"symbol": "OLDUSDT"}}


# --- _rsi_cross_gate_passed ---


def _seed_ranking_cache(by_symbol: dict):
    _RSI_CROSS_SNAPSHOT_CACHE["data"] = by_symbol
    _RSI_CROSS_SNAPSHOT_CACHE["ts"] = time.monotonic()


def _build_universe(n=40):
    return {
        f"SYM{i}USDT": {"symbol": f"SYM{i}USDT", "rsi_cross_combined": float(i)} for i in range(n)
    }


@pytest.mark.asyncio
async def test_rsi_cross_gate_fail_open_when_symbol_missing():
    _seed_ranking_cache(_build_universe())
    assert await _rsi_cross_gate_passed("NOTINSNAPSHOT", "Long") is True


@pytest.mark.asyncio
async def test_rsi_cross_gate_fail_open_when_score_none():
    universe = _build_universe()
    universe["TESTUSDT"] = {"symbol": "TESTUSDT", "rsi_cross_combined": None}
    _seed_ranking_cache(universe)
    assert await _rsi_cross_gate_passed("TESTUSDT", "Long") is True


@pytest.mark.asyncio
async def test_rsi_cross_gate_fail_open_when_universe_too_small():
    _seed_ranking_cache(_build_universe(n=10))
    assert await _rsi_cross_gate_passed("SYM5USDT", "Long") is True


@pytest.mark.asyncio
async def test_rsi_cross_gate_long_passes_top_tercile():
    universe = _build_universe(n=40)
    universe["TESTUSDT"] = {"symbol": "TESTUSDT", "rsi_cross_combined": 39.0}  # en yüksek
    _seed_ranking_cache(universe)
    assert await _rsi_cross_gate_passed("TESTUSDT", "Long") is True


@pytest.mark.asyncio
async def test_rsi_cross_gate_long_fails_bottom_score():
    universe = _build_universe(n=40)
    universe["TESTUSDT"] = {"symbol": "TESTUSDT", "rsi_cross_combined": 0.0}  # en düşük
    _seed_ranking_cache(universe)
    assert await _rsi_cross_gate_passed("TESTUSDT", "Long") is False


@pytest.mark.asyncio
async def test_rsi_cross_gate_short_uses_inverted_adjustment():
    universe = _build_universe(n=40)
    universe["TESTUSDT"] = {"symbol": "TESTUSDT", "rsi_cross_combined": 0.0}  # Short için iyi
    _seed_ranking_cache(universe)
    assert await _rsi_cross_gate_passed("TESTUSDT", "Short") is True


# --- _count_ha_ultra_confirm (symbol_buffers ile, Redis'siz) ---


def _ha_df(bullish: bool, rows: int = 3):
    if bullish:
        opens, closes = [100.0] * rows, [101.0] * rows
    else:
        opens, closes = [101.0] * rows, [100.0] * rows
    return pd.DataFrame({"ha_open": opens, "ha_close": closes})


@pytest.mark.asyncio
async def test_ha_ultra_confirm_all_bullish_long_counts_three():
    buffers = {"15m": _ha_df(True), "1h": _ha_df(True), "4h": _ha_df(True)}
    result = await _count_ha_ultra_confirm("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 3


@pytest.mark.asyncio
async def test_ha_ultra_confirm_all_bearish_short_counts_three():
    buffers = {"15m": _ha_df(False), "1h": _ha_df(False), "4h": _ha_df(False)}
    result = await _count_ha_ultra_confirm("TESTUSDT", "Short", symbol_buffers=buffers)
    assert result == 3


@pytest.mark.asyncio
async def test_ha_ultra_confirm_mixed_counts_partial():
    buffers = {"15m": _ha_df(True), "1h": _ha_df(False), "4h": _ha_df(True)}
    result = await _count_ha_ultra_confirm("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 2


@pytest.mark.asyncio
async def test_ha_ultra_confirm_none_when_tf_missing():
    buffers = {"15m": _ha_df(True), "1h": None, "4h": _ha_df(True)}
    result = await _count_ha_ultra_confirm("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result is None


@pytest.mark.asyncio
async def test_ha_ultra_confirm_none_when_too_few_rows():
    buffers = {"15m": _ha_df(True, rows=1), "1h": _ha_df(True), "4h": _ha_df(True)}
    result = await _count_ha_ultra_confirm("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result is None


@pytest.mark.asyncio
async def test_ha_ultra_confirm_none_when_columns_missing():
    buffers = {
        "15m": pd.DataFrame({"close": [1.0, 2.0, 3.0]}),
        "1h": _ha_df(True),
        "4h": _ha_df(True),
    }
    result = await _count_ha_ultra_confirm("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result is None


# --- _count_htf_ha_bullish (symbol_buffers ile, Redis'siz) ---


@pytest.mark.asyncio
async def test_htf_ha_bullish_counts_all_five():
    buffers = {tf: _ha_df(True) for tf in ["4h", "6h", "8h", "12h", "1d"]}
    result = await _count_htf_ha_bullish("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 5


@pytest.mark.asyncio
async def test_htf_ha_bullish_skips_missing_tfs():
    buffers = {"4h": _ha_df(True), "6h": None, "8h": _ha_df(True), "12h": None, "1d": _ha_df(True)}
    result = await _count_htf_ha_bullish("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 3


@pytest.mark.asyncio
async def test_htf_ha_bullish_mixed_directions():
    buffers = {
        "4h": _ha_df(True),
        "6h": _ha_df(False),
        "8h": _ha_df(True),
        "12h": _ha_df(False),
        "1d": _ha_df(True),
    }
    result = await _count_htf_ha_bullish("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 3


@pytest.mark.asyncio
async def test_htf_ha_bullish_zero_without_data():
    buffers = {tf: None for tf in ["4h", "6h", "8h", "12h", "1d"]}
    result = await _count_htf_ha_bullish("TESTUSDT", "Long", symbol_buffers=buffers)
    assert result == 0


# --- _compute_mtf_score (symbol_buffers ile, Redis'siz) ---


def _st_df(bullish: bool):
    return pd.DataFrame({"st_direction": [1.0, -1.0 if bullish else 1.0]})


@pytest.mark.asyncio
async def test_mtf_score_no_higher_tfs_returns_100():
    result = await _compute_mtf_score("TESTUSDT", "1d", "Long", symbol_buffers={})
    assert result == 100.0


@pytest.mark.asyncio
async def test_mtf_score_all_confirmed_returns_100():
    buffers = {"15m": _st_df(True), "1h": _st_df(True)}  # interval "5m" -> higher: 15m, 1h
    result = await _compute_mtf_score("TESTUSDT", "5m", "Long", symbol_buffers=buffers)
    assert result == 100.0


@pytest.mark.asyncio
async def test_mtf_score_half_confirmed_returns_50():
    buffers = {"15m": _st_df(True), "1h": _st_df(False)}
    result = await _compute_mtf_score("TESTUSDT", "5m", "Long", symbol_buffers=buffers)
    assert result == 50


@pytest.mark.asyncio
async def test_mtf_score_no_valid_data_fails_open_to_100():
    buffers = {"15m": pd.DataFrame({"close": [1.0]}), "1h": None}
    result = await _compute_mtf_score("TESTUSDT", "5m", "Long", symbol_buffers=buffers)
    assert result == 100.0


# --- _compute_fvg ---


def _bullish_fvg_df():
    rows = [
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 115.0, 100.0, 114.0, 1000.0),
        (114.0, 116.0, 108.0, 115.0, 1000.0),
    ] + [(115.0, 116.0, 112.0, 115.5, 1000.0)] * 5
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


@pytest.mark.asyncio
async def test_compute_fvg_reports_matching_timeframe(monkeypatch):
    async def _fake_mtf_klines(symbol, tf, limit=None):
        return _bullish_fvg_df() if tf == "1h" else None

    monkeypatch.setattr(RedisClient, "get_mtf_klines", _fake_mtf_klines)
    result = await _compute_fvg("TESTUSDT", "Long", entry_price=104.0)
    assert result == "1h"


@pytest.mark.asyncio
async def test_compute_fvg_returns_dash_without_any_match(monkeypatch):
    async def _fake_mtf_klines(symbol, tf, limit=None):
        return None

    monkeypatch.setattr(RedisClient, "get_mtf_klines", _fake_mtf_klines)
    result = await _compute_fvg("TESTUSDT", "Long", entry_price=104.0)
    assert result == "-"
