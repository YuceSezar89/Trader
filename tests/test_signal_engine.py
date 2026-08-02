"""
signals/signal_engine.py — SignalEngine için testler (zero-test kritik modül,
Fable 5 mimari denetiminin bulgu #2 kapsamına giriyor).

Kapsam:
- rsi_crossover_signal / ma200_crossover_signal / supertrend_signal /
  ha_crossover_signal: crossover mantığı gerçek üretim koduyla test edilir,
  gösterge kolonları (rsi_9/rsi_24/ma200/st_direction/ha_open/ha_close)
  DOĞRUDAN sağlanır (calculate_rsi/calculate_supertrend'i yeniden hesaplatmak
  yerine) — bu, fonksiyonların KENDİ tasarımının desteklediği bir yol (kolon
  varsa yeniden hesaplamaz) ve crossover karar mantığını hâlâ gerçek kodla,
  deterministik biçimde test eder.
- symbol/interval verilince SignalFilter entegrasyonu gerçek DB'ye karşı
  (TEST-prefix'li benzersiz sembol, bkz. test_signal_filter.py deseni).
- calculate_all_signals: dispatch, DISABLED_SIGNAL_TYPES filtresi, hata
  toparlama davranışı (bir görev patlarsa TÜM batch "ERROR" döner — gerçek
  kod davranışı, gizli bir dizi hatası değil, burada belgeleniyor).
- replay_filter_state: temel akış + erken-çıkış dalları.
"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from signals.signal_engine import SignalEngine

pytestmark = pytest.mark.database


@pytest.fixture
def engine():
    return SignalEngine()


# ============================================================
# rsi_crossover_signal
# ============================================================


def _rsi_df(rsi9_prev, rsi24_prev, rsi9_now, rsi24_now, close=100.0):
    return pd.DataFrame(
        {
            "open_time": [0, 300_000],
            "close": [close, close + 1.0],
            "low": [close - 1.0, close],
            "high": [close + 1.0, close + 2.0],
            "rsi_9": [rsi9_prev, rsi9_now],
            "rsi_24": [rsi24_prev, rsi24_now],
        }
    )


@pytest.mark.asyncio
async def test_rsi_crossover_long_signal(engine):
    df = _rsi_df(rsi9_prev=20.0, rsi24_prev=30.0, rsi9_now=40.0, rsi24_now=35.0)
    result = await engine.rsi_crossover_signal(df)
    assert len(result) == 1
    assert result[0]["signal_type"] == "Long"
    assert result[0]["indicators"] == "RSI_Cross(9,24)"


@pytest.mark.asyncio
async def test_rsi_crossover_short_signal(engine):
    df = _rsi_df(rsi9_prev=40.0, rsi24_prev=30.0, rsi9_now=20.0, rsi24_now=25.0)
    result = await engine.rsi_crossover_signal(df)
    assert len(result) == 1
    assert result[0]["signal_type"] == "Short"


@pytest.mark.asyncio
async def test_rsi_crossover_no_cross_returns_empty(engine):
    df = _rsi_df(rsi9_prev=20.0, rsi24_prev=30.0, rsi9_now=25.0, rsi24_now=35.0)
    assert await engine.rsi_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_rsi_crossover_nan_returns_empty(engine):
    df = _rsi_df(rsi9_prev=20.0, rsi24_prev=30.0, rsi9_now=np.nan, rsi24_now=35.0)
    assert await engine.rsi_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_rsi_crossover_insufficient_data_returns_empty(engine):
    df = _rsi_df(20.0, 30.0, 40.0, 35.0).iloc[:1]
    assert await engine.rsi_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_rsi_crossover_missing_columns_returns_empty(engine):
    df = pd.DataFrame({"close": [100.0, 101.0], "open_time": [0, 300_000]})
    assert await engine.rsi_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_rsi_crossover_filter_rejects_first_signal(engine, sym):
    df = _rsi_df(rsi9_prev=20.0, rsi24_prev=30.0, rsi9_now=40.0, rsi24_now=35.0)
    result = await engine.rsi_crossover_signal(df, symbol=sym, interval="1h")
    assert result == []  # ilk sinyal, karşıt referans yok — SignalFilter reddeder


@pytest.mark.asyncio
async def test_rsi_crossover_filter_passes_after_opposite_reference(engine, sym):
    # Önce bir Short referansı oluştur (SignalFilter'ın kendi davranışı, bkz. test_signal_filter.py)
    short_df = _rsi_df(rsi9_prev=40.0, rsi24_prev=30.0, rsi9_now=20.0, rsi24_now=25.0, close=100.0)
    await engine.rsi_crossover_signal(short_df, symbol=sym, interval="1h")

    long_df = _rsi_df(rsi9_prev=20.0, rsi24_prev=30.0, rsi9_now=40.0, rsi24_now=35.0, close=200.0)
    result = await engine.rsi_crossover_signal(long_df, symbol=sym, interval="1h")
    assert len(result) == 1
    assert result[0]["signal_type"] == "Long"


# ============================================================
# ma200_crossover_signal
# ============================================================


def _ma200_df(prev_close, prev_ma200, curr_close, curr_ma200):
    return pd.DataFrame(
        {
            "open_time": [0, 300_000],
            "close": [prev_close, curr_close],
            "ma200": [prev_ma200, curr_ma200],
            "high": [curr_close + 1.0, curr_close + 1.0],
            "low": [curr_close - 1.0, curr_close - 1.0],
        }
    )


@pytest.mark.asyncio
async def test_ma200_crossover_long_signal(engine):
    df = _ma200_df(prev_close=95.0, prev_ma200=100.0, curr_close=105.0, curr_ma200=100.0)
    result = await engine.ma200_crossover_signal(df)
    assert len(result) == 1
    assert result[0]["signal_type"] == "Long"
    assert result[0]["indicators"] == "MA200_Cross"


@pytest.mark.asyncio
async def test_ma200_crossover_short_signal(engine):
    df = _ma200_df(prev_close=105.0, prev_ma200=100.0, curr_close=95.0, curr_ma200=100.0)
    result = await engine.ma200_crossover_signal(df)
    assert result[0]["signal_type"] == "Short"


@pytest.mark.asyncio
async def test_ma200_crossover_no_cross_returns_empty(engine):
    df = _ma200_df(prev_close=105.0, prev_ma200=100.0, curr_close=106.0, curr_ma200=100.0)
    assert await engine.ma200_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_ma200_crossover_insufficient_data_returns_empty(engine):
    df = _ma200_df(95.0, 100.0, 105.0, 100.0).iloc[:1]
    assert await engine.ma200_crossover_signal(df) == []


# ============================================================
# supertrend_signal
# ============================================================


def _st_df(dir_prev, dir_now, high=101.0, low=99.0):
    """high/low SADECE son bar için anlamlı (SignalFilter bunları okur)."""
    return pd.DataFrame(
        {
            "open_time": [0, 300_000],
            "close": [100.0, 101.0],
            "high": [200.0, high],
            "low": [1.0, low],
            "st_direction": [dir_prev, dir_now],
        }
    )


@pytest.mark.asyncio
async def test_supertrend_long_flip(engine):
    df = _st_df(dir_prev=1.0, dir_now=-1.0)
    result = await engine.supertrend_signal(df)
    assert len(result) == 1
    assert result[0]["signal_type"] == "Long"
    assert result[0]["indicators"] == "Supertrend(10,3.0)"


@pytest.mark.asyncio
async def test_supertrend_short_flip(engine):
    df = _st_df(dir_prev=-1.0, dir_now=1.0)
    result = await engine.supertrend_signal(df)
    assert result[0]["signal_type"] == "Short"


@pytest.mark.asyncio
async def test_supertrend_no_flip_returns_empty(engine):
    df = _st_df(dir_prev=-1.0, dir_now=-1.0)
    assert await engine.supertrend_signal(df) == []


@pytest.mark.asyncio
async def test_supertrend_nan_direction_returns_empty(engine):
    df = _st_df(dir_prev=1.0, dir_now=np.nan)
    assert await engine.supertrend_signal(df) == []


@pytest.mark.asyncio
async def test_supertrend_repeat_same_direction_suppressed(engine, sym):
    # SignalFilter ilk-sinyal-her-zaman-geçersiz kuralına göre referans zinciri kurulur
    # (reddedilen denemeler de signal_filter_events'e kaydedilir, bkz. test_signal_filter.py):
    # 1) İlk Long (reddedilir ama high=100 referans olarak kaydedilir)
    await engine.supertrend_signal(
        _st_df(dir_prev=1.0, dir_now=-1.0, high=100.0, low=95.0), symbol=sym, interval="1h"
    )
    # 2) Short: karşıt (Long) referans low=95 var → low(90) < 95 geçerli, high=99 kaydedilir
    short_result = await engine.supertrend_signal(
        _st_df(dir_prev=-1.0, dir_now=1.0, high=99.0, low=90.0), symbol=sym, interval="1h"
    )
    assert len(short_result) == 1

    # 3) Long: karşıt (Short) referans high=99 var → high(105) > 99 geçerli — ilk kabul edilen Long
    long_flip = _st_df(dir_prev=1.0, dir_now=-1.0, high=105.0, low=92.0)
    first = await engine.supertrend_signal(long_flip, symbol=sym, interval="1h")
    assert len(first) == 1  # _st_last_valid=Long kaydedildi

    # 4) Aynı Long flip'i tekrar: filtre yine geçer (aynı Short referansına göre) ama
    # _st_last_valid aynı yönü görüp bastırmalı — trend devam, yeni sinyal değil.
    second = await engine.supertrend_signal(long_flip, symbol=sym, interval="1h")
    assert second == []


# ============================================================
# ha_crossover_signal
# ============================================================


def _ha_df(prev_bull: bool, curr_bull: bool):
    po, pc = (100.0, 101.0) if prev_bull else (101.0, 100.0)
    co, cc = (100.0, 101.0) if curr_bull else (101.0, 100.0)
    return pd.DataFrame(
        {
            "open_time": [0, 300_000],
            "ha_open": [po, co],
            "ha_close": [pc, cc],
            "close": [pc, cc],
            "high": [102.0, 102.0],
            "low": [99.0, 99.0],
        }
    )


@pytest.mark.asyncio
async def test_ha_crossover_long_flip(engine):
    df = _ha_df(prev_bull=False, curr_bull=True)
    result = await engine.ha_crossover_signal(df)
    assert len(result) == 1
    assert result[0]["signal_type"] == "Long"
    assert result[0]["indicators"] == "HA_Cross"


@pytest.mark.asyncio
async def test_ha_crossover_short_flip(engine):
    df = _ha_df(prev_bull=True, curr_bull=False)
    result = await engine.ha_crossover_signal(df)
    assert result[0]["signal_type"] == "Short"


@pytest.mark.asyncio
async def test_ha_crossover_no_flip_returns_empty(engine):
    df = _ha_df(prev_bull=True, curr_bull=True)
    assert await engine.ha_crossover_signal(df) == []


@pytest.mark.asyncio
async def test_ha_crossover_insufficient_data_returns_empty(engine):
    df = _ha_df(False, True).iloc[:1]
    assert await engine.ha_crossover_signal(df) == []


# ============================================================
# calculate_all_signals
# ============================================================


def _all_signals_compatible_df():
    """4 sinyal metodunun da (RSI/MA200/Supertrend/HA) gerekli kolonlarını
    barındıran, hiçbirinin crossover üretmediği nötr bir df."""
    return pd.DataFrame(
        {
            "open_time": [0, 300_000],
            "close": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 99.5],
            "ma200": [90.0, 90.0],  # her iki barda da close > ma200 -> cross yok
            "rsi_9": [50.0, 51.0],
            "rsi_24": [40.0, 41.0],  # fast hep > slow -> cross yok
            "st_direction": [-1.0, -1.0],  # flip yok
            "ha_open": [100.0, 100.0],
            "ha_close": [101.0, 101.0],  # ikisi de bullish -> flip yok
        }
    )


@pytest.mark.asyncio
async def test_calculate_all_signals_excludes_disabled_by_default(engine):
    assert "ma200_crossover" in Config.DISABLED_SIGNAL_TYPES
    df = _all_signals_compatible_df()
    result = await engine.calculate_all_signals(df)
    assert "ma200_crossover" not in result
    assert set(result.keys()) == {"rsi_crossover", "supertrend", "ha_crossover"}


@pytest.mark.asyncio
async def test_calculate_all_signals_explicit_disabled_type_still_excluded(engine):
    df = _all_signals_compatible_df()
    result = await engine.calculate_all_signals(df, signal_types=["ma200_crossover"])
    assert result == {}


@pytest.mark.asyncio
async def test_calculate_all_signals_respects_explicit_subset(engine):
    df = _all_signals_compatible_df()
    result = await engine.calculate_all_signals(df, signal_types=["rsi_crossover"])
    assert set(result.keys()) == {"rsi_crossover"}


@pytest.mark.asyncio
async def test_calculate_all_signals_empty_signal_types_returns_empty(engine):
    df = _all_signals_compatible_df()
    assert await engine.calculate_all_signals(df, signal_types=[]) == {}


@pytest.mark.asyncio
async def test_calculate_all_signals_task_failure_marks_whole_batch_error(engine, monkeypatch):
    """Gerçek kod davranışı: bir görev exception fırlatırsa, except bloğu TÜM
    batch'i (başarılı olacak diğer görevler dahil) "ERROR" olarak işaretler —
    gizli bir test hatası değil, calculate_all_signals'ın kendi tasarımı."""

    async def _boom(df, symbol="", interval=""):
        raise ValueError("boom")

    monkeypatch.setattr(engine, "rsi_crossover_signal", _boom)
    df = _all_signals_compatible_df()
    result = await engine.calculate_all_signals(df, signal_types=["rsi_crossover", "ha_crossover"])
    assert result == {"rsi_crossover": "ERROR", "ha_crossover": "ERROR"}


# ============================================================
# replay_filter_state
# ============================================================


@pytest.mark.asyncio
async def test_replay_filter_state_empty_df_returns_zero(engine):
    assert await engine.replay_filter_state(pd.DataFrame(), "TESTUSDT", "1h", 0) == 0


@pytest.mark.asyncio
async def test_replay_filter_state_missing_open_time_returns_zero(engine):
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert await engine.replay_filter_state(df, "TESTUSDT", "1h", 0) == 0


@pytest.mark.asyncio
async def test_replay_filter_state_replay_point_beyond_data_returns_zero(engine, sym):
    df = _all_signals_compatible_df()
    result = await engine.replay_filter_state(df, sym, "1h", replay_from_ms=10_000_000)
    assert result == 0


@pytest.mark.asyncio
async def test_replay_filter_state_replays_all_bars_from_start(engine, sym):
    df = _all_signals_compatible_df()
    result = await engine.replay_filter_state(df, sym, "1h", replay_from_ms=0)
    assert result == len(df)


@pytest.mark.asyncio
async def test_replay_filter_state_replays_partial_range(engine, sym):
    df = _all_signals_compatible_df()
    result = await engine.replay_filter_state(df, sym, "1h", replay_from_ms=300_000)
    assert result == 1  # sadece son bar (open_time=300_000) replay_from_ms >= koşulunu sağlıyor
