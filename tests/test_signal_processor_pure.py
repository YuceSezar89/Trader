"""
signals/signal_processor.py — DB/Redis bağımlılığı olmayan saf fonksiyonlar
için birim testler (trim_to_closed_bar, _classify_candle,
_compute_candle_pattern, _compute_devisso_score, _detect_fvg_in_df).
"""

import numpy as np
import pandas as pd
import pytest

from signals.signal_processor import (
    _classify_candle,
    _compute_candle_pattern,
    _compute_devisso_score,
    _detect_fvg_in_df,
    trim_to_closed_bar,
)

# --- trim_to_closed_bar ---


def test_trim_keeps_rows_up_to_closed_open_time():
    df = pd.DataFrame({"open_time": [1, 2, 3, 4, 5], "close": [10, 20, 30, 40, 50]})
    result = trim_to_closed_bar(df, 3)
    assert result["open_time"].tolist() == [1, 2, 3]


def test_trim_returns_empty_when_nothing_matches():
    df = pd.DataFrame({"open_time": [10, 20, 30], "close": [1, 2, 3]})
    result = trim_to_closed_bar(df, 5)
    assert result.empty


def test_trim_passes_through_without_open_time_column():
    df = pd.DataFrame({"close": [1, 2, 3]})
    result = trim_to_closed_bar(df, 5)
    assert len(result) == 3


def test_trim_handles_none_and_empty():
    assert trim_to_closed_bar(None, 5) is None
    assert trim_to_closed_bar(pd.DataFrame(), 5).empty


# --- _classify_candle ---


def test_classify_candle_body_dominant():
    last = pd.Series({"open": 100.0, "high": 110.0, "low": 95.0, "close": 108.0})
    kategori, body_pct, wick_pct = _classify_candle(last)
    assert kategori == "govde"
    assert body_pct == pytest.approx(53.33, abs=0.01)
    assert wick_pct == pytest.approx(46.67, abs=0.01)


def test_classify_candle_upper_wick_dominant():
    last = pd.Series({"open": 100.0, "high": 130.0, "low": 99.0, "close": 101.0})
    kategori, body_pct, wick_pct = _classify_candle(last)
    assert kategori == "ust_fitil"
    assert body_pct == pytest.approx(3.23, abs=0.01)
    assert wick_pct == pytest.approx(96.77, abs=0.01)


def test_classify_candle_zero_range_is_belirsiz():
    last = pd.Series({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    assert _classify_candle(last) == ("belirsiz", None, None)


# --- _compute_candle_pattern ---


def test_candle_pattern_short_df_returns_dash():
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [0, 1, 2], "close": [1.5, 2.5, 3.5]}
    )
    assert _compute_candle_pattern(df) == "-"


def test_candle_pattern_detects_doji():
    rows = []
    price = 100.0
    for _ in range(30):
        rows.append((price, price + 2.0, price - 2.0, price))  # open == close → doji
        price += 0.01
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    result = _compute_candle_pattern(df)
    assert "DOJI" in result


# --- _compute_devisso_score ---


def _make_price_df(n=150, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    price = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"open_time": np.arange(n) * 60_000, "close": price})


def test_devisso_score_returns_value_in_0_100_range():
    df = _make_price_df()
    score = _compute_devisso_score(df)
    assert score is not None
    assert 0.0 <= score <= 100.0


def test_devisso_score_none_when_too_short():
    df = _make_price_df(n=10)
    assert _compute_devisso_score(df) is None


def test_devisso_score_none_on_missing_columns():
    df = pd.DataFrame({"open_time": np.arange(40) * 60_000, "foo": np.arange(40) + 1.0})
    assert _compute_devisso_score(df) is None


# --- _detect_fvg_in_df ---


def _make_bullish_fvg_df():
    rows = [
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 115.0, 100.0, 114.0, 1000.0),
        (114.0, 116.0, 108.0, 115.0, 1000.0),
    ] + [(115.0, 116.0, 112.0, 115.5, 1000.0)] * 5
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_fvg_detects_active_gap_when_entry_inside():
    df = _make_bullish_fvg_df()
    assert _detect_fvg_in_df(df, "Long", entry_price=104.0) is True


def test_fvg_no_match_when_entry_outside_gap():
    df = _make_bullish_fvg_df()
    assert _detect_fvg_in_df(df, "Long", entry_price=200.0) is False


def test_fvg_no_match_for_wrong_direction():
    df = _make_bullish_fvg_df()
    assert _detect_fvg_in_df(df, "Short", entry_price=104.0) is False


def test_fvg_returns_false_for_short_df():
    df = _make_bullish_fvg_df().iloc[:2]
    assert _detect_fvg_in_df(df, "Long", entry_price=100.0) is False
