"""
signals/trailing.py — update_trailing() için saf birim testler.
DB/Redis bağımlılığı yok, SimpleNamespace ile pozisyon nesnesi taklit edilir.
"""

from types import SimpleNamespace

from signals.trailing import update_trailing


def make_pos(signal_type, sl=None, tp=None, trail=None):
    return SimpleNamespace(
        signal_type=signal_type,
        stop_loss_price=sl,
        take_profit_price=tp,
        trailing_stop_price=trail,
    )


# --- Long ---


def test_long_stop_loss_triggers():
    pos = make_pos("Long", sl=95.0, tp=110.0)
    result = update_trailing(pos, price=94.0, dist=2.0)
    assert result == "stop_loss"


def test_long_take_profit_activates_trailing_no_close():
    pos = make_pos("Long", sl=95.0, tp=110.0)
    result = update_trailing(pos, price=111.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 109.0


def test_long_between_sl_and_tp_no_change():
    pos = make_pos("Long", sl=95.0, tp=110.0)
    result = update_trailing(pos, price=100.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price is None


def test_long_trailing_updates_upward():
    pos = make_pos("Long", sl=95.0, tp=110.0, trail=108.0)
    result = update_trailing(pos, price=115.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 113.0


def test_long_trailing_does_not_move_down():
    pos = make_pos("Long", sl=95.0, tp=110.0, trail=113.0)
    result = update_trailing(pos, price=114.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 113.0


def test_long_trailing_stop_triggers():
    pos = make_pos("Long", sl=95.0, tp=110.0, trail=113.0)
    result = update_trailing(pos, price=113.0, dist=2.0)
    assert result == "trailing_stop"


# --- Short ---


def test_short_stop_loss_triggers():
    pos = make_pos("Short", sl=105.0, tp=90.0)
    result = update_trailing(pos, price=106.0, dist=2.0)
    assert result == "stop_loss"


def test_short_take_profit_activates_trailing_no_close():
    pos = make_pos("Short", sl=105.0, tp=90.0)
    result = update_trailing(pos, price=89.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 91.0


def test_short_trailing_updates_downward():
    pos = make_pos("Short", sl=105.0, tp=90.0, trail=91.0)
    result = update_trailing(pos, price=85.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 87.0


def test_short_trailing_does_not_move_up():
    pos = make_pos("Short", sl=105.0, tp=90.0, trail=87.0)
    result = update_trailing(pos, price=86.0, dist=2.0)
    assert result is None
    assert pos.trailing_stop_price == 87.0


def test_short_trailing_stop_triggers():
    pos = make_pos("Short", sl=105.0, tp=90.0, trail=87.0)
    result = update_trailing(pos, price=87.0, dist=2.0)
    assert result == "trailing_stop"


def test_no_stop_loss_set_does_not_crash():
    pos = make_pos("Long", sl=None, tp=110.0)
    result = update_trailing(pos, price=50.0, dist=2.0)
    assert result is None
