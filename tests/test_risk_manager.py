"""
signals/risk_manager.py — saf statik metodlar için birim testler
(_trail_distance, _early_exit_check, _update_trailing).
DB bağlantısı gerekmez, SimpleNamespace ile Signal taklit edilir.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

from config import Config
from signals.risk_manager import RiskManager


def make_signal(
    signal_type="Long",
    atr=None,
    sl_multiplier=None,
    stop_loss_price=None,
    take_profit_price=None,
    open_price=100.0,
    interval="5m",
    opened_at=None,
    trailing_stop_price=None,
):
    return SimpleNamespace(
        signal_type=signal_type,
        atr=atr,
        sl_multiplier=sl_multiplier,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        open_price=open_price,
        interval=interval,
        opened_at=opened_at or datetime.now(),
        trailing_stop_price=trailing_stop_price,
    )


# --- _trail_distance ---


def test_trail_distance_uses_atr_and_multiplier():
    sig = make_signal(atr=2.0, sl_multiplier=1.5)
    assert RiskManager._trail_distance(sig) == 3.0


def test_trail_distance_falls_back_to_sl_open_diff():
    sig = make_signal(atr=None, sl_multiplier=None, stop_loss_price=95.0, open_price=100.0)
    assert RiskManager._trail_distance(sig) == 5.0


def test_trail_distance_falls_back_to_pct_of_open():
    sig = make_signal(atr=None, sl_multiplier=None, stop_loss_price=None, open_price=100.0)
    assert RiskManager._trail_distance(sig) == 0.5


# --- _early_exit_check ---


def test_early_exit_disabled_by_config(monkeypatch):
    monkeypatch.setitem(Config.VPM, "EARLY_EXIT_ENABLED", False)
    sig = make_signal(atr=2.0, interval="5m", opened_at=datetime.now())
    assert RiskManager._early_exit_check(sig, price=50.0) is False


def test_early_exit_ignores_unsupported_interval():
    sig = make_signal(atr=2.0, interval="1h", opened_at=datetime.now())
    assert RiskManager._early_exit_check(sig, price=50.0) is False


def test_early_exit_requires_atr():
    sig = make_signal(atr=None, interval="5m", opened_at=datetime.now())
    assert RiskManager._early_exit_check(sig, price=50.0) is False


def test_early_exit_expired_after_max_bars():
    old_time = datetime.now() - timedelta(minutes=5 * 20)  # 20 bar (5m), max 10
    sig = make_signal(atr=2.0, interval="5m", opened_at=old_time, open_price=100.0)
    assert RiskManager._early_exit_check(sig, price=50.0) is False


def test_early_exit_triggers_long_on_adverse_move():
    sig = make_signal(
        signal_type="Long", atr=2.0, interval="5m", opened_at=datetime.now(), open_price=100.0
    )
    # adverse = (price - entry) / atr = (96 - 100) / 2 = -2.0 <= -1.5
    assert RiskManager._early_exit_check(sig, price=96.0) is True


def test_early_exit_no_trigger_long_within_threshold():
    sig = make_signal(
        signal_type="Long", atr=2.0, interval="5m", opened_at=datetime.now(), open_price=100.0
    )
    # adverse = (99 - 100) / 2 = -0.5 > -1.5
    assert RiskManager._early_exit_check(sig, price=99.0) is False


def test_early_exit_triggers_short_on_adverse_move():
    sig = make_signal(
        signal_type="Short", atr=2.0, interval="5m", opened_at=datetime.now(), open_price=100.0
    )
    # adverse = (entry - price) / atr = (100 - 104) / 2 = -2.0 <= -1.5
    assert RiskManager._early_exit_check(sig, price=104.0) is True


# --- _update_trailing (early-exit + trailing entegrasyonu) ---


def test_update_trailing_early_exit_overrides_normal_flow():
    sig = make_signal(
        signal_type="Long",
        atr=2.0,
        interval="5m",
        opened_at=datetime.now(),
        open_price=100.0,
        stop_loss_price=90.0,
        take_profit_price=110.0,
    )
    # Fiyat SL/TP aralığında ama early-exit MAE eşiğini aşıyor
    result = RiskManager._update_trailing(sig, price=96.0)
    assert result == "stop_loss"


def test_update_trailing_delegates_when_no_early_exit():
    sig = make_signal(
        signal_type="Long",
        atr=None,
        sl_multiplier=None,
        interval="1h",  # early-exit devre dışı bırakır
        opened_at=datetime.now(),
        open_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
    )
    result = RiskManager._update_trailing(sig, price=94.0)
    assert result == "stop_loss"
