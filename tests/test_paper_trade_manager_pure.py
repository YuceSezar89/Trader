"""
signals/paper_trade_manager.py — DB'ye dokunmayan saf statik metodlar için
birim testler (_static_trail_distance, _apply_close).
"""

from types import SimpleNamespace

from signals.paper_trade_manager import PaperTradeManager


def make_trade(
    signal_type="Long",
    entry_price=100.0,
    atr=None,
    stop_loss_price=None,
    strategy="rsi_cross_live",
    position_usd=100.0,
):
    return SimpleNamespace(
        signal_type=signal_type,
        entry_price=entry_price,
        atr=atr,
        stop_loss_price=stop_loss_price,
        strategy=strategy,
        position_usd=position_usd,
        symbol="TESTUSDT",
        interval="5m",
        status="open",
        closed_at=None,
        exit_price=None,
        close_reason=None,
        pnl_pct=None,
        fee_usd=None,
        pnl_usd=None,
        balance_after=None,
    )


def make_portfolio(balance=10000.0, peak_balance=10000.0, max_drawdown_pct=0.0):
    return SimpleNamespace(
        balance=balance,
        peak_balance=peak_balance,
        max_drawdown_pct=max_drawdown_pct,
        total_pnl_usd=0.0,
        total_trades=0,
        winning_trades=0,
        updated_at=None,
    )


# --- _static_trail_distance ---


def test_static_trail_distance_uses_entry_sl_diff():
    trade = make_trade(atr=2.0, stop_loss_price=95.0, entry_price=100.0)
    assert PaperTradeManager._static_trail_distance(trade) == 5.0


def test_static_trail_distance_falls_back_to_pct_of_entry():
    trade = make_trade(atr=None, stop_loss_price=None, entry_price=100.0)
    assert PaperTradeManager._static_trail_distance(trade) == 0.5


def test_static_trail_distance_zero_without_entry_price():
    trade = make_trade(atr=None, stop_loss_price=None, entry_price=None)
    assert PaperTradeManager._static_trail_distance(trade) == 0.0


# --- _apply_close ---


def test_apply_close_long_win_updates_trade_and_portfolio():
    trade = make_trade(signal_type="Long", entry_price=100.0, strategy="rsi_cross_live")
    portfolio = make_portfolio(balance=10000.0, peak_balance=10000.0)

    PaperTradeManager._apply_close(trade, exit_price=110.0, reason="take_profit", portfolio=portfolio)

    assert trade.status == "closed"
    assert trade.close_reason == "take_profit"
    assert trade.pnl_pct == 10.0
    assert trade.fee_usd == 0.1  # 100 * 0.0005 * 2
    assert trade.pnl_usd == 29.9  # (10/100)*100*3 leverage - fee
    assert portfolio.total_trades == 1
    assert portfolio.winning_trades == 1
    assert portfolio.balance == 10029.9
    assert portfolio.peak_balance == 10029.9
    assert portfolio.max_drawdown_pct == 0.0
    assert trade.balance_after == 10029.9


def test_apply_close_short_loss_tracks_drawdown_without_winning_trade():
    trade = make_trade(signal_type="Long", entry_price=100.0, strategy="unknown_strategy")
    portfolio = make_portfolio(balance=9000.0, peak_balance=10000.0, max_drawdown_pct=0.0)

    PaperTradeManager._apply_close(trade, exit_price=95.0, reason="stop_loss", portfolio=portfolio)

    assert trade.pnl_pct == -5.0
    assert trade.pnl_usd == -5.1  # (-5/100)*100*1.0 leverage - 0.1 fee
    assert portfolio.total_trades == 1
    assert portfolio.winning_trades == 0
    assert portfolio.balance == 8994.9
    assert portfolio.peak_balance == 10000.0  # düşmedi, güncellenmemeli
    assert round(portfolio.max_drawdown_pct, 3) == 10.051


def test_apply_close_without_portfolio_does_not_crash():
    trade = make_trade(signal_type="Long", entry_price=100.0)

    PaperTradeManager._apply_close(trade, exit_price=105.0, reason="manual", portfolio=None)

    assert trade.status == "closed"
    assert trade.pnl_pct == 5.0
    assert trade.balance_after is None
