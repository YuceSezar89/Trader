"""
İki lever'ın kabaca etkisini ölçer (canlı A/B testinden önce hızlı tahmin):

  Lever 1 — TP çarpanını düşürmek: gerçek fiyat barlarını (opened_at->closed_at)
  tarayıp, alternatif (daha düşük) tp_multiplier ile SL/trailing/TP mantığını
  yeniden oynatır. Sadece orijinal kapanış ANINA KADAR olan barları kullanır
  (daha düşük TP zaten o pencerede tetiklenmiş olurdu, daha geç tetiklenmez).

  Lever 2 — "reversal" kapanışını kaldırmak: close_reason='reversal' olan
  sinyalleri opened_at'ten itibaren TIMEOUT_HOURS ufkuna kadar ileri barlarla
  yeniden oynatır, gerçek SL/TP/trailing ne zaman tetiklenirdi diye bakar.

  Bar-seviyesinde konservatif kural: SL önce kontrol edilir (bar'ın en kötü
  ucu), sonra TP/trailing güncellemesi bar'ın en iyi ucuyla yapılır.

Kullanım: python -m research.pattern_lab.tp_sl_lever_sim
"""

import types
import warnings
from datetime import timedelta

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.rsi_cross_allup_candleshape_bt import (
    _bad_symbols,
    _classify_candle,
    _vpmv_components,
)
from signals.signal_lifecycle_manager import TIMEOUT_HOURS, _DEFAULT_TIMEOUT, _calc_pnl
from signals.trailing import update_trailing

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_POSITION_USD = 100.0


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )


def _fetch_entry_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    from indicators.core import truncate_after_gap

    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS open_time, open, high, low, close, volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
            "ORDER BY timestamp DESC LIMIT 220",
            (symbol, opened_at),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket AS open_time, open, high, low, close, volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT 220",
            (symbol, opened_at),
        )
    rows = cur.fetchall()
    if not rows or len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)
    df = truncate_after_gap(df)
    if len(df) < 60:
        return None
    return df


def _fetch_signals(cur, indicator: str, direction: str, exclude_symbols: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, opened_at, closed_at, open_price, atr,
               sl_multiplier, tp_multiplier, close_reason, realized_pnl
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
          AND atr IS NOT NULL AND sl_multiplier IS NOT NULL AND tp_multiplier IS NOT NULL
        ORDER BY symbol, opened_at
        """,
        (indicator, direction),
    )
    rows = cur.fetchall()
    cols = [
        "id", "symbol", "interval", "opened_at", "closed_at", "open_price", "atr",
        "sl_multiplier", "tp_multiplier", "close_reason", "realized_pnl",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["signal_type"] = direction
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)].reset_index(drop=True)
    return df


def _fetch_path_bars(cur, symbol: str, interval: str, start, end) -> pd.DataFrame:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS ts, open, high, low, close "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp > %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            (symbol, start, end),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return pd.DataFrame()
        cur.execute(
            f"SELECT bucket AS ts, open, high, low, close FROM {cagg} "
            f"WHERE symbol=%s AND bucket > %s AND bucket <= %s ORDER BY bucket ASC",
            (symbol, start, end),
        )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])


def _replay(sig_type: str, open_price: float, sl_price: float, tp_price: float, dist: float, bars: pd.DataFrame):
    """SL/trailing/TP mantığını bar bar oynatır. (reason, close_price, ts) döner ya da None."""
    pos = types.SimpleNamespace(
        signal_type=sig_type,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        trailing_stop_price=None,
    )
    for row in bars.itertuples():
        if sig_type == "Long":
            if pos.trailing_stop_price is None:
                if row.low <= sl_price:
                    return "stop_loss", sl_price, row.ts
                if row.high >= tp_price:
                    pos.trailing_stop_price = row.high - dist
                    if row.low <= pos.trailing_stop_price:
                        return "trailing_stop", pos.trailing_stop_price, row.ts
            else:
                new_trail = row.high - dist
                if new_trail > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail
                if row.low <= pos.trailing_stop_price:
                    return "trailing_stop", pos.trailing_stop_price, row.ts
        else:
            if pos.trailing_stop_price is None:
                if row.high >= sl_price:
                    return "stop_loss", sl_price, row.ts
                if row.low <= tp_price:
                    pos.trailing_stop_price = row.low + dist
                    if row.high >= pos.trailing_stop_price:
                        return "trailing_stop", pos.trailing_stop_price, row.ts
            else:
                new_trail = row.low + dist
                if new_trail < pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail
                if row.high >= pos.trailing_stop_price:
                    return "trailing_stop", pos.trailing_stop_price, row.ts
    return None


def _collect_subset(conn, indicator: str, direction: str, bad_symbols: set[str], filter_govde_allup: bool) -> pd.DataFrame:
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, indicator, direction, bad_symbols)
    if not filter_govde_allup:
        return sig_df

    # all_up + gövde filtresi icin entry-zamanindaki bileşenleri hesapla
    records = []
    for row in sig_df.itertuples():
        df = _fetch_entry_bars(cur, row.symbol, row.interval, row.opened_at)
        if df is None:
            continue
        comp = _vpmv_components(df, direction)
        if comp is None:
            continue
        records.append({"id": row.id, **comp})
    comp_df = pd.DataFrame(records)
    if comp_df.empty:
        return comp_df
    merged = sig_df.merge(comp_df, on="id", how="inner")
    merged = merged.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        merged[f"d_{col}"] = merged.groupby("symbol")[col].diff()
    merged = merged.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    merged["all_up"] = (
        (merged["d_vol"] > 0) & (merged["d_mom"] > 0) & (merged["d_volat"] > 0) & (merged["d_price"] > 0)
    )
    return merged[merged["all_up"] & (merged["kategori"] == "govde")].reset_index(drop=True)


def _usd(pnl_pct_sum_over_n, n):
    return _POSITION_USD * pnl_pct_sum_over_n / 100


def lever1_tp_tuning(conn, label: str, sig_df: pd.DataFrame, alt_tp_mults: list[float]) -> None:
    cur = conn.cursor()
    orig_total = sig_df["realized_pnl"].sum()
    alt_totals = {m: 0.0 for m in alt_tp_mults}
    n = 0
    for row in sig_df.itertuples():
        bars = _fetch_path_bars(cur, row.symbol, row.interval, row.opened_at, row.closed_at)
        if bars.empty:
            for m in alt_tp_mults:
                alt_totals[m] += row.realized_pnl
            continue
        n += 1
        sl_dist = float(row.atr) * float(row.sl_multiplier)
        sl_price = row.open_price - sl_dist if row.signal_type == "Long" else row.open_price + sl_dist
        for m in alt_tp_mults:
            tp_dist = float(row.atr) * m
            tp_price = row.open_price + tp_dist if row.signal_type == "Long" else row.open_price - tp_dist
            res = _replay(row.signal_type, row.open_price, sl_price, tp_price, sl_dist, bars)
            if res is None:
                alt_totals[m] += row.realized_pnl  # orijinal pencerede tetiklenmedi -> orijinali koru
            else:
                _, close_price, _ = res
                alt_totals[m] += _calc_pnl(row.signal_type, row.open_price, close_price)

    days = (sig_df["opened_at"].max() - sig_df["opened_at"].min()).total_seconds() / 86400
    per_month_n = len(sig_df) / days * 30 if days > 0 else 0
    print(f"\n[{label}] n={len(sig_df)} (bar verisi bulunan={n})")
    print(f"  orijinal toplam PnL%={orig_total:.1f}  ~$/ay={_usd(orig_total, len(sig_df)) / days * 30:.0f}")
    for m in alt_tp_mults:
        diff = alt_totals[m] - orig_total
        print(
            f"  alt tp_mult={m:<4} toplam PnL%={alt_totals[m]:.1f}  ~$/ay={_usd(alt_totals[m], len(sig_df)) / days * 30:.0f}"
            f"   (fark: {'+' if diff>=0 else ''}{_usd(diff, len(sig_df)) / days * 30:.0f}$/ay)"
        )


def lever2_no_reversal(conn, label: str, sig_df: pd.DataFrame) -> None:
    cur = conn.cursor()
    rev = sig_df[sig_df["close_reason"] == "reversal"].reset_index(drop=True)
    if rev.empty:
        print(f"\n[{label}] reversal ile kapanan sinyal yok.")
        return

    orig_total = rev["realized_pnl"].sum()
    alt_total = 0.0
    n_resolved_early = 0
    n_timeout = 0
    n_no_data = 0
    for row in rev.itertuples():
        hours = TIMEOUT_HOURS.get(row.interval, _DEFAULT_TIMEOUT)
        horizon = row.opened_at + timedelta(hours=hours)
        bars = _fetch_path_bars(cur, row.symbol, row.interval, row.opened_at, horizon)
        if bars.empty:
            alt_total += row.realized_pnl
            n_no_data += 1
            continue
        sl_dist = float(row.atr) * float(row.sl_multiplier)
        sl_price = row.open_price - sl_dist if row.signal_type == "Long" else row.open_price + sl_dist
        tp_dist = float(row.atr) * float(row.tp_multiplier)
        tp_price = row.open_price + tp_dist if row.signal_type == "Long" else row.open_price - tp_dist
        res = _replay(row.signal_type, row.open_price, sl_price, tp_price, sl_dist, bars)
        if res is None:
            alt_total += 0.0  # timeout kuralı: pnl=0
            n_timeout += 1
        else:
            _, close_price, _ = res
            alt_total += _calc_pnl(row.signal_type, row.open_price, close_price)
            n_resolved_early += 1

    days = (sig_df["opened_at"].max() - sig_df["opened_at"].min()).total_seconds() / 86400
    print(f"\n[{label}] reversal n={len(rev)}  (SL/TP/trailing ile çözülen={n_resolved_early}, "
          f"timeout={n_timeout}, veri yok={n_no_data})")
    print(f"  orijinal (reversal) toplam PnL%={orig_total:.1f}  ~$/ay={_usd(orig_total, len(rev)) / days * 30:.0f}")
    diff = alt_total - orig_total
    print(f"  alt (SL/TP'ye bırak) toplam PnL%={alt_total:.1f}  ~$/ay={_usd(alt_total, len(rev)) / days * 30:.0f}"
          f"   (fark: {'+' if diff>=0 else ''}{_usd(diff, len(rev)) / days * 30:.0f}$/ay)")


def main() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    long_df = _collect_subset(conn, "RSI_Cross(9,24)", "Long", bad, filter_govde_allup=True)
    short_df = _collect_subset(conn, "RSI_Cross(9,24)", "Short", bad, filter_govde_allup=False)

    print("\n" + "=" * 78)
    print("LEVER 1 — TP çarpanını düşürmek")
    print("=" * 78)
    lever1_tp_tuning(conn, "RSI_Cross Long (all_up+gövde)", long_df, [1.5, 2.0, 2.5])
    lever1_tp_tuning(conn, "RSI_Cross Short (baseline)", short_df, [1.5, 2.0, 2.5])

    print("\n" + "=" * 78)
    print("LEVER 2 — reversal yerine SL/TP/trailing'e bırakmak")
    print("=" * 78)
    lever2_no_reversal(conn, "RSI_Cross Long (all_up+gövde)", long_df)
    lever2_no_reversal(conn, "RSI_Cross Short (baseline)", short_df)

    conn.close()


if __name__ == "__main__":
    main()
