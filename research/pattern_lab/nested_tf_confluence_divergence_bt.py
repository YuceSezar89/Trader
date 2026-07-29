"""
İç-içe TF kesişimi (15m/45m/90m, her biri bir sonrakinin böleni) + kohort-içi
ayrışma testi. (24 Tem 2026, kullanıcı fikri — Tlosx'un HTF confluence
mantığından esinlenip, tf_alignment_early_divergence'ın (17-20 Tem, bkz.
memory) düştüğü look-ahead tuzağına düşmeden test ediyoruz.)

KRİTİK disiplin (tf_alignment'ın hatasından ders):
  - Sinyal SADECE gerçekten kapanmış barlarla üretilir (repaint yok, Tlosx'un
    lookahead_on'unun tersi) — 45m/90m barları 15m'in kapanmış barlarından
    KENDİMİZ inşa ediyoruz, ayrı bir HTF sorgusu/forming-bar riski yok.
  - Kohort-içi sıralama (ayrışma) SADECE gözlem penceresi [0,K]'de yapılır;
    PnL SADECE ticaret penceresi [K, K+M]'de, K anındaki fiyattan ölçülür.
    Sinyal anına (t=0) geriye kredi YOK — tf_alignment'ın PF 0.605 ile
    çökmesinin kök nedeni tam bu ayrımın yapılmamasıydı.

Sinyal: 15m kapanışı AYNI ZAMANDA 45m ve 90m kapanışına denk gelen anlarda
(günde 16 kez, UTC 00:00'dan itibaren her 90dk), üç TF'nin de Heikin Ashi
rengi aynı yönde ise (üçü de bull veya üçü de bear) sinyal.

Kullanım: python -m research.pattern_lab.nested_tf_confluence_divergence_bt
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from research.pattern_lab.do_open_streak_full_clean_bt import _conn

_LOOKBACK_DAYS = 60
_MIN_BARS = 700  # ~60 gün 15m
_MAX_SYMBOLS = 250
_MIN_COHORT = 6
_HORIZON_BARS = 40  # 15m*40 = 10 saat ileri takip
_PLACEBO_ITER = 300


def _fetch_symbols(cur) -> list[str]:
    cur.execute(
        """
        SELECT symbol, count(*) c FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '%s days'
        GROUP BY symbol HAVING count(*) >= %s
        ORDER BY c DESC LIMIT %s
        """,
        (_LOOKBACK_DAYS, _MIN_BARS, _MAX_SYMBOLS),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_15m(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT bucket, open, high, low, close FROM cagg_15m
        WHERE symbol=%s AND bucket > NOW() - INTERVAL '%s days'
        ORDER BY bucket ASC
        """,
        (symbol, _LOOKBACK_DAYS),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def _heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, ha_c


def _truncate_to_last_gapfree_run(df: pd.DataFrame) -> pd.DataFrame:
    """15m barlar arasında >20dk boşluk varsa, son (en güncel) boşluksuz
    segmentle sınırla — recursive HA formülü boşluk üzerinden sessizce
    bozulmasın diye (bkz. indicators/core.py::truncate_after_gap ile aynı
    disiplin)."""
    b = df["bucket"].to_numpy()
    if len(b) < 2:
        return df
    gaps = np.diff(b).astype("timedelta64[s]").astype(float) / 60.0
    bad = np.where(gaps > 20)[0]
    if len(bad) == 0:
        return df
    start = bad[-1] + 1
    return df.iloc[start:].reset_index(drop=True)


def _htf_bull_by_close_min(df: pd.DataFrame, period_min: int, bars_per_period: int) -> pd.Series:
    """15m barlardan gerçek-saat-hizalı (UTC epoch'a göre) period_min'lik HTF
    barları inşa eder — SADECE tam (bars_per_period sayıda, boşluksuz) grupları
    tutar. Döner: index=grup kapanış dakikası (close_min), değer=HA bull(bool)."""
    # close_min TAM period sınırındaysa önceki gruba ait (kapanış anı o grubu
    # tamamlıyor) — düz "//" bunu yanlış gruplardı (sınır barını bir SONRAKİ
    # gruba atıyordu), "-1" kaydırması doğru gruplamayı sağlıyor.
    g = (df["close_min"] - 1) // period_min
    agg = df.groupby(g).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), n=("close", "count"), close_min_last=("close_min", "last"),
    )
    agg = agg[(agg["n"] == bars_per_period) & (agg["close_min_last"] == (agg.index + 1) * period_min)]
    if agg.empty:
        return pd.Series(dtype=bool)
    ha_o, ha_c = _heikin_ashi(
        agg["open"].to_numpy(), agg["high"].to_numpy(), agg["low"].to_numpy(), agg["close"].to_numpy()
    )
    return pd.Series(ha_c > ha_o, index=(agg.index + 1) * period_min)  # index = close_min


def _build_signals(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df = _truncate_to_last_gapfree_run(df)
    if len(df) < _MIN_BARS // 2:
        return pd.DataFrame()

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    bucket = df["bucket"].to_numpy()
    n = len(c)

    ha15_o, ha15_c = _heikin_ashi(o, h, l, c)
    bull15 = ha15_c > ha15_o

    open_min = bucket.astype("datetime64[m]").astype("int64")
    close_min = open_min + 15
    df = df.copy()
    df["close_min"] = close_min

    bull45_by_cm = _htf_bull_by_close_min(df, 45, 3)
    bull90_by_cm = _htf_bull_by_close_min(df, 90, 6)

    records = []
    for i in range(n):
        cm = close_min[i]
        if cm % 90 != 0:
            continue
        if cm not in bull45_by_cm.index or cm not in bull90_by_cm.index:
            continue
        if i + 1 >= n:
            continue

        b15, b45, b90 = bull15[i], bull45_by_cm.loc[cm], bull90_by_cm.loc[cm]
        if b15 == b45 == b90:
            direction = "Long" if b15 else "Short"
            records.append(
                {
                    "symbol": symbol,
                    "signal_time": bucket[i],
                    "signal_idx": i,
                    "direction": direction,
                    "entry_price": c[i],
                    "close_series": c[i : min(i + _HORIZON_BARS + 1, n)],
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    conn = _conn()
    cur = conn.cursor()
    symbols = _fetch_symbols(cur)
    print(f"analiz edilecek sembol: {len(symbols)}")

    all_sig = []
    for si, sym in enumerate(symbols):
        df = _fetch_15m(cur, sym)
        if len(df) < _MIN_BARS:
            continue
        sig = _build_signals(sym, df)
        if not sig.empty:
            all_sig.append(sig)
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    sig_df = pd.concat(all_sig, ignore_index=True)
    print(f"\ntoplam üçlü-kesişim sinyali: {len(sig_df)} "
          f"(Long={ (sig_df['direction']=='Long').sum() }, Short={ (sig_df['direction']=='Short').sum() })")

    # kohort: aynı signal_time + aynı direction
    sig_df["cohort_key"] = sig_df["signal_time"].astype(str) + "_" + sig_df["direction"]
    cohort_sizes = sig_df.groupby("cohort_key").size()
    big_cohorts = cohort_sizes[cohort_sizes >= _MIN_COHORT].index
    sig_df = sig_df[sig_df["cohort_key"].isin(big_cohorts)].reset_index(drop=True)
    print(f"n>={_MIN_COHORT} kohortlarındaki sinyal sayısı: {len(sig_df)} "
          f"({sig_df['cohort_key'].nunique()} kohort)")

    if sig_df.empty:
        print("Yeterli kohort yok.")
        return

    def fwd_ret(row, bar_offset):
        cs = row["close_series"]
        side = 1.0 if row["direction"] == "Long" else -1.0
        if bar_offset >= len(cs):
            return np.nan
        return (cs[bar_offset] - row["entry_price"]) / row["entry_price"] * 100.0 * side

    print("\n" + "=" * 78)
    print("Baseline: kohortun TAMAMINI sinyal anında al, ufuk sonuna kadar tut")
    print("=" * 78)
    for horizon in [8, 16, 24, 40]:
        rets = sig_df.apply(lambda r: fwd_ret(r, horizon), axis=1).dropna()
        wr = (rets > 0).mean() * 100
        print(f"  ufuk={horizon*15}dk: n={len(rets)} WR=%{wr:.1f} ort=%{rets.mean():.3f} medyan=%{rets.median():.3f}")

    print("\n" + "=" * 78)
    print("Kohort-içi ayrışma: K barda sırala, top-tercili K->K+M penceresinde ticarete sok")
    print("(PnL K anındaki fiyattan ölçülüyor, sinyal anından DEĞİL — look-ahead yok)")
    print("=" * 78)
    for K, M in [(4, 8), (4, 16), (8, 16), (8, 24)]:
        sig_df["rank_ret"] = sig_df.apply(lambda r: fwd_ret(r, K), axis=1)
        sig_df["trade_ret"] = sig_df.apply(
            lambda r: (
                (r["close_series"][min(K + M, len(r["close_series"]) - 1)] - r["close_series"][K]) / r["close_series"][K]
                * 100.0 * (1.0 if r["direction"] == "Long" else -1.0)
                if K < len(r["close_series"]) else np.nan
            ),
            axis=1,
        )
        valid = sig_df.dropna(subset=["rank_ret", "trade_ret"])
        if valid.empty:
            continue

        def _top_tercile(g):
            if len(g) < _MIN_COHORT:
                return g.iloc[0:0]
            th = g["rank_ret"].quantile(2 / 3)
            return g[g["rank_ret"] >= th]

        top = valid.groupby("cohort_key", group_keys=False).apply(_top_tercile)
        base_ret = valid["trade_ret"]
        top_ret = top["trade_ret"]

        rng = np.random.default_rng(42)
        placebo_means = []
        for _ in range(_PLACEBO_ITER):
            samp = valid.groupby("cohort_key", group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), max(1, len(g) // 3)), random_state=rng.integers(0, 1_000_000))
            )
            placebo_means.append(samp["trade_ret"].mean())
        placebo_means = np.array(placebo_means)
        pct_ge = (placebo_means >= top_ret.mean()).mean() * 100

        print(f"\n  K={K*15}dk gözlem, M={M*15}dk ticaret:")
        print(f"    kohort tamamı  (K->K+M): n={len(base_ret)} WR=%{(base_ret>0).mean()*100:.1f} ort=%{base_ret.mean():.3f}")
        print(f"    top-tercil ayrışan       : n={len(top_ret)} WR=%{(top_ret>0).mean()*100:.1f} ort=%{top_ret.mean():.3f}")
        print(f"    placebo (rastgele 1/3 alt-küme bu ort'u eşitleme/geçme sıklığı): %{pct_ge:.1f}")

        if len(top_ret) >= 40:
            t_sorted = top.sort_values("signal_time")
            mid = t_sorted["signal_time"].iloc[len(t_sorted) // 2]
            fh = t_sorted[t_sorted["signal_time"] < mid]["trade_ret"]
            sh = t_sorted[t_sorted["signal_time"] >= mid]["trade_ret"]
            print(f"    split-period: ilk yarı n={len(fh)} ort=%{fh.mean():.3f} | ikinci yarı n={len(sh)} ort=%{sh.mean():.3f}")


if __name__ == "__main__":
    main()
