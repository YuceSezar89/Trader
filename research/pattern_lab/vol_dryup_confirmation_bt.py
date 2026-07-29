"""
Hacim kuruması → momentum patlaması → hacim teyidi testi (19 Tem 2026,
Hoca doktrini: "büyük hareketlerin başlaması için hacmi sıfırlarlar" +
kullanıcının 5-sembollük Pine indikatörü + [[project_turtle_traders]]
madde 3 "flow_since_prev" notu).

İkili bayrak (Madde 4'ün — manipülasyon/gerçek testinin — başarılı deseniyle
aynı çerçeve, tek bir kompozit skor DEĞİL):
  kuruma_var  = sinyalden ÖNCEKİ 5 bar'da V bileşeni düşük (<30)
  teyit_geldi = sinyalden SONRAKİ 4 bar'da V bileşeni nötr/üstüne çıktı (>=50)
  dryup_confirmed = kuruma_var AND teyit_geldi

Sadece RSI_Cross(9,24) Long (kullanıcı kararı — Short test edilmedi).
Gerçek yönlü hacim (buy_volume) gerektiği için 2026-06-22 sonrası sinyallerle
sınırlı (öncesi kapsam-delikli, bkz. [[project_directional_volume]] /
[[project_cvd_divergence]] "buy_volume kapsamı 22 Haziran'a kadar delikliydi").

Kullanım: python -m research.pattern_lab.vol_dryup_confirmation_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import truncate_after_gap
from research.pattern_lab.vol_exhaustion_bt import _stats
from utils.preprocessing import normalize_volume_0_100
from utils.vpmv import directional_volume

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_PRE_BARS = 5
_POST_BARS = 4
_LOOKBACK = 220
_DRYUP_MAX = 20.0
_CONFIRM_MIN = 65.0
_MIN_COVERAGE_DATE = "2026-06-22"


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT s.id, s.symbol, s.interval, s.opened_at, s.realized_pnl
        FROM signals s
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.signal_type = 'Long'
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.opened_at >= %s
        ORDER BY s.symbol, s.opened_at
        """,
        (_MIN_COVERAGE_DATE,),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "symbol", "interval", "opened_at", "realized_pnl"])


def _fetch_real_trades(cur) -> pd.DataFrame:
    cur.execute(
        "SELECT signal_id, pnl_usd FROM paper_trades "
        "WHERE strategy = 'rsi_cross_live' AND status = 'closed' AND signal_id IS NOT NULL"
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["signal_id", "pnl_usd"])


def _fetch_pre_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS bucket, open, high, low, close, volume, buy_volume, sell_volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
            "ORDER BY timestamp DESC LIMIT %s",
            (symbol, opened_at, _LOOKBACK),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket, open, high, low, close, volume, buy_volume, sell_volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
            (symbol, opened_at, _LOOKBACK),
        )
    rows = cur.fetchall()
    if not rows or len(rows) < 60:
        return None
    cols = ["bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    return pd.DataFrame(rows, columns=cols).iloc[::-1].reset_index(drop=True)


def _fetch_post_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS bucket, open, high, low, close, volume, buy_volume, sell_volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp > %s "
            "ORDER BY timestamp ASC LIMIT %s",
            (symbol, opened_at, _POST_BARS),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket, open, high, low, close, volume, buy_volume, sell_volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket > %s ORDER BY bucket ASC LIMIT %s",
            (symbol, opened_at, _POST_BARS),
        )
    rows = cur.fetchall()
    if not rows or len(rows) < _POST_BARS:
        return None
    cols = ["bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    return pd.DataFrame(rows, columns=cols)


def _compute_pre_post(cur, symbol: str, interval: str, opened_at) -> dict | None:
    df_pre = _fetch_pre_bars(cur, symbol, interval, opened_at)
    if df_pre is None:
        return None
    df_pre = truncate_after_gap(df_pre)
    if len(df_pre) < _PRE_BARS + 10:
        return None
    df_post = _fetch_post_bars(cur, symbol, interval, opened_at)
    if df_post is None:
        return None

    df_full = pd.concat([df_pre, df_post], ignore_index=True)
    if df_full["buy_volume"].isna().all():
        return None

    vol_series = normalize_volume_0_100(directional_volume(df_full, side=1.0))

    signal_idx = len(df_pre) - 1
    pre_slice = vol_series.iloc[signal_idx - _PRE_BARS : signal_idx]
    post_slice = vol_series.iloc[signal_idx + 1 : signal_idx + 1 + _POST_BARS]
    if len(pre_slice) < _PRE_BARS or len(post_slice) < _POST_BARS:
        return None

    return {"vol_pre": float(pre_slice.mean()), "vol_post": float(post_slice.mean())}


def _placebo(pnl: np.ndarray, flag: np.ndarray, real_gap: float, n_iter: int = 500) -> float:
    rng = np.random.default_rng(42)
    n_true = int(flag.sum())
    count_ge = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(pnl))
        fake = np.zeros(len(pnl), dtype=bool)
        fake[perm[:n_true]] = True
        pf_true = _stats(pnl[fake]).get("pf", 0) or 0
        pf_false = _stats(pnl[~fake]).get("pf", 0) or 0
        if abs(pf_true - pf_false) >= abs(real_gap):
            count_ge += 1
    return count_ge / n_iter


def _report_group(name: str, sub: pd.DataFrame, pnl_col: str, scale: float) -> None:
    s = _stats(sub[pnl_col].to_numpy() / scale)
    print(f"  {name:28} n={s.get('n',0):>5}  WR%={s.get('wr',0):>6}  ort={s.get('ort_%',0):>8}  PF={s.get('pf',0):>6}")


_CACHE_PATH = "/tmp/vol_dryup_confirmation_cache.parquet"


def main() -> None:
    import os

    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()

    if os.path.exists(_CACHE_PATH):
        df = pd.read_parquet(_CACHE_PATH)
        print(f"[cache] {len(df)} kayıt {_CACHE_PATH}'den okundu (DB sorgusu atlandı)")
    else:
        sig_df = _fetch_signals(cur)
        print(f"[fetch] {len(sig_df)} RSI_Cross(9,24) Long kapalı sinyal ({_MIN_COVERAGE_DATE} sonrası)")

        records = []
        skipped = 0
        for i, row in enumerate(sig_df.itertuples(), 1):
            out = _compute_pre_post(cur, row.symbol, row.interval, row.opened_at)
            if out is None:
                skipped += 1
                continue
            records.append(
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "opened_at": row.opened_at,
                    "pnl": float(row.realized_pnl),
                    **out,
                }
            )
            if i % 500 == 0:
                print(f"  ... {i}/{len(sig_df)} işlendi")

        df = pd.DataFrame(records)
        print(f"[collect] {len(df)} işlendi, {skipped} atlandı (yetersiz bar/buy_volume)")
        df.to_parquet(_CACHE_PATH)
        print(f"[cache] {_CACHE_PATH}'e yazıldı\n")

    real_trades = _fetch_real_trades(cur)
    conn.close()

    if len(df) < 60:
        print("Örneklem çok küçük, durduruluyor.")
        return

    df["kuruma_var"] = df["vol_pre"] < _DRYUP_MAX
    df["teyit_geldi"] = df["vol_post"] >= _CONFIRM_MIN
    df["dryup_confirmed"] = df["kuruma_var"] & df["teyit_geldi"]

    print(f"kuruma_var (vol_pre<{_DRYUP_MAX}): {df['kuruma_var'].sum()} (%{df['kuruma_var'].mean()*100:.1f})")
    print(f"teyit_geldi (vol_post>={_CONFIRM_MIN}): {df['teyit_geldi'].sum()} (%{df['teyit_geldi'].mean()*100:.1f})")
    print(f"dryup_confirmed: {df['dryup_confirmed'].sum()} (%{df['dryup_confirmed'].mean()*100:.1f})\n")

    print("=== 1) HAM realized_pnl ile karşılaştırma ===")
    _report_group("baseline (tümü)", df, "pnl", 100.0)
    grp = df[df["dryup_confirmed"]]
    rest = df[~df["dryup_confirmed"]]
    if len(grp) < 20:
        print("  dryup_confirmed örneklemi çok küçük, kalan analiz atlanıyor.")
        return
    _report_group("dryup_confirmed=True", grp, "pnl", 100.0)
    _report_group("dryup_confirmed=False", rest, "pnl", 100.0)

    pf_grp = _stats(grp["pnl"].to_numpy() / 100).get("pf", 0) or 0
    pf_rest = _stats(rest["pnl"].to_numpy() / 100).get("pf", 0) or 0
    real_gap = pf_grp - pf_rest
    print(f"\nGerçek PF farkı: {real_gap:+.3f}")

    print("\n=== 2) split-period (dryup_confirmed grubu, kronolojik ikiye böl) ===")
    t_min, t_max = grp["opened_at"].min(), grp["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    first = grp[grp["opened_at"] < mid]
    second = grp[grp["opened_at"] >= mid]
    _report_group("ilk yarı", first, "pnl", 100.0)
    _report_group("ikinci yarı", second, "pnl", 100.0)

    print("\n=== 3) placebo (rastgele aynı boyutta grup, 500 iterasyon) ===")
    p = _placebo(df["pnl"].to_numpy() / 100, df["dryup_confirmed"].to_numpy(), real_gap)
    print(f"gerçek PF farkı ({real_gap:+.3f}) rastgelede %{p*100:.1f} sıklıkta çıktı")

    print("\n=== 4) GERÇEK $ doğrulaması (paper_trades.pnl_usd, tf_alignment_live) ===")
    merged = df.merge(real_trades, left_on="id", right_on="signal_id", how="inner")
    if len(merged) < 15:
        print(f"  yetersiz örnek (n={len(merged)}) — henüz canlı işlem birikmedi")
    else:
        m_grp = merged[merged["dryup_confirmed"]]
        m_rest = merged[~merged["dryup_confirmed"]]
        _report_group("dryup_confirmed=True ($)", m_grp, "pnl_usd", 1.0)
        _report_group("dryup_confirmed=False ($)", m_rest, "pnl_usd", 1.0)


if __name__ == "__main__":
    main()
