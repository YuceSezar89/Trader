"""
Supertrend giriş-mumu outlier testi (hocanın "Supertrend STD SIGNAL" Pine
script'inden — /Users/yusuf/Downloads/Supertrend STD SIGNAL.txt).

Script her ST sinyalinde giriş mumunun Width%/Body%/RSIchg/Vol% değerlerini
kaydedip GEÇMİŞ sinyallere göre z-score alıyor, "giriş sapması" (herhangi biri
|z|>eşik) ile "getiri sapması" (|z_getiri|>eşik) arasında ilişki arıyor. Bu,
hoca doktrini madde 4'ün (manipülasyon mu gerçek mi — Δmomentum+Δhacim eşliği)
somut bir versiyonu — 6 Tem'de doğrulanan confluence/divergence bulgusuyla
aynı aile ama ham mum geometrisi (Width/Body) ayrı boyut olarak ekliyor.

Look-ahead güvenliği: her sinyalin z-score'u SADECE kendisinden ÖNCEKİ aynı
sembol+interval sinyallerinin istatistiğiyle hesaplanıyor (expanding window,
Pine'ın kendi array-tabanlı yaklaşımıyla birebir aynı mantık).

Kullanım: python -m research.pattern_lab.st_entry_outlier_bt
"""

import sys
import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_rsi, truncate_after_gap
from research.pattern_lab.vol_exhaustion_bt import _stats

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m"}
_BARS_NEEDED = 220
_MIN_HISTORY = 10  # z-score güvenilir olsun diye bu kadar önceki sinyal şart
_Z_THRESH = 2.0


def _fetch_signals(cur, interval: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, signal_type, opened_at, realized_pnl
        FROM signals
        WHERE indicators = 'Supertrend(10,3.0)' AND interval = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY symbol, opened_at
        """,
        (interval,),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "symbol", "signal_type", "opened_at", "realized_pnl"])


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    cagg = _CAGG[interval]
    cur.execute(
        f"SELECT bucket AS open_time, open, high, low, close, volume "
        f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
        (symbol, opened_at, _BARS_NEEDED),
    )
    rows = cur.fetchall()
    if not rows or len(rows) < 40:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)
    # Bugünkü gap-güvenilirlik denetiminde bulunan sorun: 220 bar geriye giderken
    # aylar süren boşlukları atlayıp "ardışık" gibi hesaplıyorduk (RSI/hacim-SMA
    # anlamsızlaşıyordu). En son boşluktan SONRAKİ barlarla sınırla.
    df = truncate_after_gap(df)
    if len(df) < 40:
        return None
    return df


def _entry_metrics(df: pd.DataFrame) -> dict | None:
    """Pine script'in Width%/Body%/RSIchg/Vol% formülleriyle BİREBİR aynı."""
    try:
        last = df.iloc[-1]
        width_pct = float((last["high"] - last["low"]) / last["low"] * 100) if last["low"] > 0 else 0.0
        body_pct = float((last["close"] - last["open"]) / last["open"] * 100) if last["open"] > 0 else 0.0

        df2 = df.copy()
        df2["log_close"] = np.log(df2["close"])
        rsi = calculate_rsi(df2, period=14, price_col="log_close")
        rsi_chg = float(rsi.diff().iloc[-1])

        vol_sma20 = df["volume"].rolling(20).mean().iloc[-1]
        vol_pct = float((last["volume"] - vol_sma20) / vol_sma20 * 100) if vol_sma20 > 0 else 0.0
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if any(pd.isna(v) for v in (width_pct, body_pct, rsi_chg, vol_pct)):
        return None
    return {"width": width_pct, "body": body_pct, "rsi_chg": rsi_chg, "vol": vol_pct}


def _collect(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, interval)
    print(f"[{interval}] {len(sig_df):,} kapalı Supertrend(10,3.0) sinyali bulundu")

    records = []
    skipped = 0
    for i, row in enumerate(sig_df.itertuples(), 1):
        df = _fetch_bars(cur, row.symbol, interval, row.opened_at)
        if df is None:
            skipped += 1
            continue
        m = _entry_metrics(df)
        if m is None:
            skipped += 1
            continue
        records.append(
            {
                "symbol": row.symbol,
                "signal_type": row.signal_type,
                "opened_at": row.opened_at,
                "pnl": float(row.realized_pnl),
                **m,
            }
        )
        if i % 500 == 0:
            print(f"  ... {i}/{len(sig_df)} işlendi")
    conn.close()
    print(f"[{interval}] tamamlandı — {len(records)} işlendi, {skipped} atlandı\n")
    return pd.DataFrame(records)


def _zscore_expanding(group: pd.DataFrame, col: str) -> pd.Series:
    """Her satır için SADECE önceki satırların mean/std'ine göre z-score (look-ahead güvenli)."""
    vals = group[col].to_numpy()
    z = np.full(len(vals), np.nan)
    for i in range(_MIN_HISTORY, len(vals)):
        hist = vals[:i]
        sd = hist.std()
        if sd > 0:
            z[i] = (vals[i] - hist.mean()) / sd
    return pd.Series(z, index=group.index)


def _add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    # Sembol başına ST kesişimi çok nadir (grup başına birkaç sinyal) — z-score'u
    # sembol bazında değil, YÖN bazında TÜM semboller zaman sırasıyla havuzlanarak
    # hesaplıyoruz (yine sadece geçmiş sinyaller kullanılıyor, look-ahead yok).
    df = df.sort_values(["signal_type", "opened_at"]).reset_index(drop=True)
    for col in ("width", "body", "rsi_chg", "vol", "pnl"):
        df[f"z_{col}"] = df.groupby("signal_type", group_keys=False).apply(
            lambda g, c=col: _zscore_expanding(g, c), include_groups=False
        )
    df = df.dropna(subset=[f"z_{c}" for c in ("width", "body", "rsi_chg", "vol", "pnl")])
    df["entry_outlier"] = (
        (df["z_width"].abs() > _Z_THRESH)
        | (df["z_body"].abs() > _Z_THRESH)
        | (df["z_rsi_chg"].abs() > _Z_THRESH)
        | (df["z_vol"].abs() > _Z_THRESH)
    )
    df["return_outlier"] = df["z_pnl"].abs() > _Z_THRESH
    return df


def _report_group(name: str, sub: pd.DataFrame) -> None:
    s = _stats(sub["pnl"].to_numpy() / 100)
    print(f"  {name:22} n={s.get('n',0):>5}  WR%={s.get('wr',0):>6}  ort%={s.get('ort_%',0):>7}  PF={s.get('pf',0):>6}")


def _placebo(df: pd.DataFrame, real_pf_gap: float, n_iter: int = 200) -> float:
    """entry_outlier bayrağını rastgele karıştırıp (aynı True sayısı) PF farkının
    gerçek farktan büyük çıktığı deneme oranını döndürür — düşükse bulgu şansa bağlı değil."""
    rng = np.random.default_rng(42)
    n_true = int(df["entry_outlier"].sum())
    pnl = df["pnl"].to_numpy() / 100
    count_ge = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(df))
        fake_outlier = np.zeros(len(df), dtype=bool)
        fake_outlier[perm[:n_true]] = True
        pf_true = _stats(pnl[fake_outlier]).get("pf", 0) or 0
        pf_false = _stats(pnl[~fake_outlier]).get("pf", 0) or 0
        gap = pf_true - pf_false
        if abs(gap) >= abs(real_pf_gap):
            count_ge += 1
    return count_ge / n_iter


def _analyze(df: pd.DataFrame, interval: str) -> None:
    for direction in ("Long", "Short"):
        sub = df[df["signal_type"] == direction]
        print(f"\n{'='*72}\nSupertrend(10,3.0) {direction} — {interval}  (n={len(sub):,})\n{'='*72}")
        if len(sub) < 60:
            print("Örneklem çok küçük, atlanıyor.")
            continue

        outlier = sub[sub["entry_outlier"]]
        normal = sub[~sub["entry_outlier"]]
        print(f"  Giriş sapması olan: {len(outlier)}  ({len(outlier)/len(sub)*100:.1f}%)")
        _report_group("baseline (tümü)", sub)
        _report_group("entry_outlier=True", outlier)
        _report_group("entry_outlier=False", normal)

        pf_out = _stats(outlier["pnl"].to_numpy() / 100).get("pf", 0) or 0
        pf_norm = _stats(normal["pnl"].to_numpy() / 100).get("pf", 0) or 0
        real_gap = pf_out - pf_norm

        # Sapma-birlikte-görülme: giriş sapması VARKEN getiri sapması var mı yok mu
        if len(outlier) >= 20:
            co = outlier["return_outlier"].mean() * 100
            base_co = sub["return_outlier"].mean() * 100
            print(f"  Getiri sapması oranı: entry_outlier=True içinde %{co:.1f}  (genel %{base_co:.1f})")

        # split-period (entry_outlier=True grubunun kendi içinde)
        if len(outlier) >= 40:
            t_min, t_max = outlier["opened_at"].min(), outlier["opened_at"].max()
            mid = t_min + (t_max - t_min) / 2
            first = outlier[outlier["opened_at"] < mid]
            second = outlier[outlier["opened_at"] >= mid]
            print("  -- split-period (entry_outlier=True) --")
            print(f"    ilk yarı:    PF={_stats(first['pnl'].to_numpy()/100).get('pf',0):>6} (n={len(first)})")
            print(f"    ikinci yarı: PF={_stats(second['pnl'].to_numpy()/100).get('pf',0):>6} (n={len(second)})")

        if len(sub) >= 60:
            p = _placebo(sub, real_gap)
            print(f"  Placebo: gerçek PF farkı ({real_gap:+.3f}) kadar/daha büyük fark rastgelede %{p*100:.1f} sıklıkta çıktı")


def run() -> None:
    for interval in ("5m", "15m"):
        raw = _collect(interval)
        if raw.empty:
            continue
        df = _add_zscores(raw)
        _analyze(df, interval)


if __name__ == "__main__":
    run()
