"""
Ardışık sinyal VPMV artışı testi (hocanın Telegram ekran görüntüsünden ilham —
Devisso'nun kutucuklarındaki Hacim%/Momentum%/Volatilite%, kaldıraç 1x-4x ile
birlikte gösteriliyordu). Kullanıcının hipotezi: "hepsinin arttığı sinyaller
muhtemelen daha kaliteli olacaktır, ama hacim ön planda" — yani aynı sembol +
aynı yön + aynı gösterge ailesinde bir sinyalin VPMV bileşenleri (Hacim/
Momentum/Fiyat/Volatilite, ranking_worker.py'deki KANONİK ağırlıklarla:
%35/%35/%20/%10) bir ÖNCEKİ sinyale göre artıyorsa sinyal daha kaliteli mi?

Bu, mtf_score (TF yön uyumu) veya devisso_score/ERSI (verimlilik oranı)
DEĞİL — ham bileşen skorlarının ardışık sinyaller arası DEĞİŞİMİ.

Look-ahead güvenliği: her sinyalin delta'sı SADECE kendisinden önceki aynı
grup sinyaliyle karşılaştırılıyor (tek adım geriye, expanding değil — hocanın
"2 kutu arasında" dediği birebir karşılaştırma).

Kullanım: python -m research.pattern_lab.consecutive_vpmv_increase_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_atr, calculate_rsi, truncate_after_gap
from research.pattern_lab.vol_exhaustion_bt import _stats
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)

INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross", "Supertrend(10,3.0)"]
DIRECTIONS = ["Long", "Short"]
_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_BARS_NEEDED = 220
_GAP_HOURS_THRESHOLD = 200  # bu esikten buyuk ic-gap iceren semboller test disi


def _bad_symbols(cur) -> set[str]:
    """13 Tem 2026 gece/sabah tespit edilen ~8-9 aylik tarihsel bosluk henuz
    tum sembollerde doldurulmadi (bkz. memory: project_historical_backfill_pilot).
    Hala buyuk ic-gap tasiyan sembolleri teste dahil etme — truncate_after_gap
    zaten cogunu temizler ama kenar etkisi riskini tamamen kapatmak icin en
    bastan disla."""
    cur.execute(
        """
        WITH gaps AS (
            SELECT symbol, EXTRACT(EPOCH FROM (curr_ts-prev_ts))/3600 AS saat
            FROM (
                SELECT symbol, timestamp AS curr_ts,
                       LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                FROM price_data WHERE interval='1m'
            ) t
            WHERE prev_ts IS NOT NULL
        )
        SELECT DISTINCT symbol FROM gaps WHERE saat > %s
        """,
        (_GAP_HOURS_THRESHOLD,),
    )
    return {r[0] for r in cur.fetchall()}


def _fetch_signals(cur, indicator: str, direction: str, exclude_symbols: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, opened_at, realized_pnl
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY symbol, opened_at
        """,
        (indicator, direction),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id", "symbol", "interval", "opened_at", "realized_pnl"])
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)].reset_index(drop=True)
    return df


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS open_time, open, high, low, close, volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
            "ORDER BY timestamp DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket AS open_time, open, high, low, close, volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
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


def _vpmv_components(df: pd.DataFrame, direction: str) -> dict | None:
    """ranking_worker.py::_vpmv ile BİREBİR aynı formül — sadece bileşik skor
    yerine 4 bileşeni ayrı ayrı döner. mom/price YÖNE HİZALANIR (test_vpmv_
    divergence.py'deki momentum_aligned ile aynı disiplin) — Short sinyalde
    fiyat/momentumun YUKARI gitmesi kötü bir işarettir, "arttı" burada hep
    "sinyal yönü lehine güçlendi" anlamına gelmeli. vol/volat yönsüzdür,
    hizalamaya gerek yok."""
    try:
        rsi_series = calculate_rsi(df, period=14)
        rsi_centered = rsi_series - 50
        atr_series = calculate_atr(df, period=14)
        price_pct = df["close"].pct_change().fillna(0.0) * 100.0

        vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])
        mom_score = float(normalize_momentum_0_100(rsi_centered).iloc[-1])
        volat_score = float(normalize_volatility_0_100(atr_series).iloc[-1])
        price_score = float(normalize_price_0_100(price_pct).iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    vals = (vol_score, mom_score, volat_score, price_score)
    if any(pd.isna(v) for v in vals):
        return None
    if direction == "Short":
        mom_score = 100.0 - mom_score
        price_score = 100.0 - price_score
    return {"vol": vol_score, "mom": mom_score, "volat": volat_score, "price": price_score}


def _collect(indicator: str, direction: str, exclude_symbols: set[str]) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    sig_df = _fetch_signals(cur, indicator, direction, exclude_symbols)
    if sig_df.empty:
        conn.close()
        return sig_df

    records = []
    for row in sig_df.itertuples():
        df = _fetch_bars(cur, row.symbol, row.interval, row.opened_at)
        if df is None:
            continue
        comp = _vpmv_components(df, direction)
        if comp is None:
            continue
        records.append(
            {
                "symbol": row.symbol,
                "opened_at": row.opened_at,
                "pnl": float(row.realized_pnl),
                **comp,
            }
        )
    conn.close()
    return pd.DataFrame(records)


def _add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Her satırın delta'sını SADECE aynı sembolün bir önceki sinyaliyle
    karşılaştırır (tek adım geriye, look-ahead yok)."""
    df = df.sort_values(["symbol", "opened_at"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        df[f"d_{col}"] = df.groupby("symbol")[col].diff()
    df = df.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    df["vol_up"] = df["d_vol"] > 0
    df["all_up"] = (df["d_vol"] > 0) & (df["d_mom"] > 0) & (df["d_volat"] > 0) & (df["d_price"] > 0)
    return df


def _report_group(name: str, sub: pd.DataFrame) -> None:
    s = _stats(sub["pnl"].to_numpy() / 100)
    print(
        f"  {name:22} n={s.get('n',0):>5}  WR%={s.get('wr',0):>6}  "
        f"ort%={s.get('ort_%',0):>7}  PF={s.get('pf',0):>6}"
    )


def _placebo(df: pd.DataFrame, flag_col: str, real_gap: float, n_iter: int = 200) -> float:
    rng = np.random.default_rng(42)
    n_true = int(df[flag_col].sum())
    pnl = df["pnl"].to_numpy() / 100
    count_ge = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(df))
        fake = np.zeros(len(df), dtype=bool)
        fake[perm[:n_true]] = True
        pf_true = _stats(pnl[fake]).get("pf", 0) or 0
        pf_false = _stats(pnl[~fake]).get("pf", 0) or 0
        if abs(pf_true - pf_false) >= abs(real_gap):
            count_ge += 1
    return count_ge / n_iter


def _analyze(df: pd.DataFrame, indicator: str, direction: str) -> None:
    print(f"\n{'='*72}\n{indicator} — {direction}  (n={len(df):,})\n{'='*72}")
    if len(df) < 60:
        print("Örneklem çok küçük, atlanıyor.")
        return

    _report_group("baseline (tümü)", df)

    for flag_col, label in (("vol_up", "vol_up=True (hacim arttı)"), ("all_up", "all_up=True (hepsi arttı)")):
        grp = df[df[flag_col]]
        rest = df[~df[flag_col]]
        print(f"\n  -- {label} (n={len(grp)}, %{len(grp)/len(df)*100:.1f}) --")
        if len(grp) < 20:
            print("    örneklem çok küçük.")
            continue
        _report_group(f"{flag_col}=True", grp)
        _report_group(f"{flag_col}=False", rest)

        pf_grp = _stats(grp["pnl"].to_numpy() / 100).get("pf", 0) or 0
        pf_rest = _stats(rest["pnl"].to_numpy() / 100).get("pf", 0) or 0
        real_gap = pf_grp - pf_rest

        if len(grp) >= 40:
            t_min, t_max = grp["opened_at"].min(), grp["opened_at"].max()
            mid = t_min + (t_max - t_min) / 2
            first = grp[grp["opened_at"] < mid]
            second = grp[grp["opened_at"] >= mid]
            print(
                f"    split-period: ilk yarı PF={_stats(first['pnl'].to_numpy()/100).get('pf',0):>6} "
                f"(n={len(first)})  ikinci yarı PF={_stats(second['pnl'].to_numpy()/100).get('pf',0):>6} "
                f"(n={len(second)})"
            )

        p = _placebo(df, flag_col, real_gap)
        print(f"    placebo: gerçek PF farkı ({real_gap:+.3f}) rastgelede %{p*100:.1f} sıklıkta çıktı")


def run() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    bad_symbols = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad_symbols)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor\n")

    for indicator in INDICATORS:
        for direction in DIRECTIONS:
            raw = _collect(indicator, direction, bad_symbols)
            if raw.empty:
                print(f"\n{indicator} — {direction}: veri yok.")
                continue
            df = _add_deltas(raw)
            _analyze(df, indicator, direction)


if __name__ == "__main__":
    run()
