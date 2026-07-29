"""
Madde-4 (Confluence/Divergence) — bugünkü TEMİZ yöntemle yeniden doğrulama
+ reversal-gürültüsü filtresi olarak test (21 Tem 2026).

Madde-4 ORİJİNAL olarak (test_vpmv_divergence.py, 6 Tem) `signals.realized_pnl`
(kirli hedef) ile doğrulanmıştı — bugünkü standart (sabit N-bar ileri getiri,
çıkış mekanizmasından bağımsız) ile TEKRAR test ediyoruz. Bu hem Madde-4'ün
kendisini daha sıkı bir testten geçiriyor hem de "reversal-gürültüsünü
Madde-4 ile filtreleyebilir miyiz" sorusuna cevap veriyor — AYNI formül,
AYNI sınıflandırma (Confluence/Nötr/Divergence).

Formül (DEĞİŞTİRİLMEDİ, test_vpmv_divergence.py'den):
  vol_score = normalize_volume_0_100(hacim)
  momentum_score = normalize_momentum_0_100(RSI(14)-50), yön-hizalı
  Confluence: vol_score>60 VE momentum_aligned>60
  Divergence: vol_score>70 VE momentum_aligned<50
  Nötr: diğerleri

Kullanım: python -m research.pattern_lab.madde4_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_rsi
from utils.preprocessing import normalize_momentum_0_100, normalize_volume_0_100

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_HISTORY_BARS = 60
_FORWARD_BARS = 24
_MAX_SIGNALS = 6000
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _classify(df: pd.DataFrame, signal_type: str) -> tuple[str, float, float] | None:
    try:
        rsi_centered = calculate_rsi(df, period=14) - 50
        vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])
        momentum_score = float(normalize_momentum_0_100(rsi_centered).iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    momentum_aligned = momentum_score if signal_type == "Long" else (100.0 - momentum_score)
    if vol_score > 60 and momentum_aligned > 60:
        cls = "Confluence"
    elif vol_score > 70 and momentum_aligned < 50:
        cls = "Divergence"
    else:
        cls = "Nötr"
    return cls, vol_score, momentum_aligned


def _fetch_signals(cur) -> pd.DataFrame:
    # 21 Tem — TAM ÖLÇEK doğrulama: orijinal Madde-4 bulgusu (test_vpmv_
    # divergence.py, n=44.312, TÜM kapalı sinyaller) ile aynı kapsamda,
    # rastgele örneklem/LIMIT YOK — kullanıcı isteği "orijinal bulguyu da
    # temiz yöntemle yeniden test et".
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, opened_at, open_price
        FROM signals
        WHERE interval IN ('5m','15m','1h','4h') AND open_price IS NOT NULL AND open_price > 0
        ORDER BY opened_at ASC
        """
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "open_price"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_bars_before(cur, symbol: str, interval: str, before, n: int) -> pd.DataFrame:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT high, low, close, volume FROM {table} WHERE symbol=%s AND bucket < %s ORDER BY bucket DESC LIMIT %s",
        (symbol, before, n),
    )
    rows = cur.fetchall()
    if len(rows) < n:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame(rows, columns=["high", "low", "close", "volume"])


def _fetch_forward_price(cur, symbol: str, interval: str, after, n_bars: int) -> float | None:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket >= %s ORDER BY bucket ASC LIMIT 1 OFFSET %s",
        (symbol, after, n_bars - 1),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} sinyal örneklendi (status/close_reason önemsiz)")

    records = []
    for i, row in signals.iterrows():
        bars = _fetch_bars_before(cur, row["symbol"], row["interval"], row["opened_at"], _HISTORY_BARS)
        if len(bars) < _HISTORY_BARS:
            continue
        result = _classify(bars, row["signal_type"])
        if result is None:
            continue
        cls, vol_score, mom_aligned = result
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        records.append({
            "cls": cls, "vol_score": vol_score, "mom_aligned": mom_aligned,
            "fwd_ret": fwd_ret, "opened_at": row["opened_at"],
        })
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için Madde-4 sınıflandırması + {_FORWARD_BARS}-bar ileri getiri\n")
    if df.empty:
        return

    print("=== [1] Sınıf bazında sabit-ileri-getiri (TEMİZ hedef) ===")
    g = df.groupby("cls")["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    conf = df[df["cls"] == "Confluence"]["fwd_ret"]
    div = df[df["cls"] == "Divergence"]["fwd_ret"]
    print(f"\n  Confluence (n={len(conf)}): {_stats(conf.to_numpy())}")
    print(f"  Divergence (n={len(div)}): {_stats(div.to_numpy())}")

    print("\n=== [2] Korelasyon (momentum_aligned, sürekli değişken olarak) ===")
    rho, p = spearmanr(df["mom_aligned"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    print("\n=== [3] Placebo (sınıf etiketini karıştır) ===")
    rng = np.random.default_rng(42)
    real_diff = conf.mean() - div.mean() if len(conf) > 0 and len(div) > 0 else float("nan")
    labels = df["cls"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        fake_conf = target[shuffled == "Confluence"]
        fake_div = target[shuffled == "Divergence"]
        if len(fake_conf) == 0 or len(fake_div) == 0:
            continue
        fake_diff = fake_conf.mean() - fake_div.mean()
        if abs(fake_diff) >= abs(real_diff):
            count_ge += 1
    print(f"  gerçek Confluence-Divergence farkı={real_diff:+.4f} — rastgele etiketlemede "
          f"aynı/daha büyük fark sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    print("\n=== [4] Split-period (çöküş vs toparlanma) ===")
    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            c = sub[sub["cls"] == "Confluence"]["fwd_ret"]
            d = sub[sub["cls"] == "Divergence"]["fwd_ret"]
            print(f"  {label} (n={len(sub)}): Confluence ort={c.mean():+.4f}(n={len(c)}) "
                  f"| Divergence ort={d.mean():+.4f}(n={len(d)})")
        else:
            print(f"  {label}: yetersiz örnek (n={len(sub)})")

    print("\n=== [5] Kronolojik yarı-yarı (Confluence grubu İÇİNDE) ===")
    conf_sorted = df[df["cls"] == "Confluence"].sort_values("opened_at")
    if len(conf_sorted) >= 50:
        mid = conf_sorted["opened_at"].iloc[len(conf_sorted)//2]
        fh = _stats(conf_sorted[conf_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(conf_sorted[conf_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"  ilk yarı: {fh}")
        print(f"  ikinci yarı: {sh}")
    else:
        print("  yetersiz örnek")


if __name__ == "__main__":
    main()
