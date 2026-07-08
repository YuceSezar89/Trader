"""
EVOL ekonomik anlamı + Confluence katmanlaması — Pattern Lab.

Madde 5: EVOL'ü doğrudan bir FİLTRE kuralı olarak uygulayıp Profit Factor (PF)
ve Win Rate (WR) üzerindeki gerçek etkisini ölçer — sadece korelasyon değil,
"gerçekten para kazandırır mı" sorusuna cevap. Bkz. [[project_devisso_ersi]]
EVOL testi (`evol_bt.py`): Short'ta rho=-0.095, 3 sağlamlık kapısını geçti,
düşük EVOL = daha iyi Short. Long'da rho=+0.048 ama split-period'da kısmi.

Madde 6: EVOL'ün, projenin en güçlü doğrulanmış bulgusu olan Confluence/
Divergence (hacim-momentum uyumu, `test_vpmv_divergence.py`, hoca madde 4)
sınıflandırmasının İÇİNDE ek ayrım sağlayıp sağlamadığını test eder — aynı
bilgiyi mi tekrarlıyor yoksa üstüne bir şey mi katıyor?

Metodoloji: `evol_bt.py::_compute_evol` ve `test_vpmv_divergence.py::_classify`
BİREBİR AYNI şekilde (import edilerek, kopyalanmadan) kullanılıyor — tek fark,
ikisinin de ihtiyaç duyduğu bar sayısını (220) TEK bir fetch ile karşılayıp
ikisini de aynı pencereden hesaplamak (DB round-trip'i yarıya indirmek için).

Kullanım: python -m research.pattern_lab.evol_backtest
"""
import warnings

import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.evol_bt import _compute_evol
from research.pattern_lab.test_vpmv_divergence import _classify

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m"}
_BARS_NEEDED = 220  # test_vpmv_divergence'ın ihtiyacı (220) >= evol_bt'nin (130)


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> "pd.DataFrame | None":
    cagg = _CAGG.get(interval)
    if not cagg:
        return None
    cur.execute(f"""
        SELECT bucket AS open_time, open, high, low, close, volume
        FROM {cagg}
        WHERE symbol = %s AND bucket <= %s
        ORDER BY bucket DESC
        LIMIT %s
    """, (symbol, opened_at, _BARS_NEEDED))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    return df.iloc[::-1].reset_index(drop=True)


def _fetch_signals(cur) -> list:
    cur.execute("""
        SELECT id, symbol, interval, opened_at, signal_type, realized_pnl
        FROM signals
        WHERE status='closed' AND realized_pnl IS NOT NULL
          AND interval IN ('5m', '15m')
        ORDER BY symbol, interval, opened_at
    """)
    return cur.fetchall()


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def _wr(pnls: pd.Series) -> float:
    return float((pnls > 0).mean())


def _summarize(sub: pd.DataFrame, label: str) -> None:
    print(f"  {label}: n={len(sub)}, PF={_pf(sub['realized_pnl']):.3f}, "
          f"WR={_wr(sub['realized_pnl'])*100:.1f}%, ort_pnl={sub['realized_pnl'].mean():+.3f}")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    signals = _fetch_signals(cur)
    print(f"Toplam kapanmış sinyal (5m/15m): {len(signals)}")

    rows = []
    for i, sig in enumerate(signals, 1):
        bars = _fetch_bars(cur, sig["symbol"], sig["interval"], sig["opened_at"])
        if bars is None or len(bars) < 60:
            continue
        evol = _compute_evol(bars)
        cls_out = _classify(bars, sig["signal_type"])
        if evol is None or cls_out is None:
            continue
        cls, _vol_score, _momentum_aligned = cls_out
        rows.append({
            "symbol": sig["symbol"], "signal_type": sig["signal_type"],
            "interval": sig["interval"], "opened_at": sig["opened_at"],
            "evol": evol, "cls": cls, "realized_pnl": sig["realized_pnl"],
        })
        if i % 5000 == 0:
            print(f"  [{i}/{len(signals)}] işlendi, {len(rows)} geçerli satır")

    conn.close()
    df = pd.DataFrame(rows)
    print(f"\nGeçerli satır (EVOL + Confluence ikisi de hesaplandı): {len(df)}\n")

    # ── Madde 5: EVOL doğrudan filtre olarak — PF/WR ─────────────────────
    print("=== MADDE 5: EVOL FİLTRE OLARAK (Profit Factor / Win Rate) ===")
    print("Short — beklenti: düşük EVOL iyi, yüksek EVOL kötü")
    short = df[df["signal_type"] == "Short"]
    _summarize(short, "baseline (tümü)      ")
    _summarize(short[short["evol"] < 35], "filtre: EVOL<35 (iyi)")
    _summarize(short[short["evol"] >= 65], "filtre: EVOL>=65 (kötü)")

    print("\nLong — beklenti: yüksek EVOL iyi, düşük EVOL kötü (daha zayıf/rejime bağlı)")
    long_ = df[df["signal_type"] == "Long"]
    _summarize(long_, "baseline (tümü)      ")
    _summarize(long_[long_["evol"] >= 65], "filtre: EVOL>=65 (iyi)")
    _summarize(long_[long_["evol"] < 35], "filtre: EVOL<35 (kötü)")

    # ── Madde 6: Confluence içinde EVOL ek ayrım sağlıyor mu ─────────────
    print("\n=== MADDE 6: CONFLUENCE/DIVERGENCE İÇİNDE EVOL KATMANLAMASI ===")
    for sig_type, evol_good, evol_bad in [("Short", 35, 65), ("Long", 65, 35)]:
        print(f"\n-- {sig_type} --")
        sub = df[df["signal_type"] == sig_type]
        for cls in ["Confluence (gerçek hareket)", "Nötr", "Divergence (olası manipülasyon)"]:
            cls_sub = sub[sub["cls"] == cls]
            if len(cls_sub) < 20:
                print(f"  [{cls}] örneklem yetersiz (n={len(cls_sub)})")
                continue
            print(f"  [{cls}] n={len(cls_sub)}")
            if sig_type == "Short":
                good = cls_sub[cls_sub["evol"] < evol_good]
                bad = cls_sub[cls_sub["evol"] >= evol_bad]
            else:
                good = cls_sub[cls_sub["evol"] >= evol_good]
                bad = cls_sub[cls_sub["evol"] < evol_bad]
            _summarize(cls_sub, "    tümü (evol filtresiz)")
            _summarize(good, "    + EVOL iyi filtre     ")
            _summarize(bad, "    + EVOL kötü filtre    ")


if __name__ == "__main__":
    main()
