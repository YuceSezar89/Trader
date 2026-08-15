"""
Trajectory Lab — Davranış Ailesi #1: göz taraması.

Amaç ölçüm değil, KEŞİF — hiçbir gösterge/metrik hesaplanmaz, sadece ham
OHLCV çizilir. Winner/loser örnekleri sembol ve zaman açısından deterministik
biçimde dengelenir (bkz. _select_balanced) — "güzel görünen" örnek elle
seçilmez.

Kullanım:
    python -m research.trajectory_lab.behavior_scan --indicator HA_Cross --direction Long --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine  # noqa: E402
from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab.corpus_builder import _fetch_case_signals, _fetch_window_klines  # noqa: E402

_UP, _DOWN = "#2ca02c", "#d62728"


def _select_balanced(meta: pd.DataFrame, n: int) -> pd.DataFrame:
    """Zamana göre sıralı havuzdan, tüm zaman aralığını eşit aralıklarla
    kapsayan n nokta seçer (np.linspace) — hiçbir sinyal görsel olarak
    önceden incelenmeden. Bir sembolün örneklemi domine etmesini önlemek
    için sembol başına üst sınır (max(2, ceil(n/10))) uygulanır; sınır
    dolarsa en yakın müsait indekse kayılır. Tamamen deterministik."""
    meta = meta.sort_values("opened_at").reset_index(drop=True)
    total = len(meta)
    if total <= n:
        return meta

    max_per_symbol = max(2, -(-n // 10))
    target_idx = np.linspace(0, total - 1, n)
    used: set[int] = set()
    counts: dict[str, int] = {}
    picked: list[int] = []

    for t in target_idx:
        base = int(round(t))
        offset = 0
        chosen = None
        while chosen is None and offset <= total:
            for cand in (base - offset, base + offset):
                if 0 <= cand < total and cand not in used:
                    sym = meta.loc[cand, "symbol"]
                    if counts.get(sym, 0) < max_per_symbol:
                        chosen = cand
                        break
            offset += 1
        if chosen is None:
            # sınır her yerde doldu (küçük havuz) — sınırı görmezden gel
            for cand in range(total):
                if cand not in used:
                    chosen = cand
                    break
        used.add(chosen)
        sym = meta.loc[chosen, "symbol"]
        counts[sym] = counts.get(sym, 0) + 1
        picked.append(chosen)

    return meta.loc[sorted(picked)].reset_index(drop=True)


def _plot_ohlc(ax, klines: pd.DataFrame, symbol: str) -> None:
    up = (klines["close"] >= klines["open"]).to_numpy()
    colors = np.where(up, _UP, _DOWN)
    ax.vlines(klines["t_offset"], klines["low"], klines["high"], colors=colors, linewidth=0.6)
    body_lo = klines[["open", "close"]].min(axis=1)
    body_hi = klines[["open", "close"]].max(axis=1)
    ax.vlines(klines["t_offset"], body_lo, body_hi, colors=colors, linewidth=2.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(C.OUTCOME_HORIZON_BARS, color="#1f77b4", linestyle=":", linewidth=1.1, alpha=0.8)
    ax.set_title(symbol, fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_yticklabels([])


async def build_scan(indicator: str, direction: str, n: int) -> None:
    signals_df = await _fetch_case_signals(indicator, direction)
    winners_meta = signals_df[signals_df["outcome"] == "winner"]
    losers_meta = signals_df[signals_df["outcome"] == "loser"]

    winners_sel = _select_balanced(winners_meta, n)
    losers_sel = _select_balanced(losers_meta, n)

    os.makedirs(C.REPORT_DIR, exist_ok=True)

    for label, sel in (("winner", winners_sel), ("loser", losers_sel)):
        cols = 6
        rows = -(-len(sel) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.0))
        axes = np.array(axes).reshape(-1)
        for ax, (_, sig) in zip(axes, sel.iterrows()):
            klines = await _fetch_window_klines(sig["symbol"], sig["interval"], sig["opened_at"])
            if klines.empty:
                ax.axis("off")
                continue
            window = klines[
                (klines["t_offset"] >= -C.WINDOW_PRE) & (klines["t_offset"] <= C.WINDOW_POST)
            ]
            _plot_ohlc(ax, window, sig["symbol"])
        for ax in axes[len(sel) :]:
            ax.axis("off")
        fig.suptitle(
            f"{indicator} {direction} — {label} örnekleri (n={len(sel)}), ham OHLCV, gösterge YOK — "
            f"kesikli çizgi: t=0 sinyal anı, noktalı çizgi: t=+{C.OUTCOME_HORIZON_BARS} karar noktası "
            f"(winner/loser SADECE burada belirleniyor)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(C.REPORT_DIR, f"behavior_scan_{indicator}_{direction}_{label}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"-> {out}")

    _write_method_note(indicator, direction, n, len(winners_meta), len(losers_meta), len(winners_sel), len(losers_sel))


def _write_method_note(
    indicator: str,
    direction: str,
    n: int,
    winner_pool: int,
    loser_pool: int,
    winner_n: int,
    loser_n: int,
) -> None:
    max_per_symbol = max(2, -(-n // 10))
    note = f"""# Behavior Scan — Seçim Yöntemi ({indicator} {direction})

Bu tarama KEŞİF amaçlıdır — hiçbir hipotez doğrulanmıyor/çürütülmüyor,
hiçbir gösterge/metrik hesaplanmadı. Grafikler sadece ham OHLCV.

## Havuz
- Winner havuzu: {winner_pool} sinyal → {winner_n} seçildi
- Loser havuzu: {loser_pool} sinyal → {loser_n} seçildi

## Seçim algoritması (deterministik, elle seçim YOK)
1. Winner ve loser havuzları ayrı ayrı `t0` (sinyal anı) zamanına göre
   kronolojik sıralandı.
2. `np.linspace(0, havuz_boyutu-1, {n})` ile tüm zaman aralığını eşit
   aralıklarla kapsayan {n} indeks belirlendi (farklı piyasa
   dönemlerini/rejimlerini temsil etmesi için).
3. Sembol başına üst sınır kondu: en fazla {max_per_symbol} örnek
   (bir sembolün örneklemi domine etmesini önlemek için). Sınır dolarsa
   en yakın müsait indekse kayıldı.
4. Hiçbir sinyal görsel olarak önceden incelenip "güzel görünüyor" diye
   seçilmedi — seçim tamamen zaman+sembol pozisyonuna göre yapıldı.

## Grafik
- Her panel bir sinyal: ham OHLC çubukları (yeşil=kapanış≥açılış,
  kırmızı=aksi).
- t=0 dikey KESİKLİ çizgi = sinyal anı.
- t=+{C.OUTCOME_HORIZON_BARS} dikey NOKTALI çizgi = karar noktası — winner/loser
  etiketi ({C.OUTCOME_METRIC}) SADECE bu barda ölçülüyor. Bu çizginin
  SAĞINDAKİ hareket (t+{C.OUTCOME_HORIZON_BARS}'tan sonrası, pencere t+{C.WINDOW_POST}'a kadar
  devam ediyor) etiketle İLGİSİZ — sadece "sonra ne oldu" merakı için
  gösteriliyor, göz taramasında yanıltmasın diye ayrı işaretlendi.
- Pencere: t∈[-{C.WINDOW_PRE},+{C.WINDOW_POST}] bar.
- Hacim eklenmedi (panel sayısı zaten yüksek, sadeliği korumak için) —
  gerekirse ayrı bir geçişte eklenebilir.
"""
    path = os.path.join(C.REPORT_DIR, f"behavior_scan_{indicator}_{direction}_method.md")
    with open(path, "w") as f:
        f.write(note)
    print(f"-> {path}")


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default="HA_Cross")
    parser.add_argument("--direction", default="Long", choices=["Long", "Short"])
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    try:
        await build_scan(args.indicator, args.direction, args.n)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
