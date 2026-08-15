"""
Trajectory Lab — "sinyal-zamanı" göz taraması: bar-zamanı DEĞİL, aynı
sembol+aile+yönün ARDIŞIK SİNYALLERİ arasında momentum(VPMV)/katılım
(CVD Level)/fiyat tepkisi(close_pos) nasıl gelişiyor? (bkz.
CONTEXT_LAB_STATUS.md, 8 Ağustos — "Zamansal Doğrulama" katmanı)

behavior_scan.py'nin AYNI felsefesi: gösterge/formül/threshold YOK,
sadece ham göz taraması. Fark: x-ekseni bar değil, ARDIŞIK SİNYAL
(S-3,S-2,S-1,S0) — her sinyal arasındaki GERÇEK zaman farkı etiketli.

Dizi tanımı (kullanıcı, 8 Ağustos):
  - Aynı sembol + aynı sinyal ailesi + aynı yön
  - S0 = winner/loser (sınıflandırdığımız sinyal)
  - S(-1),S(-2),S(-3) = SAME sembolün hemen ÖNCEKİ 3 sinyali —
    outcome'u winner/loser/NEUTRAL fark etmez (gerçek ardışıklık,
    aradaki neutral sinyaller ATLANMAZ)
  - Zaman boşluğu için kesme YOK (ilk aşama)

3 boyut (kullanıcı, 8 Ağustos — "yeni/denenmemiş metriklerle
kirletmeyelim"): VPMV (momentum/enerji), CVD Level (katılım),
close_pos (fiyat tepkisi) — üçü de zaten 0-100 skalasında, ortak
eksende çizilebilir.

Kullanım:
    python -m research.trajectory_lab.signal_sequence_scan --indicator HA_Cross --direction Long --n 24
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine, get_session  # noqa: E402
from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab.behavior_scan import _select_balanced  # noqa: E402

_SEQ_LEN = 4  # S(-3),S(-2),S(-1),S0
_CLOSE_POS_BATCH_SIZE = 300
_COLORS = {"vpmv": "#1f77b4", "cvd_level": "#ff7f0e", "close_pos": "#2ca02c"}
_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_CORPUS_FILE_TMPL = "research/trajectory_corpus/{name}.parquet"
_CORPUS_NAMES = {
    ("HA_Cross", "Long"): "HA_Cross_Long",
    ("RSI_Cross(9,24)", "Long"): "RSI_Cross_9,24_Long",
    ("Supertrend(10,3.0)", "Long"): "Supertrend_10,3.0_Long",
}


def _load_snapshots(indicator: str, direction: str) -> pd.DataFrame:
    """HA_Cross_Long.parquet vb.'den HER sinyalin (winner/loser/neutral
    dahil) t=0 anındaki vpmv+cvd_level'ini çıkarır — yeni sorgu YOK."""
    name = _CORPUS_NAMES[(indicator, direction)]
    df = pd.read_parquet(_CORPUS_FILE_TMPL.format(name=name))
    meta = df[["signal_id", "symbol", "t0", "outcome"]].drop_duplicates().set_index("signal_id")
    out = pd.DataFrame(index=meta.index)
    for metric in ("vpmv", "cvd_level"):
        sub = df[(df["metric"] == metric) & (df["t_offset"] == 0)].set_index("signal_id")["value"]
        out[metric] = sub
    out["symbol"] = meta["symbol"]
    out["t0"] = meta["t0"]
    out["outcome"] = meta["outcome"]
    return out.dropna(subset=["vpmv", "cvd_level"]).sort_values("t0").reset_index()


def _build_sequences(snapshots: pd.DataFrame) -> list[list]:
    """Her sembol için kronolojik sinyal listesi, ardından S0=winner/loser
    olan ve en az 3 öncülü bulunan HER sinyal için 4'lü pencere."""
    sequences = []
    for symbol, grp in snapshots.groupby("symbol"):
        grp = grp.sort_values("t0").reset_index(drop=True)
        for i in range(_SEQ_LEN - 1, len(grp)):
            s0 = grp.iloc[i]
            if s0["outcome"] not in ("winner", "loser"):
                continue
            window = grp.iloc[i - _SEQ_LEN + 1 : i + 1].reset_index(drop=True)
            sequences.append({"symbol": symbol, "opened_at": s0["t0"], "outcome": s0["outcome"], "window": window})
    return sequences


async def _fetch_close_pos_for_signals(signal_ids: list) -> pd.DataFrame:
    """Sadece SEÇİLEN küçük sinyal alt-kümesi için close_pos hesaplar
    (signal_anatomy.py'nin AYNI yöntemi — 1m alt-mumlar → close_pos —
    ama tüm corpus yerine sadece bu hedefli id listesi için)."""
    async with get_session() as session:
        result = await session.execute(
            text("SELECT id, symbol, interval, opened_at FROM signals WHERE id = ANY(:ids)"),
            {"ids": list(signal_ids)},
        )
        rows = result.all()
    meta = pd.DataFrame(rows, columns=["signal_id", "symbol", "interval", "opened_at"])
    meta = meta[meta["interval"] != "1m"].reset_index(drop=True)

    close_pos_vals = {}
    for interval, grp in meta.groupby("interval"):
        iv_min = _INTERVAL_MINUTES[interval]
        n = len(grp)
        for i in range(0, n, _CLOSE_POS_BATCH_SIZE):
            batch = grp.iloc[i : i + _CLOSE_POS_BATCH_SIZE]
            starts = batch["opened_at"].tolist()
            ends = [ts + pd.Timedelta(minutes=iv_min) for ts in starts]
            async with get_session() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT sw.signal_id, p.timestamp, p.open, p.high, p.low, p.close
                        FROM unnest(
                            CAST(:signal_ids AS int[]), CAST(:symbols AS text[]),
                            CAST(:starts AS timestamp[]), CAST(:ends AS timestamp[])
                        ) AS sw(signal_id, symbol, start_ts, end_ts)
                        JOIN price_data p
                          ON p.symbol = sw.symbol AND p.interval = '1m'
                         AND p.timestamp >= sw.start_ts AND p.timestamp < sw.end_ts
                        ORDER BY sw.signal_id, p.timestamp
                        """
                    ),
                    {
                        "signal_ids": batch["signal_id"].tolist(),
                        "symbols": batch["symbol"].tolist(),
                        "starts": starts,
                        "ends": ends,
                    },
                )
                rows = result.all()
            sub_df = pd.DataFrame(rows, columns=["signal_id", "timestamp", "open", "high", "low", "close"])
            for signal_id, g in sub_df.groupby("signal_id"):
                g = g.sort_values("timestamp")
                o, h, low_, c = g["open"].iloc[0], g["high"].max(), g["low"].min(), g["close"].iloc[-1]
                rng = h - low_
                if rng > 0:
                    close_pos_vals[signal_id] = (c - low_) / rng * 100.0
            print(f"  ... close_pos {interval}: {min(i + _CLOSE_POS_BATCH_SIZE, n)}/{n} işlendi", end="\r")
        print()

    return pd.DataFrame({"signal_id": list(close_pos_vals.keys()), "close_pos": list(close_pos_vals.values())})


def _fmt_gap(td: pd.Timedelta) -> str:
    total_min = int(td.total_seconds() / 60)
    if total_min < 60:
        return f"{total_min}m"
    h, m = divmod(total_min, 60)
    if h < 24:
        return f"{h}h{m}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}g{h}s" if h else f"{d}g"


def _plot_sequences(sequences: list[dict], close_pos_map: dict, label: str, out_path: str) -> None:
    cols = 4
    rows = -(-len(sequences) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for ax, seq in zip(axes, sequences):
        window = seq["window"]
        x = list(range(_SEQ_LEN))
        vpmv_vals = window["vpmv"].tolist()
        cvd_vals = window["cvd_level"].tolist()
        cp_vals = [close_pos_map.get(sid, np.nan) for sid in window["signal_id"]]

        ax.plot(x, vpmv_vals, "o-", color=_COLORS["vpmv"], label="VPMV", linewidth=1.4, markersize=4)
        ax.plot(x, cvd_vals, "o-", color=_COLORS["cvd_level"], label="CVD", linewidth=1.4, markersize=4)
        ax.plot(x, cp_vals, "o-", color=_COLORS["close_pos"], label="close_pos", linewidth=1.4, markersize=4)
        ax.set_ylim(-5, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(["S-3", "S-2", "S-1", "S0"], fontsize=6)
        ax.tick_params(labelsize=5)

        for j in range(1, _SEQ_LEN):
            gap = window["t0"].iloc[j] - window["t0"].iloc[j - 1]
            ax.text(j - 0.5, -12, _fmt_gap(gap), ha="center", fontsize=5.5, color="gray")

        outcome_color = "#2ca02c" if seq["outcome"] == "winner" else "#d62728"
        ax.set_title(f"{seq['symbol']}", fontsize=7, color=outcome_color)

    for ax in axes[len(sequences):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=_COLORS["vpmv"], marker="o", label="VPMV"),
        plt.Line2D([0], [0], color=_COLORS["cvd_level"], marker="o", label="CVD Level"),
        plt.Line2D([0], [0], color=_COLORS["close_pos"], marker="o", label="close_pos"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"{label} — sinyal-zamanı dizileri (n={len(sequences)}), gösterge/formül YOK — "
        f"x: S-3→S0 (ARDIŞIK sinyaller, bar değil), altta gerçek zaman farkı",
        fontsize=10, y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


async def run(indicator: str, direction: str, n: int) -> None:
    snapshots = _load_snapshots(indicator, direction)
    print(f"Toplam sinyal (winner+loser+neutral, vpmv+cvd_level mevcut): {len(snapshots)}")

    all_sequences = _build_sequences(snapshots)
    winner_seqs = [s for s in all_sequences if s["outcome"] == "winner"]
    loser_seqs = [s for s in all_sequences if s["outcome"] == "loser"]
    print(f"Uygun dizi (>=3 öncülü olan): winner={len(winner_seqs)}, loser={len(loser_seqs)}")

    # _select_balanced kendi içinde sort+reset_index yapıyor — orijinal listeye
    # pozisyonel index ile geri dönmek YANLIŞ eşleşir, bu yüzden seq_idx kolonu
    # ile açıkça takip ediyoruz.
    winner_meta = pd.DataFrame(
        [{"symbol": s["symbol"], "opened_at": s["opened_at"], "seq_idx": i} for i, s in enumerate(winner_seqs)]
    )
    loser_meta = pd.DataFrame(
        [{"symbol": s["symbol"], "opened_at": s["opened_at"], "seq_idx": i} for i, s in enumerate(loser_seqs)]
    )
    winner_sel = [winner_seqs[i] for i in _select_balanced(winner_meta, n)["seq_idx"]]
    loser_sel = [loser_seqs[i] for i in _select_balanced(loser_meta, n)["seq_idx"]]

    w_symbols = {s["symbol"] for s in winner_sel}
    l_symbols = {s["symbol"] for s in loser_sel}
    print(f"Seçilen: winner={len(winner_sel)}, loser={len(loser_sel)}, ortak sembol={len(w_symbols & l_symbols)}")

    all_signal_ids = set()
    for seq in winner_sel + loser_sel:
        all_signal_ids.update(seq["window"]["signal_id"].tolist())
    print(f"close_pos hesaplanacak hedefli sinyal sayısı: {len(all_signal_ids)}")

    cp_df = await _fetch_close_pos_for_signals(list(all_signal_ids))
    close_pos_map = dict(zip(cp_df["signal_id"], cp_df["close_pos"]))
    print(f"close_pos hesaplanabilen: {len(close_pos_map)}/{len(all_signal_ids)}")

    os.makedirs(C.REPORT_DIR, exist_ok=True)
    tag = f"{indicator}_{direction}".replace("(", "").replace(")", "").replace(",", "_")
    _plot_sequences(winner_sel, close_pos_map, f"{indicator} {direction} — WINNER", os.path.join(C.REPORT_DIR, f"seqscan_{tag}_winner.png"))
    _plot_sequences(loser_sel, close_pos_map, f"{indicator} {direction} — LOSER", os.path.join(C.REPORT_DIR, f"seqscan_{tag}_loser.png"))


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default="HA_Cross")
    parser.add_argument("--direction", default="Long", choices=["Long", "Short"])
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    try:
        await run(args.indicator, args.direction, args.n)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
