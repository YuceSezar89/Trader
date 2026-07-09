"""
Kohort-içi "canlı yarış" (rolling/dinamik kohort) pilotu — Supertrend 5m.

[[project_signal_radar_vision]]'daki Opsiyon B tasarımının ilk denemesi.
Önceki statik-bar kohort tanımı (cohort_rank_bt.py, `_COHORT_KEY` içinde
`opened_at` sabit — yani "aynı bar kapanışında tetiklenenler") ÇÜRÜMÜŞTÜ
(placebo'dan ayırt edilemeyen etki, split-period'da kayboluyor).

Kullanıcının kritik düzeltmesi: coin'ler farklı zamanlarda "kalkış" yapıyor —
dün sinyal alan bir coin bugün hâlâ pozisyonda olabilir, kohort statik bir ana
değil SÜREKLİ değişen bir yarışa benziyor. Bu script her sinyal S açıldığında
(T_S) aynı (indicators, signal_type, interval)'dan T_S anında HÂLÂ AÇIK olan
(opened_at < T_S AND (closed_at IS NULL OR closed_at > T_S)) tüm diğer
sembolleri kohort sayıyor — "aynı bar" değil "aynı yarış".

Her üye (anchor dahil) için devisso_score (ERSI) + cvd_slope T_S anında
YENİDEN hesaplanıyor (kendi sinyalinin eski/farklı-zamanlı değeri değil) —
`signals/signal_processor.py::_compute_devisso_score` ve cvd_slope formülüyle
birebir aynı metodoloji, sadece bellek içinde, DB'ye hiçbir şey yazılmıyor.

OI dahil değil — `oi_data` sadece sinyal ateşlendiği anın TTL'li Redis
snapshot'ı, ham zaman serisi olarak DB'de yok, keyfi bir geçmiş anda yeniden
hesaplanamaz.

Hedef değişken realized_pnl DEĞİL (kendi SL/TP/timeout gürültüsünü taşıyor) —
T_S'den başlayan SABİT ufuklu (4h) ham ileri-yön fiyat getirisi. Anchor ve her
peer AYNI başlangıç anından (T_S) değerlendiriliyor, kimin o andan itibaren
daha çok hareket ettiği net oluyor.

Bu projenin battle-tested kapıları uygulanıyor (bkz. [[project_pattern_lab]]):
placebo (rank kohort içinde rastgele karıştırılır), split-period.

Kullanım: python -m research.pattern_lab.rolling_cohort_bt
"""
import warnings

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_rsi

_INDICATOR = "Supertrend(10,3.0)"
_INTERVAL = "5m"
_CAGG = "cagg_5m"
_BARS_NEEDED = 220  # z-score EMA200/std200 gerektiriyor (>=210), devisso/cvd için zaten fazlasıyla yeterli
_HORIZON_HOURS = 4
_HORIZON_BARS = _HORIZON_HOURS * 60 // 5  # 5m bar başına
_MIN_COHORT_SIZE = 2


def _connect():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _fetch_all_signals(cur) -> pd.DataFrame:
    """Bu indikatör+interval'daki TÜM sinyaller (açık + kapalı) — hem anchor
    hem peer havuzu olarak kullanılıyor. Açık sinyaller closed_at=NULL, yani
    "hâlâ açık" testinde sonsuza kadar peer olabilirler (doğru davranış)."""
    cur.execute("""
        SELECT id, symbol, signal_type, opened_at, closed_at, realized_pnl, status
        FROM signals
        WHERE indicators = %s AND interval = %s
        ORDER BY opened_at
    """, (_INDICATOR, _INTERVAL))
    return pd.DataFrame(cur.fetchall())


def _compute_devisso_score(df: pd.DataFrame) -> "float | None":
    """signals/signal_processor.py::_compute_devisso_score ile birebir aynı."""
    try:
        if len(df) < 30:
            return None
        close = df["close"].astype(float)
        rsi = calculate_rsi(df, period=14)
        price_pct = close.pct_change() * 100.0
        raw = price_pct / rsi.diff().replace(0.0, np.nan)
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.dropna()
        if len(valid) < 20:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        rank = float((recent < current).sum()) / len(recent)
        return round(rank * 100.0, 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _compute_cvd_slope(df: pd.DataFrame) -> "float | None":
    """signals/signal_processor.py'deki cvd_slope hesaplamasıyla birebir aynı."""
    try:
        if len(df) < 15:
            return None
        if "buy_volume" in df.columns and df["buy_volume"].notna().any():
            bv = df["buy_volume"].fillna(
                df["volume"] * (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(lower=1e-8)
            )
        else:
            hl = (df["high"] - df["low"]).clip(lower=1e-8)
            bv = df["volume"] * (df["close"] - df["low"]) / hl
        cvd = (2 * bv - df["volume"]).cumsum()
        avg_vol = df["volume"].rolling(10).mean().clip(lower=1e-8)
        return round(float((cvd.diff().rolling(10).mean() / avg_vol).iloc[-1]), 4)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _compute_zscore(df: pd.DataFrame) -> "float | None":
    """signals/signal_processor.py::z_score_entry ile birebir aynı — fiyatın
    kendi EMA200'ünden kaç std sapmış olduğu. Kullanıcının tarifi: "aynı sinyali
    veren coinler arasından ORTALAMADAN EN ÇOK SAPANI" — yön değil BÜYÜKLÜK
    önemli, bu yüzden kohort sıralamasında mutlak değeri kullanılacak."""
    try:
        if len(df) < 210 or "close" not in df.columns:
            return None
        closes = df["close"].astype(float)
        ema200 = closes.ewm(span=200, adjust=False).mean()
        std200 = closes.rolling(200).std()
        return round(float((closes.iloc[-1] - ema200.iloc[-1]) / (std200.iloc[-1] + 1e-12)), 3)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _fetch_bars_at(cur, symbol: str, at) -> "pd.DataFrame | None":
    """T_S anına kadarki (dahil) son _BARS_NEEDED bar — evol_bt.py::_fetch_bars
    ile aynı desen, ama sembol/an keyfi (anchor'ın kendi sembolü olmak zorunda
    değil, herhangi bir peer olabilir)."""
    cur.execute(f"""
        SELECT bucket AS open_time, open, high, low, close, volume
        FROM {_CAGG}
        WHERE symbol = %s AND bucket <= %s
        ORDER BY bucket DESC
        LIMIT %s
    """, (symbol, at, _BARS_NEEDED))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    return df.iloc[::-1].reset_index(drop=True)


def _fetch_forward_close(cur, symbol: str, at) -> "float | None":
    """T_S + ufuk anında/hemen sonrasında ilk bar'ın close'u."""
    cur.execute(f"""
        SELECT close FROM {_CAGG}
        WHERE symbol = %s AND bucket >= %s
        ORDER BY bucket ASC
        LIMIT 1
    """, (symbol, at))
    row = cur.fetchone()
    return float(row["close"]) if row else None


def _fwd_return(entry_price: float, exit_price: float, signal_type: str) -> float:
    if signal_type == "Short":
        return (entry_price - exit_price) / entry_price * 100.0
    return (exit_price - entry_price) / entry_price * 100.0


def _build_rows(cur, all_sig: pd.DataFrame) -> list:
    """Her (anchor, kohort üyesi) çifti için satır üretir — anchor dahil."""
    horizon = pd.Timedelta(hours=_HORIZON_HOURS)
    anchors = all_sig[
        (all_sig["status"] == "closed") & all_sig["realized_pnl"].notna()
    ].reset_index(drop=True)

    rows = []
    bars_cache: dict = {}  # (symbol, at) -> devisso_score/cvd_slope, aynı (sembol,an) birden fazla kohortta tekrar edebilir

    def _features_at(symbol: str, at) -> "tuple[float, float, float] | None":
        key = (symbol, at)
        if key in bars_cache:
            return bars_cache[key]
        bars = _fetch_bars_at(cur, symbol, at)
        if bars is None or len(bars) < 30:
            bars_cache[key] = None
            return None
        d = _compute_devisso_score(bars)
        c = _compute_cvd_slope(bars)
        z = _compute_zscore(bars)  # >=210 bar gerektirir, kısa geçmişli sembollerde None dönebilir
        if d is None or c is None or z is None:
            bars_cache[key] = None
            return None
        bars_cache[key] = (d, c, z)
        return bars_cache[key]

    n_total = len(anchors)
    for i, anchor in enumerate(anchors.itertuples(), 1):
        t_s = anchor.opened_at
        sig_type = anchor.signal_type

        peers = all_sig[
            (all_sig["signal_type"] == sig_type)
            & (all_sig["symbol"] != anchor.symbol)
            & (all_sig["opened_at"] < t_s)
            & (all_sig["closed_at"].isna() | (all_sig["closed_at"] > t_s))
        ]
        if len(peers) < _MIN_COHORT_SIZE - 1:
            continue

        members = [(anchor.symbol, True)] + [(p, False) for p in peers["symbol"].tolist()]

        cohort_id = anchor.id
        cohort_rows = []
        for symbol, is_anchor in members:
            feats = _features_at(symbol, t_s)
            if feats is None:
                continue
            devisso, cvd, zscore = feats
            entry_price = _fetch_forward_close(cur, symbol, t_s)
            exit_price = _fetch_forward_close(cur, symbol, t_s + horizon)
            if entry_price is None or exit_price is None:
                continue
            fwd_ret = _fwd_return(entry_price, exit_price, sig_type)
            cohort_rows.append({
                "cohort_id": cohort_id, "symbol": symbol, "is_anchor": is_anchor,
                "signal_type": sig_type, "opened_at": t_s,
                "devisso_score": devisso, "cvd_slope": cvd, "zscore_abs": abs(zscore),
                "fwd_return": fwd_ret,
            })

        if len(cohort_rows) >= _MIN_COHORT_SIZE:
            rows.extend(cohort_rows)

        if i % 200 == 0:
            print(f"  [{i}/{n_total}] anchor işlendi, {len(rows)} satır, {len(bars_cache)} önbellek")

    return rows


def _add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Üç bileşenin ayrı ayrı rank'i + iki kompozit varyant: eski (devisso+cvd,
    z-score olmadan — önceki turla karşılaştırma için) ve yeni (üçü birden)."""
    df = df.copy()
    df["devisso_rank"] = df.groupby("cohort_id")["devisso_score"].rank(pct=True)
    df["cvd_rank"] = df.groupby("cohort_id")["cvd_slope"].rank(pct=True)
    df["zscore_rank"] = df.groupby("cohort_id")["zscore_abs"].rank(pct=True)
    df["composite_rank_2"] = df[["devisso_rank", "cvd_rank"]].mean(axis=1)
    df["composite_rank_3"] = df[["devisso_rank", "cvd_rank", "zscore_rank"]].mean(axis=1)
    return df


def _placebo(df: pd.DataFrame, rank_col: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[rank_col] = (
        out.groupby("cohort_id")[rank_col].transform(lambda s: rng.permutation(s.values))
    )
    return out


def _report_correlation(df: pd.DataFrame, rank_col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            print(f"  [{label}] {sig_type}: n={len(sub)} (yetersiz, atlandı)")
            continue
        rho, p = spearmanr(sub[rank_col], sub["fwd_return"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _report_buckets(df: pd.DataFrame, rank_col: str, label: str) -> None:
    """ort_getiri yanına medyan + "hareketsiz kaldı" oranı (|getiri|<1%) eklendi —
    ortalamanın birkaç aykırı büyük hareketle mi şişirildiğini, yoksa dilimin
    genelinde mi hareket olduğunu ayırt etmek için (kullanıcının itirazı: "en az
    ayrışan da ters sinyal gelene kadar hiç hareket etmeden kapanabilir")."""
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        tercile = pd.qcut(sub[rank_col], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["fwd_return"].agg(
            ort_getiri="mean",
            medyan_getiri="median",
            n="count",
            oran_pozitif=lambda s: (s > 0).mean(),
            hareketsiz_oran=lambda s: (s.abs() < 1.0).mean(),
        )
        print(f"  [{label}] {sig_type}:")
        print(g.to_string().replace("\n", "\n    "))


def _report_elimination(df: pd.DataFrame, rank_col: str, label: str, elim_frac: float = 1 / 3) -> None:
    """"Kim hareket etmeyecek onu bulup eleriz" testi: rank_col'a göre en alt
    dilimi (en düşük skor = en durgun aday) ELE, kalan havuzun (orta+üst)
    hareketsiz oranı/ortalaması/medyanı TAM HAVUZDAN daha mı iyi bak.
    Gerçek bir eleme filtresiyse: elenen grupta hareketsiz oranı yüksek OLMALI,
    kalan grupta belirgin biçimde DÜŞMELİ (sadece ortalama değil, medyan ve
    hareketsiz oranı da düzelmeli — aksi halde yine birkaç aykırı değerin
    eseri)."""
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        cutoff = sub[rank_col].quantile(elim_frac)
        elenen = sub[sub[rank_col] <= cutoff]
        kalan = sub[sub[rank_col] > cutoff]

        def _stats(s: pd.Series) -> dict:
            return {
                "n": len(s), "ort_getiri": s.mean(), "medyan_getiri": s.median(),
                "oran_pozitif": (s > 0).mean(), "hareketsiz_oran": (s.abs() < 1.0).mean(),
            }

        tum = _stats(sub["fwd_return"])
        el = _stats(elenen["fwd_return"])
        kal = _stats(kalan["fwd_return"])
        print(f"  [{label}] {sig_type} (alt %{elim_frac*100:.0f} elendi):")
        print(f"    TÜM HAVUZ : n={tum['n']:5d}  ort={tum['ort_getiri']:+.3f}  medyan={tum['medyan_getiri']:+.3f}  "
              f"pozitif={tum['oran_pozitif']:.3f}  hareketsiz={tum['hareketsiz_oran']:.3f}")
        print(f"    ELENEN    : n={el['n']:5d}  ort={el['ort_getiri']:+.3f}  medyan={el['medyan_getiri']:+.3f}  "
              f"pozitif={el['oran_pozitif']:.3f}  hareketsiz={el['hareketsiz_oran']:.3f}")
        print(f"    KALAN     : n={kal['n']:5d}  ort={kal['ort_getiri']:+.3f}  medyan={kal['medyan_getiri']:+.3f}  "
              f"pozitif={kal['oran_pozitif']:.3f}  hareketsiz={kal['hareketsiz_oran']:.3f}")


def main() -> None:
    conn = _connect()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    all_sig = _fetch_all_signals(cur)
    print(f"{_INDICATOR} / {_INTERVAL}: toplam {len(all_sig)} sinyal (açık+kapalı, peer havuzu)")

    anchors_n = len(all_sig[(all_sig["status"] == "closed") & all_sig["realized_pnl"].notna()])
    print(f"Anchor adayı (kapalı, realized_pnl dolu): {anchors_n}")

    print("\nKohort satırları üretiliyor (her anchor için peer'lar + T_S'de yeniden hesaplama)...")
    rows = _build_rows(cur, all_sig)
    conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("Hiç geçerli kohort satırı üretilemedi — çıkılıyor.")
        return

    n_cohorts = df["cohort_id"].nunique()
    print(f"\nToplam satır: {len(df)}, kohort (boyut>={_MIN_COHORT_SIZE}) sayısı: {n_cohorts}")
    cohort_sizes = df.groupby("cohort_id").size()
    print(f"Kohort boyutu (peer+anchor): medyan={cohort_sizes.median():.0f}, "
          f"min={cohort_sizes.min()}, maks={cohort_sizes.max()}")

    ranked = _add_ranks(df)
    mid = ranked["opened_at"].min() + (ranked["opened_at"].max() - ranked["opened_at"].min()) / 2

    # Tek tek (devisso, cvd, zscore) + iki kompozit (z-score'suz eski, z-score'lu yeni)
    # — hangi bileşen gerçekten bilgi taşıyor, z-score eklemek geliştiriyor mu bozuyor mu?
    variants = [
        ("devisso_rank", "SADECE DEVISSO_SCORE"),
        ("cvd_rank", "SADECE CVD_SLOPE"),
        ("zscore_rank", "SADECE Z-SCORE (mutlak sapma)"),
        ("composite_rank_2", "KOMPOZİT (devisso+cvd, ESKİ — z-score'suz)"),
        ("composite_rank_3", "KOMPOZİT (devisso+cvd+zscore, YENİ)"),
    ]

    for col, title in variants:
        print(f"\n########## {title} ##########")
        print("-- ana test --")
        _report_correlation(ranked, col, "gerçek")
        _report_buckets(ranked, col, "gerçek")
        print("-- placebo --")
        _report_correlation(_placebo(ranked, col), col, "placebo")
        print("-- split-period --")
        for half_name, half_df in [
            ("ilk_yari", ranked[ranked["opened_at"] < mid]),
            ("ikinci_yari", ranked[ranked["opened_at"] >= mid]),
        ]:
            print(f"  -- {half_name} ({len(half_df)} satır) --")
            _report_correlation(half_df, col, half_name)
        print("-- ELEME TESTİ (kim hareket etmeyecek, onu ele) --")
        _report_elimination(ranked, col, "gerçek")
        print("-- eleme testi, placebo (rastgele karıştırılmış rank ile) --")
        _report_elimination(_placebo(ranked, col), col, "placebo")


if __name__ == "__main__":
    main()
