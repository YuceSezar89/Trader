"""
Adım 4 — Eşleştirilmiş karşılaştırma + KEŞİF raporu.

Her özellik için:
  - vaka / kontrol medyanları
  - çift-içi fark (vaka − kontrol) medyanı ve çiftlerin % kaçında vaka üstün
  - Wilcoxon işaretli-sıra p (eşleştirilmiş, dağılım varsayımsız)
  - Cliff's delta (gruplar arası etki büyüklüğü, -1..+1)

Bu bir KEŞİF çalışmasıdır: n=20 çift, ~30 özellik → şans eseri 1-2 'bulgu'
beklenir. Hüküm Adım 5 (placebo/seed/pencere) + out-of-sample olmadan verilmez.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab import config as C

META_COLS = {"sembol", "rol", "t0"}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def run() -> pd.DataFrame:
    fm = pd.read_parquet(f"{C.CORPUS_DIR}/feature_matrix.parquet")
    feats = [c for c in fm.columns if c not in META_COLS]

    vaka = fm[fm["rol"] == "vaka"].set_index("t0")
    ctrl = fm[fm["rol"] == "kontrol"].set_index("t0")

    rows = []
    for f in feats:
        a = pd.to_numeric(vaka[f], errors="coerce")
        b = pd.to_numeric(ctrl[f], errors="coerce")
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(pair) < 8:
            continue
        diff = pair["a"] - pair["b"]
        try:
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        except ValueError:
            p = np.nan
        rows.append({
            "ozellik": f,
            "n_cift": len(pair),
            "vaka_med": round(float(a.median()), 3),
            "kontrol_med": round(float(b.median()), 3),
            "fark_med": round(float(diff.median()), 3),
            "vaka_ustun_%": round(float((diff > 0).mean() * 100), 0),
            "wilcoxon_p": round(float(p), 4),
            "cliffs_delta": round(cliffs_delta(a.to_numpy(), b.to_numpy()), 3),
        })
    res = pd.DataFrame(rows).sort_values("cliffs_delta", key=abs, ascending=False)

    os.makedirs(C.REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"{C.REPORT_DIR}/pattern_lab_{stamp}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Pattern Lab — Vaka-Kontrol KEŞİF Raporu\n\n")
        fh.write(f"Üretim: {datetime.now()}\n\n")
        fh.write("**UYARI: n=20 çift, ~30 özellik — bu rapor keşif amaçlıdır. ")
        fh.write("Şans eseri 1-2 'bulgu' beklenir; hüküm için Adım 5 sağlamlık ")
        fh.write("kapıları + gelecek dönem out-of-sample doğrulaması şarttır.**\n\n")
        fh.write("```\n" + res.to_string(index=False) + "\n```\n")
    res.to_parquet(f"{C.CORPUS_DIR}/comparison.parquet", index=False)
    print(f"Rapor: {path}\n")
    return res


if __name__ == "__main__":
    r = run()
    print(r.to_string(index=False))
