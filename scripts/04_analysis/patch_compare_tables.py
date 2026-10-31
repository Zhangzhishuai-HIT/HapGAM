"""Apply two corrections of compare_models.py to existing comparison tables without re-running the bootstrap.

1. Bootstrap P values are capped at 1, and set to 1 where the two models' predictions are identical to numerical
   precision (|difference in r| and both CI limits < 1e-6; e.g. HapGAM-E with M = 0 vs EGBLUP), where the bootstrap
   otherwise only reflects the sign of rounding noise.  Holm-adjusted P values are recomputed per baseline.
2. The corrected resampled t-test (random k-fold scenarios, column nb_p) is Holm-adjusted across traits per
   baseline (column nb_p_holm), as specified in section 7 of the protocol.

usage: patch_compare_tables.py   (idempotent)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from compare_models import holm  # noqa: E402
from hapgam.paths import RESULTS  # noqa: E402

for f in sorted((RESULTS / "tables").glob("compare_*.tsv")):
    d = pd.read_csv(f, sep="\t")
    same = (d["diff"].abs() < 1e-6) & (d["ci_lo"].abs() < 1e-6) & (d["ci_hi"].abs() < 1e-6)
    old = d["p_holm"].copy()
    d["p_boot"] = np.minimum(d["p_boot"], 1.0)
    d.loc[same, "p_boot"] = 1.0
    d["p_holm"] = d.groupby("baseline")["p_boot"].transform(lambda s: holm(s.to_numpy()))
    if "nb_p" in d.columns:
        d["nb_p_holm"] = d.groupby("baseline")["nb_p"].transform(lambda s: holm(s.to_numpy()))
    changed = int((np.abs(d["p_holm"] - old) > 1e-12).sum())
    d.to_csv(f, sep="\t", index=False)
    if changed or same.any() or "nb_p" in d.columns:
        print(f"{f.name}: identical-prediction rows {int(same.sum())}, p_holm changed {changed}"
              + (", nb_p_holm added" if "nb_p" in d.columns else ""))
