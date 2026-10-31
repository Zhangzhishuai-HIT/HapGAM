"""Print the numbers quoted in the manuscript text, from results/tables/all_metrics.tsv and compare_*.tsv.

usage: report_numbers.py [scenario ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import MODELS, PRIMARY, TRAITS, groups_won  # noqa: E402

SCEN = ["dev_region", "C1", "C2", "C3", "C4", "soynam_random", "soynam_lofo", "grin2soynam", "grin2nust"]
LINEAR = ["gblup", "egblup", "rkhs", "hapgblup"]
SINGLE = ["gblup", "egblup", "rkhs", "hapgblup", "bayesb", "lightgbm", "xgboost", "mlp", "dnngp", "hapgam_e"]


def main():
    scen = sys.argv[1:] or SCEN
    d = pd.read_csv(RESULTS / "tables" / "all_metrics.tsv", sep="\t")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for sc in scen:
        sub = d[d["scenario"] == sc]
        if sub.empty:
            print(f"\n== {sc}: no complete cells")
            continue
        metric = "within_r" if sc.startswith("grin2") else "pearson"
        for met in (["pearson", "within_r"] if sc.startswith("grin2") else ["pearson"]):
            tab = sub.pivot_table(index="trait", columns="model", values=met)
            tab = tab.reindex([t for t in TRAITS if t in tab.index])
            cols = [m for m in MODELS if m in tab.columns]
            print(f"\n== {sc} ({met}); folds max {int(sub['folds'].max())}")
            print(tab[cols].rename(columns=MODELS).round(3).to_string())
        tab = sub.pivot_table(index="trait", columns="model", values=metric)
        tab = tab.reindex([t for t in TRAITS if t in tab.index])
        if "hapgam_e" in tab:
            lin = tab[[m for m in LINEAR if m in tab]].max(axis=1)
            best = tab[[m for m in SINGLE if m in tab]].idxmax(axis=1)
            out = pd.DataFrame({"HapGAM-E": tab["hapgam_e"], "best_linear": lin, "diff_lin": tab["hapgam_e"] - lin,
                                "best_single": best.map(MODELS)})
            for b in ["egblup", "lightgbm", "bayesb"]:
                if b in tab:
                    out[f"diff_{b}"] = tab["hapgam_e"] - tab[b]
                    out[f"won_{b}"] = [("%d/%d" % groups_won(sc, t, "hapgam_e", b)) for t in tab.index]
            if "ens_hapgam_e_lightgbm" in tab:
                out["ens"] = tab["ens_hapgam_e_lightgbm"]
                out["ens_minus_best_single"] = tab["ens_hapgam_e_lightgbm"] - tab[[m for m in SINGLE if m in tab]].max(axis=1)
            if "ens_gblup_lightgbm_bayesb" in tab:
                out["ens_base"] = tab["ens_gblup_lightgbm_bayesb"]
            print(out.round(4).to_string())
        f = RESULTS / "tables" / f"compare_{sc}_hapgam_e{'_within' if sc.startswith('grin2') else ''}.tsv"
        if f.exists():
            c = pd.read_csv(f, sep="\t")
            c["sig"] = np.where(c["p_holm"] < 0.05, np.where(c["diff"] > 0, "+", "-"), "")
            summ = c.groupby("baseline").agg(traits=("trait", "size"), better=("sig", lambda s: (s == "+").sum()),
                                             worse=("sig", lambda s: (s == "-").sum()),
                                             diff_min=("diff", "min"), diff_max=("diff", "max"))
            print(f"-- bootstrap ({f.name}): significant (Holm) better/worse per baseline")
            print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
