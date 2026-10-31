"""Zero-shot transfer (C6, C7): overlap with the GRIN training set and per-trial results.

Eleven SoyNAM founder parents are GRIN accessions (matched by accession number or cultivar name; families
9, 10, 15, 22, 40, 41, 42, 48, 50, 54, 64), and ten NUST lines (13 line x trial entries) are released cultivars that
are GRIN accessions.  For each model, the mean within-family (C6) or within-trial (C7) correlation is recomputed
after excluding them.  For C7, the number of the six maturity-group trials in which HapGAM-E was more accurate than
each baseline is also counted (the bootstrap over only six trials is approximate).
Writes results/tables/transfer_sensitivity.tsv and results/tables/nust_trials_won.tsv.

usage: transfer_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hapgam.metrics import within_group_r, pearson
MODELS = ["gblup", "egblup", "rkhs", "bayesb", "lightgbm", "hapgam_e", "ens_hapgam_e_lightgbm"]
FAM_EXCL = {"9", "10", "15", "22", "40", "41", "42", "48", "50", "54", "64"}
NUST_EXCL = {"Apex_MGIII", "Lambert_MG0", "Lambert_MGI", "LD00_3309_MGIV", "LN97_15076_MGIV", "Macon_MGIII",
             "Macon_MGIV", "MN0902CN_MG0", "Pana_MGIII", "Parker_MG0", "Parker_MGI", "Strong_MGIV", "Surge_MG0"}
rows, wins = [], []
for sc, excl, traits in [("grin2soynam", FAM_EXCL, ["maturity_doy", "flowering_doy", "height_cm", "seed_weight_g", "yield_mg_ha", "protein_pct", "oil_pct", "lodging_score"]),
                         ("grin2nust", NUST_EXCL, ["maturity_doy", "height_cm", "seed_weight_g", "yield_mg_ha", "protein_pct", "oil_pct", "lodging_score"])]:
    for t in traits:
        res = {}
        per = {}
        for m in MODELS:
            p = pd.read_csv(ROOT / f"runs/{sc}/{m}/{t}/all.pred.tsv", sep="\t", keep_default_na=False,
                            na_values={"y": [""], "pred": [""]})
            p["group"] = p["group"].astype(str)
            keep = ~(p["group"].isin(excl) if sc == "grin2soynam" else p["id"].astype(str).isin(excl))
            y, pr, g = p["y"].to_numpy(float), p["pred"].to_numpy(float), p["group"].to_numpy()
            res[(m, "all")] = within_group_r(y, pr, g)
            res[(m, "excl")] = within_group_r(y[keep], pr[keep], g[keep])
            if sc == "grin2nust":
                ok = np.isfinite(y)
                per[m] = {gg: pearson(y[ok & (g == gg)], pr[ok & (g == gg)]) for gg in sorted(set(g[ok]))}
        for m in MODELS:
            rows.append({"scenario": sc, "trait": t, "model": m, "within_all": res[(m, "all")], "within_excl": res[(m, "excl")]})
        if sc == "grin2nust":
            for b in ["gblup", "egblup", "rkhs", "bayesb", "lightgbm"]:
                k = sum(per["hapgam_e"][gg] > per[b][gg] for gg in per["hapgam_e"])
                wins.append({"trait": t, "baseline": b, "trials_won": f"{k}/{len(per['hapgam_e'])}"})
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
w = d.pivot_table(index=["scenario", "trait"], columns="model", values="within_excl").round(3)
print("within-group r after exclusion\n", w[MODELS].to_string())
w2 = d.pivot_table(index=["scenario", "trait"], columns="model", values="within_all").round(3)
diff = pd.DataFrame({b: (w["hapgam_e"] - w[b]).round(3) for b in ["gblup", "egblup", "lightgbm"]})
diff_all = pd.DataFrame({b: (w2["hapgam_e"] - w2[b]).round(3) for b in ["gblup", "egblup", "lightgbm"]})
print("\nHapGAM-E minus baseline: after exclusion | all\n", pd.concat([diff, diff_all], axis=1, keys=["excl", "all"]).to_string())
print("\nC7 trials won by HapGAM-E\n", pd.DataFrame(wins).pivot(index="trait", columns="baseline", values="trials_won").to_string())
d.to_csv(ROOT / "results/tables/transfer_sensitivity.tsv", sep="\t", index=False, float_format="%.6f")
pd.DataFrame(wins).to_csv(ROOT / "results/tables/nust_trials_won.tsv", sep="\t", index=False)
