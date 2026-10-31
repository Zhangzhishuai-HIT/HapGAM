"""Pre-registered Go/No-Go check on the development region holdout (dev_region).

For each primary trait the reference baseline is whichever of GBLUP and LightGBM has the higher
mean fold-level Pearson r.  The statistic is the mean over the four primary traits of
(HapGAM - reference), with a bootstrap CI obtained by resampling test accessions within folds
(2,000 replicates, the same resamples for every trait).  Go if the mean gain >= 0.01 or the CI
excludes zero (section 8 of docs/PREREGISTRATION.md).

usage: go_nogo.py --focal hapgam_e
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.data import PRIMARY_TRAITS  # noqa: E402
from hapgam.paths import RESULTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import _r, load_preds  # noqa: E402

B = 2000


def fold_arrays(scenario, models, trait):
    preds = {m: load_preds(scenario, m, trait) for m in models}
    folds = sorted(set.intersection(*[set(p) for p in preds.values()]))
    out = []
    for f in folds:
        base = preds[models[0]][f].set_index("id")
        ids = base.index
        for m in models[1:]:
            ids = ids.intersection(preds[m][f].set_index("id").index)
        y = base.loc[ids, "y"].to_numpy(float)
        ok = np.isfinite(y)
        out.append((y[ok], {m: preds[m][f].set_index("id").loc[ids, "pred"].to_numpy(float)[ok] for m in models}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="hapgam_e")
    ap.add_argument("--scenario", default="dev_region")
    ap.add_argument("--extra", nargs="*", default=["egblup", "bayesb", "mlp", "dnngp", "rkhs"])
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    rows, gains_point, gains_boot = [], [], np.zeros(B)
    for trait in PRIMARY_TRAITS:
        fa = fold_arrays(args.scenario, [args.focal, "gblup", "lightgbm"], trait)
        if not fa:
            print(f"{trait}: missing folds for focal/gblup/lightgbm")
            return
        mean_r = {m: np.mean([_r(y, p[m]) for y, p in fa]) for m in (args.focal, "gblup", "lightgbm")}
        ref = "gblup" if mean_r["gblup"] >= mean_r["lightgbm"] else "lightgbm"
        g = mean_r[args.focal] - mean_r[ref]
        gains_point.append(g)
        bs = np.empty(B)
        for i in range(B):
            vals = []
            for y, p in fa:
                idx = rng.integers(0, len(y), len(y))
                vals.append(_r(y[idx], p[args.focal][idx]) - _r(y[idx], p[ref][idx]))
            bs[i] = np.mean(vals)
        gains_boot += bs / len(PRIMARY_TRAITS)
        row = {"trait": trait, "folds": len(fa), "focal": mean_r[args.focal], "gblup": mean_r["gblup"],
               "lightgbm": mean_r["lightgbm"], "reference": ref, "gain": g,
               "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))}
        for m in args.extra:
            fx = fold_arrays(args.scenario, [args.focal, m], trait)
            if len(fx) == len(fa):
                row[m] = float(np.mean([_r(y, p[m]) for y, p in fx]))
        rows.append(row)
    df = pd.DataFrame(rows)
    mean_gain = float(np.mean(gains_point))
    lo, hi = np.percentile(gains_boot, [2.5, 97.5])
    decision = "GO" if (mean_gain >= 0.01 or lo > 0) else "NO-GO"
    summary = {"focal": args.focal, "mean_gain_vs_max_gblup_lightgbm": mean_gain, "ci": [float(lo), float(hi)],
               "decision": decision}
    out = RESULTS / "tables" / f"go_nogo_{args.scenario}_{args.focal}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
