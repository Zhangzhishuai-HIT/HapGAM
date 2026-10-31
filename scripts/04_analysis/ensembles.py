"""Equal-weight ensembles from frozen model predictions (pre-specified in PREREGISTRATION, record 3).

For each fold where every member has predictions, member predictions are z-scored within the
test fold (prediction statistics only, no labels) and averaged with equal weights.  Outputs are
written like any other model: runs/<scenario>/<name>/<trait>/<fold>.{pred.tsv,json}.

usage: ensembles.py --scenarios dev_region C1 C2 ... [--name ens_hapgam_e_lightgbm --members hapgam_e lightgbm]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.metrics import pearson, rmse, spearman, top_coincidence, within_group_r  # noqa: E402
from hapgam.paths import RUNS  # noqa: E402

DEFAULTS = {"ens_hapgam_e_lightgbm": ["hapgam_e", "lightgbm"],
            "ens_gblup_lightgbm_bayesb": ["gblup", "lightgbm", "bayesb"]}


def build(scenario, name, members):
    base = RUNS / scenario
    traits = sorted({p.name for m in members for p in (base / m).glob("*") if p.is_dir()})
    n = 0
    for trait in traits:
        dirs = [base / m / trait for m in members]
        folds = set.intersection(*[{f.name[: -len(".pred.tsv")] for f in d.glob("*.pred.tsv")} for d in dirs])
        for fold in sorted(folds):
            tabs = [pd.read_csv(d / f"{fold}.pred.tsv", sep="\t", keep_default_na=False,
                                na_values={"y": [""], "pred": [""]}).set_index("id") for d in dirs]
            ids = tabs[0].index
            for t in tabs[1:]:
                ids = ids.intersection(t.index)
            z = [(t.loc[ids, "pred"] - t.loc[ids, "pred"].mean()) / t.loc[ids, "pred"].std() for t in tabs]
            pred = np.mean(np.vstack([v.to_numpy(float) for v in z]), 0)
            y = tabs[0].loc[ids, "y"].to_numpy(float)
            grp = tabs[0].loc[ids, "group"].to_numpy()
            out = base / name / trait
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": ids, "group": grp, "y": y, "pred": pred}).to_csv(out / f"{fold}.pred.tsv", sep="\t",
                                                                               index=False)
            ok = np.isfinite(y)
            # fold label as the members record it (file names replace '/' and ' ' by '_', e.g. 'S/SE Asia')
            mj = dirs[0] / f"{fold}.json"
            label = json.loads(mj.read_text()).get("fold", fold) if mj.exists() else fold
            m = {"n": int(ok.sum()), "pearson": pearson(y[ok], pred[ok]), "spearman": spearman(y[ok], pred[ok]),
                 "rmse": float("nan"), "top20": top_coincidence(y[ok], pred[ok]),
                 "within_group_r": within_group_r(y, pred, grp), "fold": label, "model": name, "scenario": scenario,
                 "trait": trait, "members": members, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            (out / f"{fold}.json").write_text(json.dumps(m, indent=1))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--members", nargs="*", default=None)
    args = ap.parse_args()
    ens = {args.name: args.members} if args.name else DEFAULTS
    for sc in args.scenarios:
        for name, members in ens.items():
            print(sc, name, build(sc, name, members), "folds")


if __name__ == "__main__":
    main()
