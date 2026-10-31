"""Write per-fold inputs for the BGLR R script and collect its predictions (any scenario).

Each fold directory receives Xtr.int8 / Xte.int8 (row-major int8 genotypes from
scenarios.get_task), dims.txt (n_train n_test p), y_train.txt and test_ids.txt.
Fold inputs and BGLR outputs live in work/bglr/<scenario>/<trait>/<fold>/ (never under runs/).

usage: bglr_prepare.py prepare --scenario C2 [--traits ...]
       bglr_prepare.py collect --scenario C2 [--traits ...] --model BayesB
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import scenarios  # noqa: E402
from hapgam.metrics import pearson, rmse, spearman, top_coincidence, within_group_r  # noqa: E402
from hapgam.paths import ROOT  # noqa: E402
from hapgam.runner import code_hash, run_dir  # noqa: E402
from hapgam.targets import load_targets, safe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "collect"])
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--model", default="BayesB")
    args = ap.parse_args()
    base = ROOT / "work" / "bglr" / args.scenario
    traits = args.traits or scenarios.traits_for(args.scenario)
    for trait in traits:
        for fold in scenarios.folds(args.scenario, trait):
            d = base / trait / safe(fold)
            if args.action == "prepare":
                if (d / "dims.txt").exists() or (d / "train_rows.txt").exists():
                    print(d)
                    continue
                d.mkdir(parents=True, exist_ok=True)
                if args.scenario in scenarios.GRIN_SCENARIOS:
                    # GRIN folds reference rows of the shared genotype file (no per-fold copies)
                    td = load_targets(args.scenario, trait, fold)
                    task = scenarios.get_task(args.scenario, trait, fold)
                    np.savetxt(d / "train_rows.txt", td.train_rows, fmt="%d")
                    np.savetxt(d / "test_rows.txt", td.test_rows, fmt="%d")
                    np.savetxt(d / "y_train.txt", td.y_train, fmt="%.6f")
                    pd.DataFrame({"id": task.test_ids, "group": task.test_groups, "y": task.yte}).to_csv(
                        d / "test.tsv", sep="\t", index=False)
                    print(d)
                    continue
                task = scenarios.get_task(args.scenario, trait, fold)
                np.ascontiguousarray(task.Xtr, dtype=np.int8).tofile(d / "Xtr.int8")
                np.ascontiguousarray(task.Xte, dtype=np.int8).tofile(d / "Xte.int8")
                (d / "dims.txt").write_text(f"{task.Xtr.shape[0]} {task.Xte.shape[0]} {task.Xtr.shape[1]}\n")
                np.savetxt(d / "y_train.txt", task.ytr, fmt="%.6f")
                pd.DataFrame({"id": task.test_ids, "group": task.test_groups, "y": task.yte}).to_csv(
                    d / "test.tsv", sep="\t", index=False)
                print(d)
            else:
                f = d / f"{args.model}.pred.txt"
                if not f.exists():
                    print("missing", f)
                    continue
                pred = np.loadtxt(f)
                if (d / "test.tsv").exists():
                    te = pd.read_csv(d / "test.tsv", sep="\t", keep_default_na=False, na_values={"y": [""]})
                else:   # older GRIN folds prepared with row indices
                    task = scenarios.get_task(args.scenario, trait, fold)
                    te = pd.DataFrame({"id": task.test_ids, "group": task.test_groups, "y": task.yte})
                y = te["y"].to_numpy(float)
                ok = np.isfinite(y)
                out = run_dir(args.scenario, args.model.lower(), trait)
                te.assign(pred=pred).to_csv(out / f"{safe(fold)}.pred.tsv", sep="\t", index=False)
                m = {"n": int(ok.sum()), "pearson": pearson(y[ok], pred[ok]), "spearman": spearman(y[ok], pred[ok]),
                     "rmse": rmse(y[ok], pred[ok]), "top20": top_coincidence(y[ok], pred[ok]),
                     "within_group_r": within_group_r(y, pred, te["group"].to_numpy()), "fold": fold,
                     "model": args.model.lower(), "scenario": args.scenario, "trait": trait, "code": code_hash(),
                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                (out / f"{safe(fold)}.json").write_text(json.dumps(m, indent=1))
                print(trait, fold, round(m["pearson"], 4))


if __name__ == "__main__":
    main()
