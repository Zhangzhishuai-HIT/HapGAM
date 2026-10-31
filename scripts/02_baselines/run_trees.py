"""LightGBM / XGBoost baselines with Optuna tuning on the shared inner CV.

usage: run_trees.py --model lightgbm --scenario dev_region --trait maturity_doy [--fold China]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.runner import run_dir, save_fold  # noqa: E402
from hapgam.targets import load_targets, safe  # noqa: E402
from hapgam.trees import fit_lightgbm, fit_xgboost, tune_lightgbm, tune_xgboost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lightgbm", "xgboost"], required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    G = data.grin_dosage()
    out = run_dir(args.scenario, args.model, args.trait)
    for fold, _, _ in data.scenario_partitions(args.scenario):
        if args.folds and fold not in args.folds:
            continue
        if (out / f"{safe(fold)}.json").exists():
            continue
        t0 = time.time()
        td = load_targets(args.scenario, args.trait, fold)
        X = np.asarray(G[td.train_rows])
        Xte = np.asarray(G[td.test_rows])
        folds = data.inner_folds(td.train_groups, data.INNER_KIND[args.scenario])
        if args.model == "lightgbm":
            params, rounds, study = tune_lightgbm(X, td.y_train, folds, n_trials=args.trials,
                                                  n_threads=args.threads)
            pred = fit_lightgbm(X, td.y_train, params, rounds).predict(Xte)
        else:
            params, rounds, study = tune_xgboost(X, td.y_train, folds, n_trials=args.trials,
                                                 device=args.device, nthread=args.threads)
            import xgboost as xgb
            pred = fit_xgboost(X, td.y_train, params, rounds).predict(xgb.DMatrix(Xte))
        info = {"model": args.model, "params": params, "rounds": rounds, "inner_r": study.best_value,
                "trials": len(study.trials), "seconds": time.time() - t0}
        res = save_fold(out, fold, td, pred, info)
        print(json.dumps({k: res[k] for k in ("fold", "n", "pearson", "within_batch_r", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
