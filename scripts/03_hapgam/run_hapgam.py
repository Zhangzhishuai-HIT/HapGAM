"""HapGAM with inner-CV hyper-parameter selection (final protocol).

For each outer fold:
  1. inner CV (shared folds; random or grouped to match the outer scenario).  In each inner
     fold, 10% of the inner-training accessions are held out for early stopping, every grid
     configuration is fitted, and Pearson r is measured on the inner validation fold;
  2. the configuration with the best mean inner r is selected (M = 0, i.e. GBLUP, is part of
     the grid, so the procedure can fall back to GBLUP);
  3. the selected configuration is fitted on 90% of the outer training set with early stopping,
     then refitted on 100% for the selected number of steps, and the test fold is predicted.
Outputs: runs/<scenario>/<model_name>/<trait>/<fold>.{json,pred.tsv,tuning.tsv,blocks.tsv}
usage: run_hapgam.py --scenario dev_region --traits maturity_doy [--folds China]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.blocks import block_index, make_scheme  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.metrics import pearson  # noqa: E402
from hapgam.model import Context, HapGAM, HapGAMConfig  # noqa: E402
from hapgam.runner import run_dir, save_fold  # noqa: E402
from hapgam.targets import load_targets, safe  # noqa: E402

GRID = {"scheme": ["fixed8", "fixed16", "fixed32"], "top_m": [8, 16, 32, 64], "weight_decay": [1e-3, 1e-2]}
BASE = dict(hidden=8, lr=5e-4, steps=4000, eval_every=10, patience=20)


def grid_configs(grid):
    cfgs = [{"scheme": "fixed16", "top_m": 0, "weight_decay": 0.0}]          # GBLUP
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        cfgs.append(dict(zip(keys, combo)))
    return cfgs


def es_split(n, rng, frac=0.1):
    es = np.zeros(n, bool)
    es[rng.choice(n, max(int(frac * n), 50), replace=False)] = True
    return es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS + data.SECONDARY_TRAITS)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--model-name", default="hapgam")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    limit_threads(4)
    dev = setup_torch_device(cap_mb=14000)
    G = data.grin_dosage()
    snps = data.grin_snps()
    schemes = {name: block_index(make_scheme(name, snps))[2] for name in GRID["scheme"]}
    configs = grid_configs(GRID)
    for trait in args.traits:
        out = run_dir(args.scenario, args.model_name, trait)
        for fold, _, _ in data.scenario_partitions(args.scenario):
            if args.folds and fold not in args.folds:
                continue
            if (out / f"{safe(fold)}.json").exists():
                continue
            t0 = time.time()
            rng = np.random.default_rng(args.seed)
            td = load_targets(args.scenario, trait, fold)
            X = torch.tensor(np.asarray(G[td.train_rows]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(np.asarray(G[td.test_rows]), dtype=torch.float32, device=dev)
            y = td.y_train
            inner = data.inner_folds(td.train_groups, data.INNER_KIND[args.scenario])
            rows = []
            for f, va in enumerate(inner):
                tr = np.setdiff1d(np.arange(len(y)), va)
                es = es_split(len(tr), rng)
                fit_i, es_i = tr[~es], tr[es]
                ti = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
                ctx = Context(X[ti(fit_i)], y[fit_i], dev)
                for c in configs:
                    cfg = HapGAMConfig(**BASE, top_m=c["top_m"], weight_decay=c["weight_decay"], seed=args.seed)
                    m = HapGAM(schemes[c["scheme"]], cfg, dev, scheme_key=c["scheme"]).fit(
                        X[ti(fit_i)], y[fit_i], X[ti(es_i)], y[es_i], ctx=ctx)
                    r = pearson(y[va], m.predict(X[ti(va)]))
                    rows.append({**c, "inner_fold": f, "r": r, "best_step": m.best_step})
                    del m
                del ctx
                torch.cuda.empty_cache()
            tune = pd.DataFrame(rows)
            tune.to_csv(out / f"{safe(fold)}.tuning.tsv", sep="\t", index=False)
            agg = tune.groupby(["scheme", "top_m", "weight_decay"], as_index=False).agg(r=("r", "mean"))
            agg = agg.sort_values(["r", "top_m"], ascending=[False, True])
            best = agg.iloc[0].to_dict()
            gb = agg[agg["top_m"] == 0]["r"].iloc[0]
            # final fit: early stopping on 10% of the outer training set, then refit on 100%
            cfg = HapGAMConfig(**BASE, top_m=int(best["top_m"]), weight_decay=float(best["weight_decay"]),
                               seed=args.seed)
            idx = schemes[best["scheme"]]
            es = es_split(len(y), rng)
            ti = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
            steps = cfg.steps
            if cfg.top_m > 0:
                m1 = HapGAM(idx, cfg, dev, scheme_key=best["scheme"]).fit(
                    X[ti(np.flatnonzero(~es))], y[~es], X[ti(np.flatnonzero(es))], y[es])
                steps = max(m1.best_step, 1)
                del m1
            cfg2 = HapGAMConfig(**{**cfg.__dict__, "steps": steps})
            m = HapGAM(idx, cfg2, dev, scheme_key=best["scheme"]).fit(X, y)
            pred, Fb, lin = m.predict(Xte, return_parts=True)
            Kte = ((Xte - m.mu_x) @ m.Zc_train.T / m.Zc_train.shape[1]).double()
            pg = m.null.predict(Kte).cpu().numpy() * m.y_sd + m.y_mean
            ok = np.isfinite(td.y_test)
            blocks = pd.DataFrame({
                "rank": np.arange(len(m.selected.cpu().numpy())),
                "block": m.selected.cpu().numpy(),
                "screen_score": m.block_scores[m.selected].cpu().numpy(),
                "test_contrib_var": Fb.var(0) if Fb.shape[1] else np.zeros(0)})
            blocks.to_csv(out / f"{safe(fold)}.blocks.tsv", sep="\t", index=False)
            info = {"model": args.model_name, "chosen": best, "inner_r_gblup": float(gb), "steps": int(steps),
                    "r_gblup_same_fit": pearson(td.y_test[ok], pg[ok]), "seconds": time.time() - t0}
            res = save_fold(out, fold, td, pred, info)
            print(f"{trait} {fold} chosen={best} inner_gblup={gb:.4f} r={res['pearson']:.4f} "
                  f"r_gblup={info['r_gblup_same_fit']:.4f} t={time.time()-t0:.0f}s", flush=True)
            del X, Xte, m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
