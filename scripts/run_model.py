"""Generic runner: any model x any scenario (GRIN folds, SoyNAM, GRIN->SoyNAM, GRIN->NUST).

usage: run_model.py --model hapgam --scenario C2 [--traits ...] [--folds ...] [--threads 16]
Outputs runs/<scenario>/<model>/<trait>/<fold>.{json,pred.tsv[,tuning.tsv,blocks.tsv,parts.npz]}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hapgam import api, scenarios  # noqa: E402
from hapgam.metrics import auc, pearson, rmse, spearman, top_coincidence, within_batch_r, within_group_r  # noqa: E402
from hapgam.runner import code_hash, run_dir  # noqa: E402
from hapgam.targets import safe  # noqa: E402

GPU_MODELS = {"gblup", "hapgam", "hapgam_linear", "hapgam_e", "hapgam_e_linear", "rkhs", "egblup", "hapgblup",
              "dnngp", "mlp"}


def evaluate(task, pred):
    y = np.asarray(task.yte, float)
    ok = np.isfinite(y)
    out = {"n": int(ok.sum()), "pearson": pearson(y[ok], pred[ok]), "spearman": spearman(y[ok], pred[ok]),
           "rmse": rmse(y[ok], pred[ok]), "top20": top_coincidence(y[ok], pred[ok])}
    out["within_group_r"] = within_group_r(y, pred, task.test_groups)
    if set(np.unique(y[ok])) <= {0.0, 1.0}:
        out["auc"] = auc(y[ok], pred[ok])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the model directory name")
    ap.add_argument("--gpu-cap", type=int, default=14000, help="per-process GPU memory cap (MiB)")
    args = ap.parse_args()
    dev = None
    if args.model in GPU_MODELS:
        import torch

        from hapgam.gpu import limit_threads, setup_torch_device

        limit_threads(args.threads if not torch.cuda.is_available() else 4)
        dev = setup_torch_device(cap_mb=args.gpu_cap)
        print(f"device={dev} threads={torch.get_num_threads()}", flush=True)
    traits = args.traits or scenarios.traits_for(args.scenario)
    name = args.model + args.tag
    for trait in traits:
        out = run_dir(args.scenario, name, trait)
        for fold in scenarios.folds(args.scenario, trait):
            if args.folds and fold not in args.folds:
                continue
            if (out / f"{safe(fold)}.json").exists():
                continue
            t0 = time.time()
            task = scenarios.get_task(args.scenario, trait, fold)
            extra = {}
            if args.model == "gblup":
                pred, info = api.gblup(task.Xtr, task.ytr, task.Xte, dev)
            elif args.model in ("rkhs", "egblup", "hapgblup"):
                idx = api.schemes_for(task.chrom, ("fixed16",))["fixed16"] if args.model == "hapgblup" else None
                pred, info = api.kernel_model(args.model, task.Xtr, task.ytr, task.Xte, dev, block_idx=idx)
            elif args.model.startswith("hapgam"):
                pred, info = api.hapgam(task.Xtr, task.ytr, task.groups, task.inner_kind, task.Xte, dev,
                                        api.schemes_for(task.chrom), seed=args.seed,
                                        linear=args.model.endswith("linear"),
                                        backbone="AE" if args.model.startswith("hapgam_e") else "A")
                info["tuning"].to_csv(out / f"{safe(fold)}.tuning.tsv", sep="\t", index=False)
                sel = info["selected_blocks"]
                pd.DataFrame({"rank": np.arange(len(sel)), "block": sel,
                              "screen_score": info["block_scores"][sel] if len(sel) else [],
                              "test_contrib_var": info["test_block_contrib"].var(0) if len(sel) else []}
                             ).to_csv(out / f"{safe(fold)}.blocks.tsv", sep="\t", index=False)
                np.savez_compressed(out / f"{safe(fold)}.parts.npz", block_contrib=info["test_block_contrib"],
                                    linear=info["test_linear"], scheme_idx=info["scheme_idx"],
                                    selected=sel, block_scores=info["block_scores"], snp_cols=task.cols,
                                    pred_gblup=info["pred_gblup_same_fit"])
                extra["r_gblup_same_fit"] = pearson(task.yte[np.isfinite(task.yte)],
                                                    info["pred_gblup_same_fit"][np.isfinite(task.yte)])
                info = {k: info[k] for k in ("chosen", "steps", "inner_r_gblup", "w_epi")}
            elif args.model in ("lightgbm", "xgboost"):
                pred, info = api.trees(args.model, task.Xtr, task.ytr, task.groups, task.inner_kind, task.Xte,
                                       threads=args.threads, trials=args.trials)
            elif args.model in ("dnngp", "mlp"):
                pred, info = api.nets(args.model, task.Xtr, task.ytr, task.groups, task.inner_kind, task.Xte, dev)
                info["seed_r"] = [pearson(task.yte[np.isfinite(task.yte)], p[np.isfinite(task.yte)])
                                  for p in info.pop("seed_preds")]
            else:
                raise ValueError(args.model)
            pred = np.asarray(pred, float)
            pd.DataFrame({"id": task.test_ids, "group": task.test_groups, "y": task.yte, "pred": pred}).to_csv(
                out / f"{safe(fold)}.pred.tsv", sep="\t", index=False)
            m = evaluate(task, pred)
            if task.records is not None:
                from hapgam import data as _d
                acc = _d.grin_accessions()
                rows = acc.set_index("accession").loc[task.test_ids, "row"].to_numpy()
                m["within_batch_r"] = within_batch_r(task.records, dict(zip(rows, pred)))
            m.update({"fold": fold, "model": name, "scenario": args.scenario, "trait": trait,
                      "n_train": int(len(task.ytr)), "code": code_hash(), "seconds": time.time() - t0,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"), **task.meta, **extra, **info})
            (out / f"{safe(fold)}.json").write_text(json.dumps(m, indent=1, default=lambda o: o.tolist()
                                                               if hasattr(o, "tolist") else str(o)))
            print(f"{name} {args.scenario} {trait} {fold} r={m['pearson']:.4f} wg={m['within_group_r']:.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
