"""DNNGP replica and MLP baselines.

Hyper-parameters are chosen on the shared inner CV (grid below, same folds as every other
model); the final model is a 5-seed ensemble trained on all training accessions for the
inner-CV-selected number of epochs.
usage: run_nets.py --model dnngp --scenario dev_region [--traits ...]
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.metrics import pearson  # noqa: E402
from hapgam.nets import DNNGP, MLP, predict_net, train_net  # noqa: E402
from hapgam.runner import run_dir, save_fold  # noqa: E402
from hapgam.targets import load_targets, safe  # noqa: E402

GRID = {
    "dnngp": {"dropout": [0.1, 0.3, 0.5], "lr": [1e-3, 1e-2], "n_pc": [256]},
    "mlp": {"dropout": [0.1, 0.3, 0.5], "lr": [1e-4, 1e-3], "wd": [1e-4]},
}


def pcs(Ztr, Zother, k):
    """Top-k principal component scores from training genotypes; others projected."""
    U, S, V = torch.linalg.svd(Ztr, full_matrices=False)
    V = V[:k].T
    return Ztr @ V, Zother @ V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dnngp", "mlp"], required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS + data.SECONDARY_TRAITS)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    limit_threads(4)
    dev = setup_torch_device(cap_mb=14000)
    G = data.grin_dosage()
    for trait in args.traits:
        out = run_dir(args.scenario, args.model, trait)
        for fold, _, _ in data.scenario_partitions(args.scenario):
            if (out / f"{safe(fold)}.json").exists():
                continue
            t0 = time.time()
            td = load_targets(args.scenario, trait, fold)
            Xtr = torch.tensor(np.asarray(G[td.train_rows]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(np.asarray(G[td.test_rows]), dtype=torch.float32, device=dev)
            ytr = torch.tensor(td.y_train, dtype=torch.float32, device=dev)
            folds = data.inner_folds(td.train_groups, data.INNER_KIND[args.scenario])
            keys = list(GRID[args.model])
            results = []
            for combo in itertools.product(*GRID[args.model].values()):
                hp = dict(zip(keys, combo))
                rs, eps = [], []
                for va in folds:
                    trm = np.ones(len(ytr), bool)
                    trm[va] = False
                    tri = torch.as_tensor(np.flatnonzero(trm), device=dev)
                    vai = torch.as_tensor(va, device=dev)
                    A, Bv = Xtr[tri], Xtr[vai]
                    mu = A.mean(0, keepdim=True)
                    A, Bv = A - mu, Bv - mu
                    if args.model == "dnngp":
                        A, Bv = pcs(A, Bv, hp["n_pc"])
                        make = lambda: DNNGP(hp["n_pc"], hp["dropout"])  # noqa: E731
                        net, ep, r = train_net(make, A, ytr[tri], Bv, ytr[vai], lr=hp["lr"], device=dev)
                    else:
                        make = lambda: MLP(A.shape[1], dropout=hp["dropout"])  # noqa: E731
                        net, ep, r = train_net(make, A, ytr[tri], Bv, ytr[vai], lr=hp["lr"], wd=hp["wd"], device=dev)
                    rs.append(r)
                    eps.append(ep)
                    del net
                results.append((float(np.mean(rs)), hp, int(np.mean(eps))))
            score, hp, ep = max(results, key=lambda t: t[0])
            mu = Xtr.mean(0, keepdim=True)
            A, T = Xtr - mu, Xte - mu
            if args.model == "dnngp":
                A, T = pcs(A, T, hp["n_pc"])
                make = lambda: DNNGP(hp["n_pc"], hp["dropout"])  # noqa: E731
            else:
                make = lambda: MLP(A.shape[1], dropout=hp["dropout"])  # noqa: E731
            preds = []
            for seed in range(args.seeds):
                net, _, _ = train_net(make, A, ytr, None, None, lr=hp["lr"], wd=hp.get("wd", 0.0),
                                      seed=seed, device=dev, fixed_epochs=max(ep, 1))
                preds.append(predict_net(net, T))
                del net
            seed_r = [pearson(td.y_test[np.isfinite(td.y_test)], p[np.isfinite(td.y_test)]) for p in preds]
            pred = np.mean(preds, 0)
            res = save_fold(out, fold, td, pred, {"model": args.model, "hp": hp, "epochs": ep, "inner_r": score,
                                                  "seed_r": seed_r, "seconds": time.time() - t0})
            print(f"{trait} {fold} {hp} ep={ep} r={res['pearson']:.4f} seeds={np.round(seed_r, 4)} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            del Xtr, Xte
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
