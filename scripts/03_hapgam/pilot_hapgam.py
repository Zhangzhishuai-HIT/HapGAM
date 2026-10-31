"""Development pilot: HapGAM vs GBLUP on dev folds (dev accessions only).

For each fold, 10% of the training accessions form an early-stopping validation set; the
model is then refitted on all training accessions for the selected number of steps.
usage: pilot_hapgam.py --scenario dev_region --traits maturity_doy --scheme fixed16 --top-m 8 32 128
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.blocks import block_index, make_scheme  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.metrics import pearson  # noqa: E402
from hapgam.model import HapGAM, HapGAMConfig  # noqa: E402
from hapgam.paths import RUNS  # noqa: E402
from hapgam.targets import load_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dev_region")
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS)
    ap.add_argument("--scheme", default="fixed16")
    ap.add_argument("--top-m", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--ols", action="store_true")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    limit_threads(4)
    dev = setup_torch_device(cap_mb=14000)
    G = data.grin_dosage()
    snps = data.grin_snps()
    scheme = make_scheme(args.scheme, snps, G if args.scheme.startswith("ld") else None)
    B, kmax, idx = block_index(scheme)
    print(f"scheme {args.scheme}: B={B} kmax={kmax}", flush=True)
    out = RUNS / "pilot" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for trait in args.traits:
        for fold, _, _ in data.scenario_partitions(args.scenario):
            if args.folds and fold not in args.folds:
                continue
            td = load_targets(args.scenario, trait, fold)
            n = len(td.train_rows)
            val = np.zeros(n, bool)
            val[rng.choice(n, n // 10, replace=False)] = True
            Xall = torch.tensor(np.asarray(G[td.train_rows]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(np.asarray(G[td.test_rows]), dtype=torch.float32, device=dev)
            ok = np.isfinite(td.y_test)
            for M in args.top_m:
                t0 = time.time()
                cfg = HapGAMConfig(top_m=M, hidden=args.hidden, lr=args.lr, weight_decay=args.wd,
                                   steps=args.steps, gls=not args.ols, eval_every=args.eval_every,
                                   patience=args.patience)
                m = HapGAM(idx, cfg, dev).fit(Xall[~val], td.y_train[~val], Xall[val], td.y_train[val])
                cfg2 = HapGAMConfig(**{**cfg.__dict__, "steps": max(m.best_step, 1)})
                m2 = HapGAM(idx, cfg2, dev).fit(Xall, td.y_train)
                p = m2.predict(Xte)
                Kte = ((Xte - m2.mu_x) @ m2.Zc_train.T / m2.Zc_train.shape[1]).double()
                pg = m2.null.predict(Kte).cpu().numpy() * m2.y_sd + m2.y_mean
                res = {"trait": trait, "fold": fold, "top_m": M, "lr": args.lr, "scheme": args.scheme, "hidden": args.hidden,
                       "wd": args.wd, "gls": not args.ols, "best_step": m.best_step, "val_r": m.best_val,
                       "r_hapgam": pearson(td.y_test[ok], p[ok]), "r_gblup": pearson(td.y_test[ok], pg[ok]),
                       "sec": round(time.time() - t0, 1)}
                res["gain"] = res["r_hapgam"] - res["r_gblup"]
                print(json.dumps(res), flush=True)
                with open(out / "results.jsonl", "a") as fh:
                    fh.write(json.dumps(res) + "\n")
                del m, m2
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
