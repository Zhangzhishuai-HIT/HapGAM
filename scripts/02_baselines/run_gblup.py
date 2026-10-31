"""GBLUP (VanRaden kernel, REML) for GRIN scenarios.

usage: run_gblup.py --scenario dev_random --traits maturity_doy height_cm
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.reml import SpectralREML, center_genotypes, vanraden_kernel  # noqa: E402
from hapgam.runner import run_dir, save_fold  # noqa: E402
from hapgam.targets import load_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS + data.SECONDARY_TRAITS)
    ap.add_argument("--model-name", default="gblup")
    args = ap.parse_args()
    limit_threads(4)
    dev = setup_torch_device(cap_mb=12000)
    G = data.grin_dosage()
    for trait in args.traits:
        out = run_dir(args.scenario, args.model_name, trait)
        for fold, tr_acc, te_acc in data.scenario_partitions(args.scenario):
            t0 = time.time()
            td = load_targets(args.scenario, trait, fold)
            Xtr = torch.tensor(np.asarray(G[td.train_rows]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(np.asarray(G[td.test_rows]), dtype=torch.float32, device=dev)
            Ztr, Zte, _ = center_genotypes(Xtr, Xte)
            K = vanraden_kernel(Ztr).double()
            Kte = vanraden_kernel(Zte, Ztr, scale=Ztr.shape[1]).double()
            del Xtr, Xte, Ztr, Zte
            m = SpectralREML().fit(K, torch.tensor(td.y_train, device=dev))
            pred = m.predict(Kte).cpu().numpy()
            res = save_fold(out, fold, td, pred, {"model": args.model_name, "delta": m.delta, "h2": m.h2,
                                                  "seconds": time.time() - t0})
            print(f"{trait} {fold} n={res['n']} r={res['pearson']:.4f} wb={res['within_batch_r']:.4f} "
                  f"h2={m.h2:.3f} t={time.time()-t0:.1f}s", flush=True)
            del K, Kte, m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
