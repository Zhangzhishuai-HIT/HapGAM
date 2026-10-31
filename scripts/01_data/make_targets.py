"""Compute and cache fold-internal adjusted targets for GRIN scenarios (GPU).

usage: make_targets.py --scenarios dev_random dev_region [--traits ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.targets import build_targets, target_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS + data.SECONDARY_TRAITS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gpu-cap", type=int, default=7000, help="per-process GPU memory cap (MiB)")
    args = ap.parse_args()
    import os
    import torch

    limit_threads(int(os.environ.get("OMP_NUM_THREADS", "4")) if not torch.cuda.is_available() else 4)
    dev = setup_torch_device(cap_mb=args.gpu_cap)
    for sc in args.scenarios:
        for trait in args.traits:
            for fold, tr, te in data.scenario_partitions(sc):
                p = target_path(sc, trait, fold)
                if p.exists() and not args.overwrite:
                    continue
                t0 = time.time()
                try:
                    out = build_targets(trait, tr, te, dev)
                except torch.OutOfMemoryError:
                    print(f"{sc} {trait} {fold}: GPU memory cap reached; skipped (rerun on CPU)", flush=True)
                    torch.cuda.empty_cache()
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, **out)
                print(f"{sc} {trait} {fold} ntr={len(out['y_train'])} nte={len(out['y_test'])} "
                      f"{out['info'][0]} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
