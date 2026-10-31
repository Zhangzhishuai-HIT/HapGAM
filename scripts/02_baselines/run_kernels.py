"""REML kernel baselines: RKHS (Gaussian), EGBLUP (A + AxA) and haplotype GBLUP.

Kernel hyper-parameters (bandwidth, kernel weight) are chosen by the REML likelihood on the
training data only.
usage: run_kernels.py --model rkhs --scenario dev_region [--traits ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.blocks import block_index, make_scheme  # noqa: E402
from hapgam.gpu import limit_threads, setup_torch_device  # noqa: E402
from hapgam.kernels import additive, gaussian, hadamard_epistatic, haplotype_onehot  # noqa: E402
from hapgam.reml import SpectralREML  # noqa: E402
from hapgam.runner import run_dir, save_fold  # noqa: E402
from hapgam.targets import load_targets, safe  # noqa: E402


def best_by_reml(kernels, y):
    """kernels: list of (label, Ktr, Kte).  Returns (label, prediction, loglik table)."""
    best, table = None, {}
    for label, Ktr, Kte in kernels:
        m = SpectralREML().fit(Ktr, y)
        table[label] = m.loglik
        if best is None or m.loglik > best[0]:
            best = (m.loglik, label, m.predict(Kte).cpu().numpy())
    return best[1], best[2], table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rkhs", "egblup", "hapgblup"], required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--traits", nargs="+", default=data.PRIMARY_TRAITS + data.SECONDARY_TRAITS)
    ap.add_argument("--scheme", default="fixed16")
    args = ap.parse_args()
    limit_threads(4)
    dev = setup_torch_device(cap_mb=16000)
    G = data.grin_dosage()
    idx = None
    if args.model == "hapgblup":
        _, _, idx = block_index(make_scheme(args.scheme, data.grin_snps()))
    for trait in args.traits:
        out = run_dir(args.scenario, args.model, trait)
        for fold, _, _ in data.scenario_partitions(args.scenario):
            if (out / f"{safe(fold)}.json").exists():
                continue
            t0 = time.time()
            td = load_targets(args.scenario, trait, fold)
            y = torch.tensor(td.y_train, device=dev)
            Xtr = torch.tensor(np.asarray(G[td.train_rows]), dtype=torch.float32, device=dev)
            Xte = torch.tensor(np.asarray(G[td.test_rows]), dtype=torch.float32, device=dev)
            mu = Xtr.mean(0, keepdim=True)
            Ztr, Zte = Xtr - mu, Xte - mu
            del Xtr, Xte
            if args.model == "rkhs":
                ks = [(f"h={h}", *gaussian(Ztr, Zte, h)) for h in (0.25, 0.5, 1.0, 2.0, 4.0)]
            elif args.model == "egblup":
                Ka, Kat = additive(Ztr, Zte)
                Kaa, Kaat = hadamard_epistatic(Ka, Kat)
                ks = [(f"w={w}", (1 - w) * Ka + w * Kaa, (1 - w) * Kat + w * Kaat) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
            else:
                rows = np.concatenate([td.train_rows, td.test_rows])
                Xn = np.asarray(G[rows])
                mask = np.zeros(len(rows), bool)
                mask[: len(td.train_rows)] = True
                H = torch.tensor(haplotype_onehot(Xn, idx, train_mask=mask), device=dev)
                H = H - H[: len(td.train_rows)].mean(0, keepdim=True)
                Htr, Hte = H[: len(td.train_rows)], H[len(td.train_rows):]
                Kh, Kht = additive(Htr, Hte)
                Ka, Kat = additive(Ztr, Zte)
                ks = [(f"w={w}", (1 - w) * Ka / Ka.diagonal().mean() + w * Kh / Kh.diagonal().mean(),
                       (1 - w) * Kat / Ka.diagonal().mean() + w * Kht / Kh.diagonal().mean())
                      for w in (0.0, 0.5, 0.75, 1.0)]
            label, pred, table = best_by_reml(ks, y)
            res = save_fold(out, fold, td, pred, {"model": args.model, "chosen": label, "loglik": table,
                                                  "seconds": time.time() - t0})
            print(f"{trait} {fold} {label} r={res['pearson']:.4f} t={time.time()-t0:.1f}s", flush=True)
            del ks
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
