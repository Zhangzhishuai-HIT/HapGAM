"""Hyper-parameters chosen by inner cross-validation for HapGAM models, per scenario.

Reads runs/<scenario>/<model>/<trait>/<fold>.json ('chosen', 'w_epi', 'steps') and writes
results/tables/hapgam_choices_<model>.tsv (one row per fold) and a per-scenario summary
(share of folds with M = 0, i.e. the kernel model, distribution of M and block size, epistatic weight).

usage: hapgam_choices.py [--model hapgam_e]
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS, RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hapgam_e")
    args = ap.parse_args()
    rows = []
    for f in sorted(RUNS.glob(f"*/{args.model}/*/*.json")):
        d = json.loads(f.read_text())
        ch = d.get("chosen")
        if isinstance(ch, str):
            ch = ast.literal_eval(ch)
        if not ch:
            continue
        rows.append({"scenario": d["scenario"], "trait": d["trait"], "fold": d["fold"], "M": int(ch["top_m"]),
                     "block_snps": int(str(ch["scheme"]).replace("fixed", "")) if int(ch["top_m"]) > 0 else None,
                     "weight_decay": ch["weight_decay"] if int(ch["top_m"]) > 0 else None,
                     "w_epi": d.get("w_epi"), "steps": d.get("steps"), "inner_r": ch.get("r"),
                     "inner_r_kernel": d.get("inner_r_gblup")})
    df = pd.DataFrame(rows)
    out = RESULTS / "tables"
    df.to_csv(out / f"hapgam_choices_{args.model}.tsv", sep="\t", index=False)
    g = df.groupby("scenario")
    summ = pd.DataFrame({
        "folds": g.size(),
        "kernel_model_M0": g["M"].apply(lambda m: int((m == 0).sum())),
        "M8": g["M"].apply(lambda m: int((m == 8).sum())), "M16": g["M"].apply(lambda m: int((m == 16).sum())),
        "M32": g["M"].apply(lambda m: int((m == 32).sum())), "M64": g["M"].apply(lambda m: int((m == 64).sum())),
        "blk8": g["block_snps"].apply(lambda b: int((b == 8).sum())),
        "blk16": g["block_snps"].apply(lambda b: int((b == 16).sum())),
        "blk32": g["block_snps"].apply(lambda b: int((b == 32).sum())),
        "w_epi_mean": g["w_epi"].mean(),
        "inner_gain_mean": g["inner_r"].mean() - g["inner_r_kernel"].mean(),   # unrounded: tables round once
    })
    summ.to_csv(out / f"hapgam_choices_{args.model}_summary.tsv", sep="\t")
    pd.set_option("display.width", 200)
    print(summ.to_string())


if __name__ == "__main__":
    main()
