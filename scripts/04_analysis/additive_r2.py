"""How additive are the learned block functions?

For every block selected by HapGAM in one fold, the share of the variance of the block's contribution to
the prediction that a linear function of the block's SNP dosages explains (R^2_additive), together with the
block's share of the summed block-contribution variance and its number of distinct haplotypes.
Reads the stored fold outputs only.

usage: additive_r2.py --scenario C2 --model hapgam_e --fold lockbox_japan
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_haplotype_effects import additive_r2, load_fold  # noqa: E402

TRAITS = ["maturity_doy", "flowering_doy", "height_cm", "seed_weight_g", "yield_mg_ha", "protein_pct", "oil_pct",
          "lodging_score", "maturity_group"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="C2")
    ap.add_argument("--model", default="hapgam_e")
    ap.add_argument("--fold", default="lockbox_japan")
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    args = ap.parse_args()
    rows = []
    for t in args.traits:
        z, G, _, _, _ = load_fold(args.scenario, args.model, t, args.fold)
        cols, contrib = z["snp_cols"], z["block_contrib"]
        var = contrib.var(0)
        share = var / var.sum()
        for r, b in enumerate(z["selected"]):
            mem = z["scheme_idx"][b]
            mem = cols[mem[mem >= 0]]
            Gb = G[:, mem].astype(float)
            n_hap = len({tuple(x) for x in Gb.astype(int)})
            rows.append((t, int(b), share[r], additive_r2(Gb - Gb.mean(0), contrib[:, r]), n_hap, len(mem)))
    df = pd.DataFrame(rows, columns=["trait", "block", "share", "r2", "n_hap", "n_snp"])
    df.to_csv(RESULTS / "tables" / f"additive_r2_{args.scenario}_{args.model}.tsv", sep="\t", index=False)
    w = np.average(df.r2, weights=df.share)
    print(f"blocks {len(df)}: median R2 {df.r2.median():.3f} (IQR {df.r2.quantile(.25):.3f}-{df.r2.quantile(.75):.3f}); "
          f"variance-weighted mean {w:.3f}; share of blocks with R2 > 0.9: {(df.r2 > 0.9).mean():.2f}")
    print(df.groupby("trait").apply(lambda g: pd.Series({"n": len(g), "median_r2": g.r2.median(),
                                                         "wmean_r2": np.average(g.r2, weights=g.share),
                                                         "min_r2": g.r2.min()})).round(3).to_string())


if __name__ == "__main__":
    main()
