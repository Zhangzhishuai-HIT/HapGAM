"""Map HapGAM-selected blocks to the genome and relate them to known soybean genes.

For every fold of runs/<scenario>/<model>/<trait>/ the selected blocks (parts.npz) are mapped
to Wm82.a2 intervals.  Each block gets its test-set contribution variance (importance) and
the distance to the nearest trait-relevant known gene (data/processed/known_genes.tsv).
Enrichment: share of selected blocks within `--window` bp of a relevant gene compared with
10,000 random draws of the same number of blocks from the same block scheme.

usage: block_maps.py --scenario C3 --model hapgam_e --traits maturity_doy height_cm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.paths import PROCESSED, RESULTS, RUNS  # noqa: E402


# positive-control traits -> descriptor names used in known_genes.tsv
PC_MAP = {"pc_flower_purple": "flower_color", "pc_pubescence_tawny": "pubescence_color",
          "pc_indeterminate": "stem_termination", "pc_seedcoat_pigmented": "seed_coat_color"}


def block_intervals(scheme_idx, snp_cols, snps):
    rows = []
    for b, members in enumerate(scheme_idx):
        j = snp_cols[members[members >= 0]]
        sub = snps.iloc[j]
        rows.append((b, sub["chrom"].iloc[0], int(sub["pos"].min()), int(sub["pos"].max()), len(j)))
    return pd.DataFrame(rows, columns=["block", "chrom", "start", "end", "n_snps"])


def gene_distance(iv, genes):
    """Distance (bp) from each block interval to the nearest gene in `genes` (0 = overlap)."""
    d = np.full(len(iv), np.inf)
    name = np.array([""] * len(iv), dtype=object)
    for _, g in genes.iterrows():
        m = (iv["chrom"] == g["chrom"]).to_numpy()
        gap = np.maximum(0, np.maximum(g["start"] - iv["end"].to_numpy(), iv["start"].to_numpy() - g["end"]))
        better = m & (gap < d)
        d[better] = gap[better]
        name[better] = g["locus"]
    return d, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", default="hapgam_e")
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=1_000_000)
    ap.add_argument("--perm", type=int, default=10000)
    args = ap.parse_args()
    snps = data.grin_snps()
    genes = pd.read_csv(PROCESSED / "known_genes.tsv", sep="\t")
    rng = np.random.default_rng(1)
    out_rows, summary = [], []
    for trait in args.traits:
        key = PC_MAP.get(trait, trait)
        rel = genes[genes["traits"].str.split(",").map(lambda t: key in t)]
        d = RUNS / args.scenario / args.model / trait
        for f in sorted(d.glob("*.parts.npz")):
            z = np.load(f, allow_pickle=True)
            sel = z["selected"]
            if len(sel) == 0:
                continue
            cols = z["snp_cols"] if "snp_cols" in z.files else np.arange(len(snps))
            iv = block_intervals(z["scheme_idx"], cols, snps)
            dist, name = gene_distance(iv, rel) if len(rel) else (np.full(len(iv), np.inf), np.array([""] * len(iv)))
            imp = z["block_contrib"].var(0)
            fold = f.name[: -len(".parts.npz")]
            for r, b in enumerate(sel):
                out_rows.append({"trait": trait, "fold": fold, "rank": r, "block": int(b),
                                 "chrom": iv.at[b, "chrom"], "start": iv.at[b, "start"], "end": iv.at[b, "end"],
                                 "importance": float(imp[r]), "nearest_gene": name[b],
                                 "gene_distance": float(dist[b])})
            near_all = dist <= args.window
            obs = near_all[sel].mean()
            perm = np.array([near_all[rng.choice(len(iv), len(sel), replace=False)].mean() for _ in range(args.perm)])
            top = sel[np.argsort(-imp)[: min(5, len(sel))]]
            summary.append({"trait": trait, "fold": fold, "n_selected": len(sel), "share_near_gene": float(obs),
                            "expected_share": float(perm.mean()), "p_enrich": float((perm >= obs).mean()),
                            "top5_near_gene": ",".join(f"{name[b]}" if dist[b] <= args.window else "-" for b in top)})
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scenario}_{args.model}"
    pd.DataFrame(out_rows).to_csv(out / f"blocks_{tag}.tsv", sep="\t", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / f"blocks_enrichment_{tag}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
