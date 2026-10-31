"""Relate HapGAM's gain to genetic architecture.

Major-locus concentration index (per outer fold, training data only): share of the summed
block-screening scores (variance of each block's additive BLUP contribution, 16-SNP blocks) that
falls in the top 8 blocks.  Gain: Pearson r of HapGAM minus that of its own backbone (EGBLUP for
HapGAM-E, GBLUP for HapGAM-A) on the same test fold.

usage: gain_vs_architecture.py --scenario dev_region --focal hapgam_e --base egblup
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import FIGURES, RESULTS, RUNS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_results import TRAIT_LABEL, TRAIT_ORDER  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#898781"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dev_region")
    ap.add_argument("--focal", default="hapgam_e")
    ap.add_argument("--base", default="egblup")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for tdir in sorted((RUNS / args.scenario / args.focal).iterdir()):
        trait = tdir.name
        for f in sorted(tdir.glob("*.parts.npz")):
            fold = f.name[: -len(".parts.npz")]
            bj = RUNS / args.scenario / args.base / trait / f"{fold}.json"
            fj = tdir / f"{fold}.json"
            if not bj.exists():
                continue
            sc = np.sort(np.load(f, allow_pickle=True)["block_scores"])[::-1]
            conc = float(sc[: args.top].sum() / sc.sum())
            r_f = json.loads(fj.read_text())["pearson"]
            r_b = json.loads(bj.read_text())["pearson"]
            rows.append({"trait": trait, "fold": fold, "concentration": conc, "gain": r_f - r_b})
    df = pd.DataFrame(rows)
    out = RESULTS / "tables" / f"gain_vs_architecture_{args.scenario}_{args.focal}.tsv"
    df.to_csv(out, sep="\t", index=False)
    rho, p = stats.spearmanr(df["concentration"], df["gain"])
    print(df.groupby("trait")[["concentration", "gain"]].mean().round(4).to_string())
    print(f"Spearman rho={rho:.3f} p={p:.2g} (n={len(df)} folds)")
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=300)
    traits = [t for t in TRAIT_ORDER if t in set(df["trait"])]
    for i, t in enumerate(traits):
        s = df[df["trait"] == t]
        ax.scatter(s["concentration"], s["gain"], s=14, color=PALETTE[i % len(PALETTE)], edgecolor="white", lw=0.4,
                   label=TRAIT_LABEL[t])
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel(f"Share of BLUP block variance in top {args.top} blocks", color=INK, fontsize=7)
    ax.set_ylabel(f"Gain in r over {args.base.upper()}", color=INK, fontsize=7)
    ax.text(0.98, 0.03, f"Spearman ρ = {rho:.2f} (P = {p:.1g}, {len(df)} folds)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5)
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"gain_vs_architecture_{args.scenario}_{args.focal}.{ext}", bbox_inches="tight")
    print(FIGURES / f"gain_vs_architecture_{args.scenario}_{args.focal}.png")


if __name__ == "__main__":
    main()
