"""Genome-wide map of HapGAM-selected blocks with known soybean genes.

Input: results/tables/blocks_<scenario>_<model>.tsv (from block_maps.py).  For each trait the
selected blocks are drawn as lollipops along the concatenated chromosomes; height is the block's
share of the total block-network variance on the test set (averaged over outer folds, 0 when not
selected), marker area the share of folds selecting it (multi-fold scenarios).  Blocks within
1 Mb of a trait-relevant known gene are drawn in blue, others in grey; trait-relevant known genes
are marked by triangles with labels that are spread apart where genes are close.

usage: fig_blockmap.py --scenario C3 --model hapgam_e --traits maturity_doy flowering_doy height_cm seed_weight_g
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.paths import FIGURES, PROCESSED, RESULTS  # noqa: E402

from block_maps import PC_MAP  # noqa: E402

PC_LABEL = {"pc_flower_purple": "Purple flower", "pc_pubescence_tawny": "Tawny pubescence",
            "pc_indeterminate": "Indeterminate stem", "pc_seedcoat_pigmented": "Pigmented seed coat"}
NEAR, FAR, GENE = fs.colour("hapgam_e"), "#a3adb8", "#3a3a3a"
WINDOW = 1_000_000


def chrom_offsets():
    snps = data.grin_snps()
    ends = snps.groupby("chrom")["pos"].max().sort_index()
    gap = 4e6
    off = np.concatenate([[0], np.cumsum(ends.to_numpy()[:-1] + gap)])
    return dict(zip(ends.index, off)), ends, float(off[-1] + ends.iloc[-1])


def spread(xs, widths, lo, hi, iters=6):
    """Label centres close to `xs` such that intervals [x - w/2, x + w/2] do not overlap and stay in [lo, hi]."""
    x0 = np.array(xs, float)
    w = np.array(widths, float)
    x = x0.copy()
    for _ in range(iters):
        for i in range(1, len(x)):                      # push right
            x[i] = max(x[i], x[i - 1] + (w[i - 1] + w[i]) / 2)
        x[-1] = min(x[-1], hi - w[-1] / 2)
        for i in range(len(x) - 2, -1, -1):              # push left
            x[i] = min(x[i], x[i + 1] - (w[i] + w[i + 1]) / 2)
        x[0] = max(x[0], lo + w[0] / 2)
        for i in range(1, len(x)):
            x[i] = max(x[i], x[i - 1] + (w[i - 1] + w[i]) / 2)
        # re-centre each cluster of touching labels on its genes
        start, n = 0, len(x)
        for i in range(1, n + 1):
            if i == n or x[i] - x[i - 1] > (w[i - 1] + w[i]) / 2 * (1 + 1e-6):
                lo_lim = x[start - 1] + (w[start - 1] + w[start]) / 2 if start > 0 else lo + w[start] / 2
                hi_lim = x[i] - (w[i - 1] + w[i]) / 2 if i < n else hi - w[i - 1] / 2
                shift = x0[start:i].mean() - x[start:i].mean()
                x[start:i] += np.clip(shift, min(0.0, lo_lim - x[start]), max(0.0, hi_lim - x[i - 1]))
                start = i
    return x


def label_widths(ax, names, x0, x1, size=6.0, pad_mm=1.8):
    """Width of each italic gene label in data units (measured with the renderer) plus a small gap."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    px_per_unit = ax.get_window_extent(r).width / (x1 - x0)
    pad_px = pad_mm / 25.4 * fig.dpi
    out = []
    for n in names:
        t = ax.text(0, 0, n, fontsize=size, style="italic")
        out.append((t.get_window_extent(r).width + pad_px) / px_per_unit)
        t.remove()
    return out


def draw(axes, blocks, genes, traits, off, ends, total, ymax_round=True):
    x0, x1 = -total * 0.004, total * 1.004
    ax_w_mm = axes[0].get_position().width * axes[0].figure.get_figwidth() / fs.MM
    mm = total / ax_w_mm               # genome units per mm
    for ax, trait in zip(axes, traits):
        ax.set_xlim(x0, x1)
        b = blocks[blocks["trait"] == trait].copy()
        nfold = max(b["fold"].nunique(), 1)
        b["share"] = b["importance"] / b.groupby("fold")["importance"].transform("sum")
        b["mid"] = (b["start"] + b["end"]) / 2 + b["chrom"].map(off)
        b["near"] = b["gene_distance"] <= WINDOW
        agg = b.groupby(["chrom", "start", "end", "mid"], as_index=False).agg(
            freq=("fold", "nunique"), share=("share", "sum"), near=("near", "max"))
        agg["share"] = agg["share"] / nfold            # mean share over folds (0 when not selected)
        agg["freq"] = agg["freq"] / nfold
        agg = agg.sort_values("share")
        for i, c in enumerate(ends.index):
            if i % 2 == 1:
                ax.axvspan(off[c], off[c] + ends[c], color="#f1f2f4", lw=0, zorder=0)
        key = PC_MAP.get(trait, trait)
        rel = genes[genes["traits"].str.split(",").map(lambda t: key in t)].copy()
        rel["x"] = rel["chrom"].map(off) + (rel["start"] + rel["end"]) / 2
        rel = rel.sort_values("x")
        top = agg["share"].max() if len(agg) else 0.05
        step = 0.2 if top > 0.45 else (0.1 if top > 0.2 else (0.05 if top > 0.08 else 0.02))
        ymax = np.ceil(top * 1.05 / step) * step
        for _, g in rel.iterrows():
            ax.plot([g["x"], g["x"]], [0, ymax], color=GENE, lw=0.4, ls=(0, (1.2, 1.4)), zorder=1)
        col = np.where(agg["near"], NEAR, FAR)
        ax.vlines(agg["mid"], 0, agg["share"], colors=col, lw=0.7, zorder=2)
        size = 4 + 22 * agg["freq"] if nfold > 1 else np.full(len(agg), 11.0)
        ax.scatter(agg["mid"], agg["share"], s=size, c=col, edgecolor="white", lw=0.35, zorder=3)
        # gene markers and labels above the track
        if len(rel):
            widths = label_widths(ax, rel["locus"], x0, x1)
            lx = spread(rel["x"].to_numpy(), widths, x0, x1)
            shifted = np.abs(lx - rel["x"].to_numpy()) > 0.3 * mm
            yl = ymax * (1.20 if shifted.any() else 1.10)
            for (_, g), x_lab, sh in zip(rel.iterrows(), lx, shifted):
                ax.plot([g["x"]], [ymax * 1.035], marker="v", ms=2.6, color=GENE, mew=0, clip_on=False, zorder=4)
                if sh:
                    ax.plot([g["x"], x_lab], [ymax * 1.07, yl - ymax * 0.015], color=GENE, lw=0.35, clip_on=False)
                ax.text(x_lab, yl, g["locus"], ha="center", va="bottom", fontsize=6.0, style="italic", color=fs.INK,
                        clip_on=False)
        ax.set_ylim(0, ymax)
        nstep = int(round(ymax / step))
        ticks = np.arange(0, nstep + 1, 1 if nstep <= 3 else 2) * step
        ax.set_yticks(ticks)
        dec = max(len(f"{t:g}".split(".")[1]) if "." in f"{t:g}" else 0 for t in ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=dec: "0" if v == 0 else f"{v:.{d}f}"))
        # trait name left of the tick labels (fixed offset in points, so it never touches them)
        ax.annotate(TRAIT_NAME(trait).replace(" ", "\n", 1), xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-24, 0), textcoords="offset points", ha="right", va="center", fontsize=6.8,
                    color=fs.INK, linespacing=1.1, annotation_clip=False)
        ax.spines["left"].set_bounds(0, ymax)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks([off[c] + ends[c] / 2 for c in ends.index])
    axes[-1].set_xticklabels([str(int(c.replace("Gm", ""))) for c in ends.index], fontsize=6.2)
    for ax in axes:
        ax.set_xlim(x0, x1)


def TRAIT_NAME(trait):
    return fs.TRAIT_LABEL.get(trait, PC_LABEL.get(trait, trait))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--model", default="hapgam_e")
    ap.add_argument("--traits", nargs="+", default=None)
    ap.add_argument("--scenario2", default=None, help="optional second panel (b) from another scenario")
    ap.add_argument("--traits2", nargs="*", default=None)
    ap.add_argument("--titles", nargs="*", default=None, help="panel titles for a (and b)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", choices=["fig5", "figS2"], default=None)
    args = ap.parse_args()
    if args.preset == "fig5":
        args.scenario, args.traits = "pc_random", list(PC_LABEL)
        args.scenario2, args.traits2 = "C2", fs.PRIMARY
        args.titles = ["Mendelian traits", "Primary quantitative traits (Japanese test set, C2)"]
        args.out = str(FIGURES / "fig5_blockmaps")
    elif args.preset == "figS2":
        args.scenario, args.traits = "dev_region", fs.PRIMARY
        args.out = str(FIGURES / "figS2_blockmap_dev_region")
    genes = pd.read_csv(PROCESSED / "known_genes.tsv", sep="\t")
    off, ends, total = chrom_offsets()
    panels = [(args.scenario, args.traits)] + ([(args.scenario2, args.traits2)] if args.scenario2 else [])
    track, sep, head, gap_mm, foot = 14.5, 7.0, 6.5, 5.0, 9.0      # mm
    h_mm = sum(head + len(t) * (track + sep) for _, t in panels) + foot * len(panels) + gap_mm * (len(panels) - 1)
    fig = plt.figure(figsize=(fs.W2, h_mm * fs.MM))
    left, right = 0.165, 0.985
    y = h_mm
    for k, (sc, traits) in enumerate(panels):
        blocks = pd.read_csv(RESULTS / "tables" / f"blocks_{sc}_{args.model}.tsv", sep="\t")
        y -= head
        if len(panels) > 1 or args.titles:
            title = args.titles[k] if args.titles and k < len(args.titles) else sc
            fs.panel_label(fig, 0.004, (y + head - 0.6) / h_mm, "ab"[k])
            fig.text(0.03, (y + head - 1.0) / h_mm, title, fontsize=7, fontweight="bold", va="top")
        axes = []
        for i in range(len(traits)):
            y -= sep + track
            axes.append(fig.add_axes([left, y / h_mm, right - left, track / h_mm]))
        draw(axes, blocks, genes, traits, off, ends, total)
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        axes[-1].set_xlabel("Chromosome", labelpad=2)
        ymid = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
        fig.text(0.012, ymid, "Share of block-network variance", rotation=90, va="center", ha="left", fontsize=6.8)
        y -= foot + (gap_mm if k < len(panels) - 1 else 0)
    tag = f"{args.scenario}" + (f"+{args.scenario2}" if args.scenario2 else "")
    out = Path(args.out) if args.out else FIGURES / f"blockmap_{tag}_{args.model}"
    fs.save(fig, out)


if __name__ == "__main__":
    main()
