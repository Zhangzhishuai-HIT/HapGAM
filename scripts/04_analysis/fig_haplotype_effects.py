"""Learned block functions by haplotype allele for the most important HapGAM blocks, with a test-set check.

For a GRIN scenario fold, test accessions are grouped by their haplotype (allele string) in a
selected block.  For every haplotype carried by >= min_n test accessions the figure shows
  - the block's learned contribution to the prediction (median over carriers; the function was
    estimated on the training accessions only), and
  - the mean partial residual of the carriers in the test set, (y - y_hat) - mean(y - y_hat) + f_b,
    with a 95% confidence interval: the part of the observed phenotype that the rest of the model
    leaves to the block.  Agreement between the two (carrier-weighted correlation r_w) shows how well
    the haplotype effects estimated in the training set hold in the test accessions (post hoc check).
The additive R^2 of a block is the share of the variance of its contribution that a linear
function of its SNP dosages explains.

usage: fig_haplotype_effects.py --preset fig6|figS3
       fig_haplotype_effects.py --scenario C2 --fold lockbox_japan --panels maturity_doy:0 height_cm:0 --out ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.paths import FIGURES, PROCESSED, RUNS  # noqa: E402
from hapgam.targets import safe  # noqa: E402

from block_maps import block_intervals, gene_distance  # noqa: E402

LEARNED, OBSERVED = fs.colour("hapgam_e"), "#3a3a3a"
UNITS = {"maturity_doy": "days", "flowering_doy": "days", "height_cm": "cm", "seed_weight_g": "g",
         "yield_mg_ha": "Mg ha$^{-1}$", "protein_pct": "%", "oil_pct": "%", "lodging_score": "score",
         "maturity_group": "MG units"}
PRESETS = {
    "fig6": dict(panels=["maturity_doy:0", "maturity_doy:2", "flowering_doy:2", "height_cm:0"], r2=False,
                 out="fig6_haplotypes_C2", layout=[[0, 1], [2, 3]]),
    "figS3": dict(panels=["maturity_doy:1", "flowering_doy:1", "maturity_group:0", "maturity_group:1",
                          "lodging_score:1", "height_cm:1", "height_cm:2", "height_cm:3"], r2=True,
                  out="figS3_haplotypes_C2_extra", layout=[[0, 0, 0, 1, 1, 1], [2, 2, 3, 3, 4, 4],
                                                           [5, 5, 6, 6, 7, 7]]),
}


def additive_r2(Zb, contrib):
    """R^2 of the best linear (additive-in-SNPs) fit to the block contribution."""
    A = np.column_stack([np.ones(len(Zb)), Zb])
    coef, *_ = np.linalg.lstsq(A, contrib, rcond=None)
    res = contrib - A @ coef
    return 1 - res.var() / max(contrib.var(), 1e-12)


def load_fold(scenario, model, trait, fold):
    d = RUNS / scenario / model / trait
    z = np.load(d / f"{safe(fold)}.parts.npz", allow_pickle=True)
    pred = pd.read_csv(d / f"{safe(fold)}.pred.tsv", sep="\t")
    acc = data.grin_accessions().set_index("accession")
    rows = acc.loc[pred["id"], "row"].to_numpy()
    G = np.asarray(data.grin_dosage()[rows])
    iv = block_intervals(z["scheme_idx"], z["snp_cols"], data.grin_snps())
    genes = pd.read_csv(PROCESSED / "known_genes.tsv", sep="\t")
    rel = genes[genes["traits"].str.split(",").map(lambda t: trait in t)]
    dist, name = gene_distance(iv, rel)
    return z, G, iv, dist, name


def wcorr(a, b, w):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    return float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b)))


def haplotype_table(z, G, pred, r, min_n):
    """Per-haplotype learned contribution and test-set partial residual for the block of importance rank r."""
    contrib = z["block_contrib"]
    order = np.argsort(-contrib.var(0))
    k = int(order[r])
    b = int(z["selected"][k])
    members = z["scheme_idx"][b]
    members = z["snp_cols"][members[members >= 0]]
    Gb = G[:, members].astype(int)
    hap = np.array(["".join(map(str, row // 2 if row.max() <= 2 else row)) for row in Gb])
    c = contrib[:, k]
    y, yhat = pred["y"].to_numpy(float), pred["pred"].to_numpy(float)
    ok = np.isfinite(y)
    res = y - yhat
    pr = res - np.nanmean(res[ok]) + c
    df = pd.DataFrame({"hap": hap, "c": c, "pr": pr})[ok]
    cnt = df["hap"].value_counts()
    sub = df[df["hap"].isin(cnt[cnt >= min_n].index)]
    tab = sub.groupby("hap").agg(n=("c", "size"), learned=("c", "median"), observed=("pr", "mean"), sd=("pr", "std"))
    tab["ci"] = 1.96 * tab["sd"] / np.sqrt(tab["n"])
    tab = tab.sort_values("learned")
    r2 = additive_r2(Gb - Gb.mean(0), c)          # all test accessions, as in additive_r2.py
    return b, members, tab, r2


def panel(ax, fold_data, pred, r, trait, min_n, letter=None, show_r2=True, ylabel=True):
    z, G, iv, dist, name = fold_data
    b, members, tab, r2 = haplotype_table(z, G, pred, r, min_n)
    x = np.arange(len(tab))
    w = tab["n"].to_numpy(float)
    rw = wcorr(tab["learned"].to_numpy(), tab["observed"].to_numpy(), w)
    ax.axhline(0, color=fs.LIGHT, lw=0.5, zorder=0)
    ax.errorbar(x + 0.17, tab["observed"], yerr=tab["ci"], fmt="D", ms=2.6, mfc="white", mec=OBSERVED, mew=0.7,
                ecolor=OBSERVED, elinewidth=0.6, capsize=0, zorder=2)
    ax.scatter(x - 0.12, tab["learned"], s=9 + 34 * np.sqrt(w / w.max()), color=LEARNED, edgecolor="white", lw=0.4,
               zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in tab["n"]], fontsize=6.0)
    ax.tick_params(axis="x", length=1.5, pad=1.5)
    ax.set_xlim(-0.6, len(tab) - 0.4)
    ax.set_xlabel("Haplotype (no. of Japanese carriers)", fontsize=6.3)
    near = np.isfinite(dist[b]) and dist[b] <= 1e6
    where = f"{iv.at[b, 'chrom']}: {iv.at[b, 'start'] / 1e6:.2f}–{iv.at[b, 'end'] / 1e6:.2f} Mb"
    bold = {"fontweight": "bold"}
    gene = {"fontweight": "bold", "style": "italic"}
    parts = [(fs.TRAIT_LABEL.get(trait, trait), bold)]
    if near and dist[b] == 0:                      # block contains the gene
        parts += [(", ", bold), (str(name[b]), gene), (" block", bold)]
    elif near:                                     # block within 1 Mb of the gene
        kb = dist[b] / 1e3
        parts += [(", block near ", bold), (str(name[b]), gene), (f" ({kb:.0f} kb)", bold)]
    fs.rich_text(ax, 0, 1, parts, fontsize=6.8, va="baseline", dy=11.5)
    fs.rich_text(ax, 0, 1, [(where, {"color": fs.INK2})], fontsize=6.2, va="baseline", dy=3.5)
    note = f"$r_w$ = {rw:.2f} ({len(tab)} haplotypes)" + (f"\nadditive $R^2$ = {r2:.2f}" if show_r2 else "")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=6.2, va="top", ha="left", color=fs.INK,
            linespacing=1.3)
    if ylabel:
        ax.set_ylabel(f"Effect ({UNITS.get(trait, 'trait units')})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.minus(f"{v:g}")))
    if letter:
        fs.letter_at(ax, letter, dx_mm=-10.5, dy_pt=11.5)
    lo, hi = np.nanmin(tab["observed"] - tab["ci"]), np.nanmax(tab["observed"] + tab["ci"])
    lo, hi = min(lo, tab["learned"].min()), max(hi, tab["learned"].max())
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + (0.30 if show_r2 else 0.20) * (hi - lo) + pad)
    return {"trait": trait, "rank": r, "block": b, "chrom": iv.at[b, "chrom"], "start": iv.at[b, "start"],
            "end": iv.at[b, "end"], "n_snps": len(members), "n_haplotypes": len(tab),
            "span_learned": float(tab["learned"].max() - tab["learned"].min()), "additive_r2": r2,
            "weighted_r_learned_observed": rw, "nearest_gene": name[b] if near else "", "gene_distance": dist[b],
            "haplotypes": ";".join(f"{h}:{int(n)}:{l:.3f}:{o:.3f}:{c:.3f}" for h, n, l, o, c in
                                   zip(tab.index, tab["n"], tab["learned"], tab["observed"], tab["ci"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--scenario", default="C2")
    ap.add_argument("--model", default="hapgam_e")
    ap.add_argument("--fold", default="lockbox_japan")
    ap.add_argument("--panels", nargs="*", default=None, help="trait:rank pairs (rank 0 = most important block)")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-r2", action="store_true", help="omit additive R^2 from the panels")
    args = ap.parse_args()
    if args.preset:
        p = PRESETS[args.preset]
        args.panels, args.out, show_r2, layout = p["panels"], str(FIGURES / p["out"]), p["r2"], p["layout"]
    else:
        show_r2 = not args.no_r2
        layout = [list(range(len(args.panels)))]
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in args.panels]
    nrow, ncol = len(layout), len(layout[0])
    row_h = 56 if nrow == 1 else 52
    H = row_h * nrow + 10
    fig = plt.figure(figsize=(fs.W2, H * fs.MM))
    gs = fig.add_gridspec(nrow, ncol, left=0.075, right=0.985, top=1 - 19 / H, bottom=12 / H,
                          wspace=0.85 if ncol > 4 else (0.32 if ncol == 2 else 0.52), hspace=0.62)
    cache, summary = {}, []
    for k, (trait, r) in enumerate(specs):
        if trait not in cache:
            fd = load_fold(args.scenario, args.model, trait, args.fold)
            pred = pd.read_csv(RUNS / args.scenario / args.model / trait / f"{safe(args.fold)}.pred.tsv", sep="\t")
            cache[trait] = (fd, pred)
        cells = [(i, j) for i, row in enumerate(layout) for j, v in enumerate(row) if v == k]
        i0, j0, j1 = cells[0][0], min(c[1] for c in cells), max(c[1] for c in cells)
        ax = fig.add_subplot(gs[i0, j0:j1 + 1])
        summary.append(panel(ax, *cache[trait], r, trait, args.min_n, letter="abcdefgh"[k], show_r2=show_r2))
    h = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=LEARNED, mec="white", mew=0.4,
                label="Effect learned outside Japan (block network; marker area ∝ carriers)"),
         Line2D([], [], ls="-", lw=0.6, marker="D", ms=2.8, mfc="white", mec=OBSERVED, mew=0.7, color=OBSERVED,
                label="Observed in Japan: mean partial residual ± 95% CI")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.0), columnspacing=2.0)
    out = Path(args.out)
    fs.save(fig, out)
    pd.DataFrame(summary).to_csv(f"{out}.tsv", sep="\t", index=False)
    print(pd.DataFrame(summary).drop(columns=["haplotypes"]).to_string(index=False))


if __name__ == "__main__":
    main()
