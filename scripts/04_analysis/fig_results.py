"""Result figures: per-trait accuracy across models, and paired gains of HapGAM with CIs.

usage: fig_results.py accuracy --scenario dev_region --out figures/dev_region_accuracy
       fig_results.py gain --table results/tables/compare_dev_region_hapgam_e.tsv --out figures/...
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import collect  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# fixed colour per model (validated categorical order); unlisted models are drawn in gray
MODEL_STYLE = {
    "hapgam_e": ("HapGAM-E", "#2a78d6"),
    "gblup": ("GBLUP", "#eb6834"),
    "egblup": ("EGBLUP", "#1baf7a"),
    "lightgbm": ("LightGBM", "#eda100"),
    "bayesb": ("BayesB", "#e87ba4"),
    "hapgam_e_linear": ("HapGAM-E linear", "#008300"),
    "dnngp": ("DNNGP", "#6250d6"),
    "rkhs": ("RKHS", "#e34948"),
    "hapgam": ("HapGAM-A", "#86b6ef"),
    "hapgam_linear": ("HapGAM-A linear", MUTED),
    "hapgblup": ("Haplotype GBLUP", MUTED),
    "mlp": ("MLP", MUTED),
    "xgboost": ("XGBoost", MUTED),
}
TRAIT_LABEL = {
    "maturity_doy": "Maturity date", "flowering_doy": "Flowering date", "height_cm": "Plant height",
    "seed_weight_g": "100-seed weight", "yield_mg_ha": "Seed yield", "protein_pct": "Protein",
    "oil_pct": "Oil", "lodging_score": "Lodging", "maturity_group": "Maturity group",
}
TRAIT_ORDER = list(TRAIT_LABEL)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2})


def accuracy(scenario, out, metric="pearson", models=None):
    df = collect(scenario)
    if models:
        df = df[df["model"].isin(models)]
    tab = df.groupby(["trait", "model"])[metric].mean().unstack()
    traits = [t for t in TRAIT_ORDER if t in tab.index]
    models = [m for m in MODEL_STYLE if m in tab.columns]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(traits) + 1.0), dpi=300)
    off = np.linspace(-0.32, 0.32, len(models))
    for j, m in enumerate(models):
        label, col = MODEL_STYLE[m]
        y = np.arange(len(traits)) + off[j]
        x = tab.loc[traits, m].to_numpy()
        focal = m == "hapgam_e"
        ax.scatter(x, y, s=22 if focal else 12, color=col, edgecolor="white", linewidth=0.5,
                   zorder=3 if focal else 2, label=label)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TRAIT_LABEL[t] for t in traits])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel({"pearson": "Prediction accuracy (Pearson r, mean over folds)",
                   "within_group_r": "Within-group accuracy (mean r)"}.get(metric, metric), color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7, handletextpad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    tab.loc[traits, models].round(4).to_csv(f"{out}.tsv", sep="\t")
    print(f"{out}.png")


def delta(scenario, out, models, ref="gblup", traits=None, title=None):
    """Gain over a reference model (default GBLUP) per trait, on folds where every model finished.

    Large markers: mean over folds; small markers: individual folds (grouped scenarios)."""
    df = collect(scenario)
    df = df[df["model"].isin(set(models) | {ref})]
    if traits:
        df = df[df["trait"].isin(traits)]
    nm = df["model"].nunique()
    df = df[df.groupby(["trait", "fold"])["model"].transform("nunique") == nm]
    wide = df.pivot_table(index=["trait", "fold"], columns="model", values="pearson")
    d = wide.sub(wide[ref], axis=0).drop(columns=[ref])
    tr = [t for t in TRAIT_ORDER if t in d.index.get_level_values(0)]
    ms = [m for m in models if m in d.columns]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(tr) + 1.1), dpi=300)
    off = np.linspace(-0.3, 0.3, len(ms)) if len(ms) > 1 else [0.0]
    for j, m in enumerate(ms):
        label, col = MODEL_STYLE.get(m, (m, MUTED))
        for i, t in enumerate(tr):
            v = d.loc[t, m].to_numpy()
            ax.scatter(v, np.full(len(v), i + off[j]), s=5, color=col, alpha=0.45, lw=0, zorder=2)
            ax.scatter([v.mean()], [i + off[j]], s=26 if m == "hapgam_e" else 16, color=col, edgecolor="white",
                       lw=0.5, zorder=3, label=label if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(tr)))
    ax.set_yticklabels([TRAIT_LABEL[t] for t in tr])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Gain in Pearson r over {MODEL_STYLE.get(ref, (ref,))[0]}", color=INK)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7, handletextpad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    d.round(4).to_csv(f"{out}.tsv", sep="\t")
    print(f"{out}.png  (folds per trait: {d.groupby(level=0).size().to_dict()})")


def gain(table, out, baselines=("gblup", "egblup", "lightgbm", "bayesb")):
    df = pd.read_csv(table, sep="\t")
    df = df[df["baseline"].isin(baselines)]
    traits = [t for t in TRAIT_ORDER if t in set(df["trait"])]
    bl = [b for b in baselines if b in set(df["baseline"])]
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(traits) + 1.0), dpi=300)
    off = np.linspace(-0.27, 0.27, len(bl)) if len(bl) > 1 else [0.0]
    for j, b in enumerate(bl):
        label, col = MODEL_STYLE.get(b, (b, MUTED))
        sub = df[df["baseline"] == b].set_index("trait").reindex(traits)
        y = np.arange(len(traits)) + off[j]
        ax.errorbar(sub["diff"], y, xerr=[sub["diff"] - sub["ci_lo"], sub["ci_hi"] - sub["diff"]], fmt="o",
                    ms=3.5, color=col, ecolor=col, elinewidth=1.0, capsize=0, label=f"vs {label}")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TRAIT_LABEL[t] for t in traits])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Gain of HapGAM in Pearson r (95% bootstrap CI)", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"{out}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["accuracy", "gain", "delta"])
    ap.add_argument("--models", nargs="*", default=["egblup", "rkhs", "bayesb", "lightgbm", "mlp", "dnngp", "hapgam_e"])
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--scenario")
    ap.add_argument("--table")
    ap.add_argument("--metric", default="pearson")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = str(ROOT / args.out) if not args.out.startswith("/") else args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "accuracy":
        accuracy(args.scenario, out, args.metric)
    elif args.kind == "delta":
        delta(args.scenario, out, args.models, traits=args.traits, title=args.title)
    else:
        gain(args.table, out)


if __name__ == "__main__":
    main()
