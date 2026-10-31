"""Composite main-text and supplementary result figures (final print size, shared style in figstyle.py).

  fig2   training-set cross-validation (dev_region): (a) gain over GBLUP per trait for EGBLUP, BayesB, LightGBM
         and HapGAM, (b) gain of HapGAM over EGBLUP against the share of the BLUP block variance in the top eight blocks
  fig3   test sets: HapGAM minus GBLUP, EGBLUP, BayesB and LightGBM with 95% bootstrap CIs in the Japanese test set
         (C2, a) and the random test set (C1, b)
  fig4   overview of the reported scenarios (primary traits): HapGAM minus the better of GBLUP and EGBLUP, minus
         LightGBM, and the HapGAM + LightGBM ensemble minus the best of the five single models
  figS1  C3: HapGAM minus EGBLUP and minus LightGBM in each held-out region (four primary traits)

usage: fig_main.py fig2|fig3|fig4|figS1
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import scenarios  # noqa: E402
from hapgam.paths import FIGURES, RESULTS  # noqa: E402
from summarize import collect  # noqa: E402

TL, ORDER, PRIMARY = fs.TRAIT_LABEL, fs.TRAIT_ORDER, fs.PRIMARY
TABLES = RESULTS / "tables"


@lru_cache(maxsize=None)
def folds(scenario, trait):
    return tuple(scenarios.folds(scenario, trait))


def complete(df, scenario):
    """Keep model x trait cells that have every fold of the scenario."""
    keep = []
    for (m, t), g in df.groupby(["model", "trait"]):
        need = set(folds(scenario, t))
        if need <= set(g["fold"]):
            keep.append(g[g["fold"].isin(need)])
    return pd.concat(keep)


def bands(ax, n, horizontal=True):
    """Light alternating bands behind rows 0..n-1 (rows centred on integers)."""
    for i in range(0, n, 2):
        if horizontal:
            ax.axhspan(i - 0.5, i + 0.5, color=fs.BAND, lw=0, zorder=0)


def model_handle(m, filled=True, size=4.2):
    c = fs.colour(m)
    return Line2D([], [], ls="none", marker=fs.marker(m), ms=size, mfc=c if filled else "white", mec=c, mew=0.8,
                  label=fs.label(m))


# --------------------------------------------------------------------------------------------- fig 2
def fig2():
    fig = plt.figure(figsize=(fs.W2, 82 * fs.MM))
    ax1 = fig.add_axes([0.125, 0.115, 0.36, 0.80])
    ax2 = fig.add_axes([0.615, 0.115, 0.37, 0.80])
    models = ["egblup", "bayesb", "lightgbm", "hapgam_e"]
    df = complete(collect("dev_region"), "dev_region")
    wide = df[df["model"].isin(models + ["gblup"])].pivot_table(index=["trait", "fold"], columns="model",
                                                                values="pearson")
    d = wide[models].sub(wide["gblup"], axis=0)
    tr = [t for t in ORDER if t in d.index.get_level_values(0)]
    bands(ax1, len(tr))
    off = dict(zip(models, np.linspace(-0.27, 0.27, len(models))))
    for m in models:
        c = fs.colour(m)
        for i, t in enumerate(tr):
            v = d.loc[t, m].dropna().to_numpy()
            y = i + off[m]
            ax1.scatter(v, np.full(len(v), y), s=3.2, color=c, alpha=0.45, lw=0, zorder=2)
            big = m == "hapgam_e"
            ax1.scatter([v.mean()], [y], s=26 if big else 17, marker=fs.marker(m), color=c, edgecolor="white",
                        lw=0.5, zorder=4 if big else 3)
    ax1.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
    ax1.set_yticks(range(len(tr)))
    ax1.set_yticklabels([TL[t] for t in tr])
    ax1.tick_params(axis="y", length=0)
    ax1.set_ylim(len(tr) - 0.5, -0.5)
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Gain in accuracy over GBLUP (Δ$r$)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.minus(f"{v:.2f}") if v else "0"))
    ax1.legend(handles=[model_handle(m) for m in models], loc="lower right", ncol=1, borderaxespad=0.1,
               handletextpad=0.2, labelspacing=0.3)
    fs.panel_label(fig, 0.005, 0.985, "a")
    # b: gain over the backbone vs concentration of the BLUP block variance
    g = pd.read_csv(TABLES / "gain_vs_architecture_dev_region_hapgam_e.tsv", sep="\t")
    ax2.axhline(0, color=fs.INK2, lw=0.6, zorder=1)
    ax2.scatter(g["concentration"], g["gain"], s=7, color="#b8c4d3", lw=0, zorder=2)
    s = g.groupby("trait").agg(cm=("concentration", "mean"), gm=("gain", "mean"), gs=("gain", "sem"))
    for t, r in s.iterrows():
        ax2.errorbar(r["cm"], r["gm"], yerr=r["gs"], fmt="o", ms=4.0, color=fs.colour("hapgam_e"), mec="white",
                     mew=0.5, elinewidth=0.7, capsize=0, zorder=4)
    # label position (x, y, ha) in data units; 100-seed weight gets a leader line
    # label position (x, y, ha) in data units; labels away from their point get a leader line
    place = {"height_cm": (0.1565, 0.0206, "left"), "maturity_doy": (0.1478, 0.0113, "left"),
             "lodging_score": (0.1350, 0.0029, "left"), "maturity_group": (0.1180, 0.0168, "center"),
             "flowering_doy": (0.0960, 0.0126, "center"), "oil_pct": (0.0753, 0.0160, "center"),
             "protein_pct": (0.0630, 0.0055, "right"), "yield_mg_ha": (0.0446, -0.0031, "center"),
             "seed_weight_g": (0.0705, -0.0060, "left")}
    leaders = {"maturity_group": 0.0158, "flowering_doy": 0.0116, "seed_weight_g": -0.0052}
    for t, r in s.iterrows():
        x, y, ha = place[t]
        ax2.text(x, y, TL[t], fontsize=6.2, ha=ha, va="center", color=fs.INK, zorder=5,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        if t in leaders:
            y_end = r["gm"] + (r["gs"] + 0.0006 if leaders[t] > r["gm"] else -(r["gs"] + 0.0006))
            ax2.plot([r["cm"], x], [y_end, leaders[t]], color=fs.INK2, lw=0.4, zorder=3)
    rho, p = stats.spearmanr(g["concentration"], g["gain"])
    ax2.text(0.03, 0.05, f"Spearman ρ = {rho:.2f}, $P$ = {p:.3f}\n{len(g)} trait × region folds",
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=6.5, color=fs.INK, linespacing=1.3)
    ax2.set_xlabel("Share of BLUP block variance in the top eight blocks")
    ax2.set_ylabel("Gain of HapGAM over EGBLUP (Δ$r$)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.minus(f"{v:.2f}") if v else "0"))
    ax2.set_xlim(0.02, 0.225)
    ax2.set_xticks([0.05, 0.10, 0.15, 0.20])
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fs.panel_label(fig, 0.525, 0.985, "b")
    fs.save(fig, FIGURES / "fig2_development")


# --------------------------------------------------------------------------------------------- fig 3
def forest(ax, scenario, bases, xlim, show_y=True):
    comp = pd.read_csv(TABLES / f"compare_{scenario}_hapgam_e.tsv", sep="\t")
    df = complete(collect(scenario), scenario)
    rfocal = df[df["model"] == "hapgam_e"].groupby("trait")["pearson"].mean()
    traits = [t for t in ORDER if t in set(comp["trait"])]
    bands(ax, len(traits))
    off = dict(zip(bases, np.linspace(-0.27, 0.27, len(bases))))
    for b in bases:
        sub = comp[comp["baseline"] == b].set_index("trait").reindex(traits)
        c = fs.colour(b)
        for i, t in enumerate(traits):
            r = sub.loc[t]
            if not np.isfinite(r["diff"]):
                continue
            sig = r["p_holm"] < 0.05
            y = i + off[b]
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot([r["diff"]], [y], ls="none", marker=fs.marker(b), ms=3.6, mfc=c if sig else "white", mec=c,
                    mew=0.8, zorder=3)
    ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TL[t] for t in traits] if show_y else [])
    ax.tick_params(axis="y", length=0)
    # accuracy of HapGAM-E, right of the panel
    for i, t in enumerate(traits):
        ax.text(1.015, i, f"{rfocal[t]:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.2, color=fs.INK2)
    ax.text(1.015, -0.75, "r", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.2,
            color=fs.INK2, style="italic")
    ax.set_ylim(len(traits) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("HapGAM minus other model (Δ$r$, 95% CI)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.minus(f"{v:.2f}") if v else "0"))


def fig3():
    fig = plt.figure(figsize=(fs.W2, 84 * fs.MM))
    bases = ["gblup", "egblup", "bayesb", "lightgbm"]
    xlim = (-0.042, 0.092)
    ax1 = fig.add_axes([0.125, 0.115, 0.355, 0.735])
    ax2 = fig.add_axes([0.575, 0.115, 0.355, 0.735])
    forest(ax1, "C2", bases, xlim)
    forest(ax2, "C1", bases, xlim, show_y=False)
    for ax, title in ((ax1, "C2: Japanese test set (2,931 accessions)"),
                      (ax2, "C1: random test set (2,175 accessions)")):
        ax.set_title(title, loc="left", fontsize=7, fontweight="bold", pad=9)
    h = [model_handle(b) for b in bases]
    h += [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=fs.INK, mec=fs.INK, label="Holm-adjusted $P$ < 0.05"),
          Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=fs.INK, mew=0.8, label="$P$ ≥ 0.05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.53, 0.998), ncol=6, handletextpad=0.25,
               columnspacing=1.4)
    fs.panel_label(fig, 0.005, 0.925, "a")
    fs.panel_label(fig, 0.535, 0.925, "b")
    fs.save(fig, FIGURES / "fig3_C2")


# --------------------------------------------------------------------------------------------- fig 4
SCEN6 = ["C1", "C2", "C3", "grin2soynam", "grin2nust"]
NAMES6 = {"C1": "C1  random test set", "C2": "C2  Japanese test set", "C3": "C3  leave-one-region-out",
          "grin2soynam": "C4  GRIN → SoyNAM", "grin2nust": "C5  GRIN → NUST"}
LINEAR = ["gblup", "egblup"]
SINGLES = ["gblup", "egblup", "bayesb", "lightgbm", "hapgam_e_linear", "hapgam_e"]


def _overview():
    am = pd.read_csv(TABLES / "all_metrics.tsv", sep="\t")
    out = {k: np.full((len(SCEN6), len(PRIMARY)), np.nan) for k in ("lin", "lgb", "ens", "lin_sig", "lgb_sig")}
    for i, sc in enumerate(SCEN6):
        metric = "within_r" if sc.startswith("grin2") else "pearson"
        tab = am[am["scenario"] == sc].pivot_table(index="trait", columns="model", values=metric)
        comp_f = TABLES / (f"compare_{sc}_hapgam_e_within.tsv" if sc.startswith("grin2") else f"compare_{sc}_hapgam_e.tsv")
        comp = pd.read_csv(comp_f, sep="\t").set_index(["trait", "baseline"])
        pcol = "nb_p_holm" if sc == "soynam_random" else "p_holm"
        for j, t in enumerate(PRIMARY):
            if t not in tab.index or "hapgam_e" not in tab.columns or not np.isfinite(tab.at[t, "hapgam_e"]):
                continue
            row = tab.loc[t]
            f = row["hapgam_e"]
            lin = row.reindex(LINEAR).dropna()
            if len(lin):
                best = lin.idxmax()
                out["lin"][i, j] = f - lin[best]
                if (t, best) in comp.index:
                    out["lin_sig"][i, j] = comp.at[(t, best), pcol] < 0.05
            if "lightgbm" in row and np.isfinite(row["lightgbm"]):
                out["lgb"][i, j] = f - row["lightgbm"]
                if (t, "lightgbm") in comp.index:
                    out["lgb_sig"][i, j] = comp.at[(t, "lightgbm"), pcol] < 0.05
            if "ens_hapgam_e_lightgbm" in row and np.isfinite(row["ens_hapgam_e_lightgbm"]):
                out["ens"][i, j] = row["ens_hapgam_e_lightgbm"] - row.reindex(SINGLES).dropna().max()
    return out


def fig4():
    o = _overview()
    fig = plt.figure(figsize=(fs.W2, 47 * fs.MM))
    vmax = 0.05
    x0, w, gap = 0.215, 0.215, 0.022
    axes = [fig.add_axes([x0 + k * (w + gap), 0.157, w, 0.61]) for k in range(3)]
    titles = ["HapGAM \u2212 better of\nGBLUP and EGBLUP", "HapGAM \u2212 LightGBM", "Ensemble (HapGAM + LightGBM)\n\u2212 best single model"]
    for k, (ax, key, title) in enumerate(zip(axes, ["lin", "lgb", "ens"], titles)):
        M = o[key]
        ax.imshow(np.clip(M, -vmax, vmax), cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        sig = o.get(f"{key}_sig")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec="none"))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=6.2, color=fs.MUTED)
                    continue
                dark = abs(v) > 0.032
                txt = "0.000" if abs(v) < 5e-4 else f"{v:+.3f}".replace("-", "−")
                bold = sig is not None and sig[i, j] == 1
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.0, color="white" if dark else fs.INK,
                        fontweight="bold" if bold else "normal")
        for i in range(1, M.shape[0]):
            ax.axhline(i - 0.5, color="white", lw=0.8)
        for j in range(1, M.shape[1]):
            ax.axvline(j - 0.5, color="white", lw=0.8)
        ax.set_xticks(range(len(PRIMARY)))
        ax.set_xticklabels([TL[t].replace(" ", "\n", 1) for t in PRIMARY], fontsize=6.2, linespacing=1.0)
        ax.set_yticks(range(len(SCEN6)))
        ax.set_yticklabels([NAMES6[s] for s in SCEN6] if k == 0 else [])
        ax.tick_params(length=0, pad=2)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, fontsize=6.8, fontweight="bold", linespacing=1.1, pad=4)
    cax = fig.add_axes([x0 + 3 * w + 2 * gap + 0.012, 0.20, 0.009, 0.52])
    sm = plt.cm.ScalarMappable(cmap=fs.DIVERGING, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[-0.05, -0.025, 0, 0.025, 0.05])
    cb.ax.set_yticklabels(["≤ −0.05", "−0.025", "0", "+0.025", "≥ +0.05"], fontsize=6.0)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.8)
    cb.set_label("Δ$r$", fontsize=6.5, labelpad=2)
    fs.save(fig, FIGURES / "fig4_overview")


# --------------------------------------------------------------------------------------------- figure S1
def figS1():
    """C3: HapGAM minus EGBLUP and minus LightGBM in each held-out region, four primary traits."""
    fig, axes = plt.subplots(1, 4, figsize=(fs.W2, 72 * fs.MM), sharex=True,
                             gridspec_kw={"wspace": 0.08, "left": 0.165, "right": 0.99, "top": 0.83, "bottom": 0.16})
    comps = [("egblup", -0.15), ("lightgbm", 0.15)]
    sc = "C3"
    df = collect(sc)
    for col, t in enumerate(PRIMARY):
        ax = axes[col]
        groups = list(folds(sc, t))
        sub = df[df["trait"] == t].pivot_table(index="fold", columns="model", values="pearson")
        bands(ax, len(groups))
        for b, off in comps:
            if "hapgam_e" not in sub or b not in sub:
                continue
            dd = (sub["hapgam_e"] - sub[b]).reindex(groups)
            ax.plot(dd.to_numpy(), np.arange(len(groups)) + off, ls="none", marker=fs.marker(b), ms=3.6,
                    mfc=fs.colour(b), mec="white", mew=0.4, zorder=3)
        ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g.replace("Russia/C.Asia", "Russia/C. Asia") for g in groups] if col == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.set_title(TL[t], fontsize=7, fontweight="bold")
        ax.set_xlabel("Δ$r$")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.minus(f"{v:.2f}") if v else "0"))
        ax.set_xticks([-0.04, 0, 0.04])
        ax.set_xlim(-0.05, 0.075)
    axes[0].text(-0.75, 0.5, "Held-out region (C3)", transform=axes[0].transAxes, rotation=90, ha="center",
                 va="center", fontsize=7)
    h = [Line2D([], [], ls="none", marker=fs.marker(b), ms=4, mfc=fs.colour(b), mec="white", mew=0.4,
                label=f"HapGAM − {fs.label(b)}") for b, _ in comps]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.56, 1.0))
    fs.save(fig, FIGURES / "figS1_C3_regions")


if __name__ == "__main__":
    {"fig2": fig2, "fig3": fig3, "fig4": fig4, "figS1": figS1}[sys.argv[1]]()
