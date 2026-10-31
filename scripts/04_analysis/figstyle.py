"""Shared figure style: journal print sizes, Times New Roman and one fixed colour per model.

Figures are drawn at their final printed size (Springer double column 174 mm, single column 84 mm)
with 7-pt text and 0.5-pt axes, saved as vector PDF (fonts embedded) and as 600-dpi PNG.
The key-model colours (HapGAM-E, EGBLUP, BayesB, LightGBM, GBLUP) were checked for separation
under simulated protanopia/deuteranopia (worst pair OKLab dE >= 11); within a model family the
hue is shared and the marker shape differs, so identity never depends on colour alone.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
W2 = 174 * MM   # double column
W1 = 84 * MM    # single column
FONT = "Times New Roman"
# register Times New Roman when it is installed outside matplotlib's font cache (e.g. in ~/.local/share/fonts)
for _d in ("~/.local/share/fonts", "~/.fonts"):
    for _f in glob.glob(os.path.join(os.path.expanduser(_d), "**", "[Tt]imes*.[Tt][Tt][Ff]"), recursive=True):
        font_manager.fontManager.addfont(_f)

INK, INK2, MUTED, LIGHT, GRID, BAND = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#bdbdbd", "#e8e8e8", "#f4f4f2"

# model -> (label, colour, marker); hue = model family, every model has its own marker shape
MODEL = {
    "hapgam_e": ("HapGAM", "#1d4f91", "o"),
    "hapgam_e_linear": ("HapGAM-linear", "#5b8fd0", "8"),
    "hapgam": ("HapGAM-A", "#7fa7dc", "H"),
    "gblup": ("GBLUP", "#6b6b6b", "s"),
    "hapgblup": ("Haplotype GBLUP", "#a9a9a9", "X"),
    "egblup": ("EGBLUP", "#1b9e77", "D"),
    "rkhs": ("RKHS", "#7cc8ae", "h"),
    "bayesb": ("BayesB", "#8e6bbf", "P"),
    "lightgbm": ("LightGBM", "#e66101", "^"),
    "xgboost": ("XGBoost", "#f5a55c", "v"),
    "mlp": ("MLP", "#c2477a", "p"),
    "dnngp": ("DNNGP", "#8c510a", "<"),
    "ens_hapgam_e_lightgbm": ("HapGAM + LightGBM", "#111111", "*"),
    "ens_gblup_lightgbm_bayesb": ("GBLUP + LightGBM + BayesB", "#7a7a7a", "*"),
}

TRAIT_LABEL = {
    "maturity_doy": "Maturity date", "flowering_doy": "Flowering date", "height_cm": "Plant height",
    "seed_weight_g": "100-seed weight", "yield_mg_ha": "Seed yield", "protein_pct": "Protein",
    "oil_pct": "Oil", "lodging_score": "Lodging", "maturity_group": "Maturity group",
}
TRAIT_ORDER = list(TRAIT_LABEL)
PRIMARY = ["maturity_doy", "flowering_doy", "height_cm", "seed_weight_g"]

# diverging map for differences: blue = HapGAM-E (first term) larger, orange = second term larger
DIVERGING = LinearSegmentedColormap.from_list(
    "hapgam_div", ["#8c3a00", "#e66101", "#f9c493", "#f7f7f5", "#a9c4e8", "#3d73bd", "#173f78"])

RC = {
    "font.family": [FONT, "STIXGeneral", "DejaVu Serif"], "font.size": 7,   # later fonts fill glyphs FONT lacks (∝)
    "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.title_fontsize": 6.5, "figure.titlesize": 7,
    "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2.5, "axes.titlepad": 3,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "text.color": INK,
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0,
    "legend.handlelength": 1.2,
    "lines.linewidth": 0.8, "lines.markersize": 3.5, "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
    "grid.color": GRID, "grid.linewidth": 0.4,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.default": "it",
    "mathtext.fallback": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.pad_inches": 0.02,
}
plt.rcParams.update(RC)


def label(model):
    return MODEL.get(model, (model, MUTED, "o"))[0]


def colour(model):
    return MODEL.get(model, (model, MUTED, "o"))[1]


def marker(model):
    return MODEL.get(model, (model, MUTED, "o"))[2]


def panel_label(fig_or_ax, x, y, letter, transform=None):
    """Bold lowercase panel letter (8.5 pt) at figure coordinates (or `transform`)."""
    tgt = fig_or_ax
    tr = transform if transform is not None else (tgt.transFigure if hasattr(tgt, "transFigure") else tgt.transAxes)
    tgt.text(x, y, letter, transform=tr, fontsize=8.5, fontweight="bold", va="top", ha="left", color=INK)


def rich_text(ax, x, y, parts, **common):
    """Draw [(text, kwargs), ...] side by side from (x, y) in axes fraction (DPI-independent offsets)."""
    r = ax.figure.canvas.get_renderer()
    off = 0.0
    for s, kw in parts:
        t = ax.annotate(s, xy=(x, y), xycoords="axes fraction", xytext=(off, common.get("dy", 0)),
                        textcoords="offset points", annotation_clip=False,
                        **{k: v for k, v in {**common, **kw}.items() if k != "dy"})
        off += t.get_window_extent(r).width * 72 / ax.figure.dpi
    return off


def letter_at(ax, letter, dx_mm=-9.0, dy_pt=11.0):
    """Panel letter at a fixed offset (mm, points) from the top-left corner of an axes."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx_mm / 25.4 * 72, dy_pt),
                textcoords="offset points", fontsize=8.5, fontweight="bold", va="baseline", ha="left",
                annotation_clip=False)


def fmt_diff(v, digits=3):
    """+.012 / −.004 with a true minus sign."""
    s = f"{v:+.{digits}f}"
    return s.replace("+0.", "+.").replace("-0.", "−.")


def minus(s):
    return str(s).replace("-", "−")


def save(fig, path, png=True):
    """Save vector PDF and 600-dpi PNG next to each other (path without extension)."""
    path = str(path)
    fig.savefig(f"{path}.pdf")
    if png:
        fig.savefig(f"{path}.png", dpi=600)
    plt.close(fig)
    print(f"{path}.png")
