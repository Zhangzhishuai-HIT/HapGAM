"""Figure 1: the HapGAM model (a) and the pre-registered evaluation design (b).

Drawn in millimetre coordinates at the final printed size (174 mm wide).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import FIGURES  # noqa: E402

W, H = 174, 128   # drawing coordinates (mm)
Y0 = 10           # the bottom 10 mm of the drawing area are not shown (figure height H - Y0)
BLUE, BLUE_L, BLUE_XL = fs.colour("hapgam_e"), "#a9c4e8", "#e3ecf7"
TEAL, TEAL_L = fs.colour("egblup"), "#cfeee3"
INK, INK2, MUTED = fs.INK, fs.INK2, fs.MUTED
TRAIN, TEST1, TEST2 = "#d9dee4", "#7d93ab", "#33475b"
RNG = np.random.default_rng(7)


def T(ax, x, y, s, size=6.5, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("va", "center")
    kw.setdefault("ha", "left")
    return ax.text(x, y, s, fontsize=size, **kw)


def arrow(ax, p0, p1, color=INK2, lw=0.6, rad=0.0, ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=4))


def rbox(ax, x, y, w, h, fc="white", ec=MUTED, lw=0.5, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def genotype_glyph(ax, x0, y0, ncol=24, nrow=14, cell=1.15, gap=0.55, block=4, hi=(1, 4)):
    """Accessions x SNPs; columns grouped into blocks, each block built from a few haplotypes."""
    pal = {0: "#e6ebf1", 2: "#8a9bb1"}
    nb = ncol // block
    for b in range(nb):
        haps = RNG.integers(0, 2, size=(RNG.integers(2, 5), block)) * 2
        carrier = RNG.integers(0, len(haps), size=nrow)
        bx = x0 + b * (block * cell + gap)
        for r in range(nrow):
            for c in range(block):
                ax.add_patch(Rectangle((bx + c * cell, y0 + r * cell), cell, cell, fc=pal[int(haps[carrier[r], c])],
                                       ec="white", lw=0.25, zorder=2))
        if b in hi:
            ax.add_patch(Rectangle((bx - 0.25, y0 - 0.25), block * cell + 0.5, nrow * cell + 0.5, fc="none", ec=BLUE,
                                   lw=0.9, zorder=3))
    return x0 + nb * (block * cell + gap) - gap, y0 + nrow * cell


def kinship_glyph(ax, x0, y0, n=12, cell=1.15):
    grp = np.repeat([0, 1, 2], [5, 4, 3])
    for i in range(n):
        for j in range(n):
            v = 1.0 if i == j else (0.55 if grp[i] == grp[j] else 0.12) + RNG.uniform(-0.06, 0.06)
            col = plt.matplotlib.colors.to_hex(np.array([1, 1, 1]) * (1 - v) + np.array(
                plt.matplotlib.colors.to_rgb(TEAL)) * v)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell), cell, cell, fc=col, ec="white", lw=0.2,
                                   zorder=2))
    return x0 + n * cell, y0 + n * cell


def screening_glyph(ax, x0, y0, w=30, h=11, nbar=46, top=5):
    v = RNG.gamma(0.7, 1.0, nbar)
    idx = RNG.choice(nbar, top, replace=False)
    v[idx] = v.max() * RNG.uniform(1.6, 2.6, top)
    v = v / v.max() * h
    bw = w / nbar
    for k in range(nbar):
        ax.add_patch(Rectangle((x0 + k * bw + 0.08, y0), bw - 0.16, v[k], fc=BLUE if k in idx else "#c4ccd6", lw=0,
                               zorder=2))
    ax.plot([x0, x0 + w], [y0, y0], color=INK2, lw=0.5, zorder=3)
    thr = np.sort(v)[-top] - 0.3
    ax.plot([x0, x0 + w], [thr, thr] if False else [y0 + thr, y0 + thr], color=BLUE, lw=0.5, ls=(0, (2, 1.5)),
            zorder=3)
    T(ax, x0 + w + 0.8, y0 + thr, "top $M$", size=6.0, color=BLUE)


def net_glyph(ax, x0, yc, w=12, h=6.0):
    xs = [x0, x0 + w / 2, x0 + w]
    layers = [np.linspace(yc - h / 2, yc + h / 2, 4), np.linspace(yc - h / 2 - 0.4, yc + h / 2 + 0.4, 5), [yc]]
    for a, b, xa, xb in ((layers[0], layers[1], xs[0], xs[1]), (layers[1], layers[2], xs[1], xs[2])):
        for ya in a:
            for yb in b:
                ax.plot([xa, xb], [ya, yb], color="#b5c4d6", lw=0.3, zorder=2)
    for k, (xx, ys) in enumerate(zip(xs, layers)):
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 0.55 if k < 2 else 0.75, fc=["#8a9bb1", "white", BLUE][k],
                                ec=["#8a9bb1", BLUE, BLUE][k], lw=0.5, zorder=3))


def panel_a(ax):
    fs.panel_label(ax, 0.5, H - 0.6, "a", transform=ax.transData)
    T(ax, 5.2, H - 2.1, "HapGAM: a genomic kernel model plus sparse haplotype-block networks", size=7.5,
      fontweight="bold")
    top = H - 6
    # genotypes
    T(ax, 4, top - 3, "SNP genotypes", size=6.8, fontweight="bold")
    T(ax, 4, top - 6, "$n$ accessions × $p$ SNPs", size=6.3, color=INK2)
    gx1, gy1 = genotype_glyph(ax, 4, top - 27.5)
    T(ax, 4, top - 30.5, "consecutive windows of 8, 16", size=6.0, color=INK2)
    T(ax, 4, top - 33.2, "or 32 SNPs (blocks)", size=6.0, color=INK2)
    # kinship branch
    kx0, ky0 = 47, top - 16
    T(ax, kx0, top - 3, "Genomic random effect", size=6.8, fontweight="bold")
    kx1, ky1 = kinship_glyph(ax, kx0, ky0 - 1.5, cell=1.08)
    T(ax, kx1 + 2.0, ky0 + 9.3, r"$\mathbf{g} \sim N(\mathbf{0},\ \sigma_{\mathrm{g}}^2\,\mathbf{K})$", size=6.8)
    T(ax, kx1 + 2.0, ky0 + 5.4, r"$\mathbf{K}=(1-w)\mathbf{K}_{\mathrm{A}} + w\mathbf{K}_{\mathrm{AA}}$", size=6.3, color=INK2)
    T(ax, kx1 + 2.0, ky0 + 1.9, r"$w=0$: GBLUP;  $w>0$: EGBLUP", size=6.3, color=INK2)
    T(ax, kx1 + 2.0, ky0 - 1.4, r"$w,\ \delta=\sigma_{\mathrm{e}}^2/\sigma_{\mathrm{g}}^2$ by REML", size=6.3, color=INK2)
    arrow(ax, (gx1 + 1.2, gy1 - 3), (kx0 - 1.2, ky0 + 5.5))
    # screening branch
    sx0, sy0 = 47, top - 47
    T(ax, sx0, top - 28.5, "Block screening (training data only)", size=6.8, fontweight="bold")
    screening_glyph(ax, sx0, sy0 + 2.5, w=31, h=11)
    T(ax, sx0, sy0 - 0.6, r"rank blocks by Var($\mathbf{Z}_b\boldsymbol{\beta}_b$) from the REML-BLUP",
      size=6.0, color=INK2)
    arrow(ax, (gx1 + 1.2, top - 22), (sx0 - 1.2, sy0 + 9))
    # networks
    nx0 = 99
    T(ax, nx0 - 7, top - 28.5, "$M$ block networks", size=6.8, fontweight="bold")
    T(ax, nx0 - 7, top - 31.5, "one hidden layer,", size=6.0, color=INK2)
    T(ax, nx0 - 7, top - 34.2, "8 GELU units", size=6.0, color=INK2)
    ys = [top - 39.0, top - 44.8, top - 52.2]
    for yy, lab in zip(ys, [r"$f_1(\mathbf{z}_1)$", r"$f_2(\mathbf{z}_2)$", r"$f_M(\mathbf{z}_M)$"]):
        net_glyph(ax, nx0, yy, w=10.5, h=3.8)
        T(ax, nx0 - 1.6, yy, lab, size=6.3, ha="right")
    for dy in (-1.2, 0, 1.2):
        ax.add_patch(Circle((nx0 + 5.25, (ys[1] + ys[2]) / 2 + dy), 0.28, fc=INK2, ec="none", zorder=3))
    arrow(ax, (sx0 + 31.6, sy0 + 6.5), (nx0 - 9.8, ys[1]))
    # combination
    cx, cy = 127.5, top - 20
    ax.add_patch(Circle((cx, cy), 2.6, fc="white", ec=INK2, lw=0.6, zorder=3))
    T(ax, cx, cy + 0.15, "+", size=9, ha="center", color=INK)
    arrow(ax, (110, ky0 + 6.5), (cx - 1.9, cy + 1.9), rad=-0.18)
    T(ax, 119.2, ky0 + 5.6, r"$\hat{g}$", size=6.8)
    arrow(ax, (nx0 + 12.0, ys[0] + 0.6), (cx - 1.4, cy - 2.25), rad=-0.22)
    T(ax, 120.6, top - 29.6, r"$\Sigma_b\, f_b(\mathbf{z}_b)$", size=6.5)
    # prediction box
    rbox(ax, 134.5, cy - 6, 37.5, 12, fc=BLUE_XL, ec=BLUE, lw=0.6)
    arrow(ax, (cx + 2.7, cy), (134.1, cy))
    T(ax, 136.5, cy + 2.6, "Prediction", size=6.8, fontweight="bold")
    T(ax, 136.5, cy - 2.2, r"$\hat{y}=\hat{\mu}+\hat{g}+\Sigma_b\, f_b(\mathbf{z}_b)$", size=6.8)
    # training and tuning
    rbox(ax, 118.0, top - 45.5, 54.0, 13.5, fc="#f6f6f4", ec="#c8c8c8")
    T(ax, 119.8, top - 35.0, "Training: genomic effect profiled out", size=6.5, fontweight="bold")
    T(ax, 119.8, top - 39.2,
      r"$L(\mathbf{F})=(\mathbf{y}-\mu\mathbf{1}-\mathbf{F})^{\mathrm{T}}(\mathbf{K}/\delta+\mathbf{I})^{-1}(\mathbf{y}-\mu\mathbf{1}-\mathbf{F})$", size=6.3)
    T(ax, 119.8, top - 43.0, "networks fit mainly what relatedness leaves", size=6.0, color=INK2)
    rbox(ax, 118.0, top - 57.6, 54.0, 10.6, fc="#f6f6f4", ec="#c8c8c8")
    T(ax, 119.8, top - 49.5, "Tuning: inner cross-validation", size=6.5, fontweight="bold")
    T(ax, 119.8, top - 52.8, r"block size, weight decay, $M\in\{0,8,16,32,64\}$", size=6.1, color=INK2)
    T(ax, 119.8, top - 55.6, r"$M=0$ recovers GBLUP or EGBLUP", size=6.1, color=INK2)


def mini(ax, x, y, kind):
    """Small train -> test glyph for a scenario (x, y = left, centre)."""
    if kind == "pair":
        ax.add_patch(Rectangle((x, y - 1.2), 5.5, 2.4, fc=TRAIN, ec="none"))
        arrow(ax, (x + 6.0, y), (x + 8.4, y), lw=0.5, ms=4)
        ax.add_patch(Rectangle((x + 8.8, y - 1.2), 2.6, 2.4, fc=TEST2, ec="none"))
    else:
        k = {"regions": 7, "mg": 5, "families": 5}[kind]
        wbox = 11.4 / k
        for i in range(k):
            ax.add_patch(Rectangle((x + i * wbox + 0.15, y - 1.2), wbox - 0.3, 2.4,
                                   fc=TEST2 if i == 2 else TRAIN, ec="none"))


def panel_b(ax):
    y0 = 61.0
    fs.panel_label(ax, 0.5, y0 + 0.6, "b", transform=ax.transData)
    T(ax, 5.2, y0 - 0.9, "Evaluation design", size=7.5, fontweight="bold")
    # GRIN bar
    T(ax, 4, y0 - 6.3, "USDA Soybean Germplasm Collection (GRIN)", size=7, fontweight="bold")
    T(ax, 4, y0 - 9.2, "17,407 accessions × 38,016 SNPs; nine quantitative and four Mendelian traits", size=6.2,
      color=INK2)
    bx, by, bw, bh = 4, y0 - 20.0, 76, 7.6
    parts = [("Training set", 12301, TRAIN, INK), ("Random test set", 2175, TEST1, "white"),
             ("Japanese test set", 2931, TEST2, "white")]
    x, tot = bx, 17407
    for name, n, fc, tc in parts:
        w = bw * n / tot
        ax.add_patch(Rectangle((x, by), w - 0.35, bh, fc=fc, ec="none"))
        if name == "Training set":
            T(ax, x + w / 2, by + bh / 2, f"Training set\n{n:,}", size=6.2, ha="center", color=tc,
              linespacing=1.15)
        else:
            T(ax, x + (w - 0.35) / 2, by + bh / 2, name.replace(" ", "\n", 1) + f"\n{n:,}", size=6.0, ha="center",
              color=tc, linespacing=1.05)
        x += w
    # regions
    T(ax, 4, by - 3.6, "Accessions by region of origin", size=6.5, fontweight="bold")
    regions = [("China", 6204), ("Korea", 3632), ("Japan", 2931), ("S/SE Asia", 1408), ("Americas", 1108),
               ("Europe", 1022), ("Russia/C. Asia", 745), ("Other/unknown", 357)]
    rx, ry, rh, scale = 22.5, by - 6.9, 2.15, 34 / 6204
    for k, (name, n) in enumerate(regions):
        yy = ry - k * rh
        T(ax, rx - 1.2, yy, name, size=6.0, ha="right", color=INK)
        ax.add_patch(Rectangle((rx, yy - 0.72), n * scale, 1.44, fc=TEST2 if name == "Japan" else "#a7b3c1",
                               ec="none"))
        T(ax, rx + n * scale + 0.9, yy, f"{n:,}", size=6.0, color=INK2)
    # scenarios
    sx = 86
    T(ax, sx, y0 - 6.3, "Evaluation scenarios", size=7, fontweight="bold")
    rows = [("Within the collection", None, None, None),
            ("C1", "pair", "training set → random test set", False),
            ("C2", "pair", "non-Japanese → Japanese accessions", True),
            ("C3", "regions", "leave one of seven regions out", False),
            ("Across-population prediction (accuracy within families or trials)", None, None, None),
            ("C4", "pair", "GRIN → SoyNAM (39 families)", False),
            ("C5", "pair", "GRIN → NUST elite lines (2,511)", False)]
    yy = y0 - 10.4
    for tag, kind, text, primary in rows:
        if kind is None:
            yy -= 0.6
            T(ax, sx, yy, tag, size=6.2, color=INK2, style="italic")
            yy -= 4.0
            continue
        rbox(ax, sx, yy - 1.45, 6.6, 2.9, fc=BLUE if primary else "white", ec=BLUE if primary else INK2, lw=0.5, r=0.6)
        T(ax, sx + 3.3, yy, tag, size=6.2, ha="center", fontweight="bold", color="white" if primary else INK)
        mini(ax, sx + 8.6, yy, kind)
        T(ax, sx + 22, yy, text, size=6.3, fontweight="bold" if primary else "normal")
        yy -= 4.9

def main():
    fig = plt.figure(figsize=(W * fs.MM, (H - Y0) * fs.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")
    panel_a(ax)
    ax.plot([3, W - 2], [63.6, 63.6], color="#d9d9d9", lw=0.5)
    panel_b(ax)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fs.save(fig, FIGURES / "fig1_schematic")


if __name__ == "__main__":
    main()
