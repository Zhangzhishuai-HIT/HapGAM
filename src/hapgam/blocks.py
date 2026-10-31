"""Haplotype block definitions (phenotype-free).

A block scheme is an integer array ``block_of_snp`` (length p, values 0..B-1, contiguous
and never crossing chromosomes).  Schemes:

  fixed<k>    consecutive windows of k SNPs (a short chromosome tail is merged into the
              previous window if it has fewer than k/2 SNPs)
  ld<tau>     greedy LD segmentation: a SNP joins the current block while its mean r^2
              with the block's SNPs is >= tau; blocks are capped at ``max_size`` SNPs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_blocks(chrom: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(chrom), dtype=np.int64)
    b = 0
    for c in pd.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        lab = np.arange(len(idx)) // k
        tail = len(idx) % k
        if 0 < tail < k / 2 and lab.max() > 0:
            lab[lab == lab.max()] = lab.max() - 1
        out[idx] = lab + b
        b = out[idx].max() + 1
    return out


def _r2_with_block(G: np.ndarray, j: int, members: list[int]) -> float:
    x = G[:, j]
    Y = G[:, members]
    xc = x - x.mean()
    Yc = Y - Y.mean(0)
    num = xc @ Yc
    den = np.sqrt((xc @ xc) * (Yc * Yc).sum(0)) + 1e-12
    return float(np.mean((num / den) ** 2))


def ld_blocks(G: np.ndarray, chrom: np.ndarray, tau: float = 0.5, max_size: int = 32,
              n_sub: int = 4000, seed: int = 0) -> np.ndarray:
    """Greedy LD segmentation on a random subset of individuals (rows of G)."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(G.shape[0], size=min(n_sub, G.shape[0]), replace=False)
    Gs = np.asarray(G[rows], dtype=np.float32)
    out = np.empty(len(chrom), dtype=np.int64)
    b = -1
    prev_c = None
    members: list[int] = []
    for j in range(len(chrom)):
        if chrom[j] != prev_c or len(members) >= max_size or _r2_with_block(Gs, j, members) < tau:
            b += 1
            members = []
            prev_c = chrom[j]
        members.append(j)
        out[j] = b
    return out


def make_scheme(name: str, snps: pd.DataFrame, G=None) -> np.ndarray:
    chrom = snps["chrom"].to_numpy()
    if name.startswith("fixed"):
        return fixed_blocks(chrom, int(name[5:]))
    if name.startswith("ld"):
        tau = float(name[2:]) / 100.0 if len(name) > 2 else 0.5
        return ld_blocks(G, chrom, tau=tau)
    raise ValueError(name)


def block_index(block_of_snp: np.ndarray):
    """Return (B, k_max, index[B, k_max] with -1 padding)."""
    B = int(block_of_snp.max()) + 1
    sizes = np.bincount(block_of_snp, minlength=B)
    kmax = int(sizes.max())
    idx = -np.ones((B, kmax), dtype=np.int64)
    pos = np.zeros(B, dtype=np.int64)
    for j, b in enumerate(block_of_snp):
        idx[b, pos[b]] = j
        pos[b] += 1
    return B, kmax, idx
