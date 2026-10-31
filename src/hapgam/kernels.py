"""Genomic kernels for REML baselines (computed on GPU with training-only centring)."""

from __future__ import annotations

import numpy as np
import torch


def additive(Ztr: torch.Tensor, Zte: torch.Tensor):
    p = Ztr.shape[1]
    return (Ztr @ Ztr.T / p).double(), (Zte @ Ztr.T / p).double()


def _sqdist(A: torch.Tensor, B: torch.Tensor):
    a = (A * A).sum(1, keepdim=True)
    b = (B * B).sum(1, keepdim=True)
    return (a + b.T - 2 * A @ B.T).clamp_min(0)


def gaussian(Ztr: torch.Tensor, Zte: torch.Tensor, h: float):
    """exp(-d^2 / (h * median d^2)); median taken over training pairs."""
    Dtr = _sqdist(Ztr, Ztr)
    med = torch.median(Dtr[torch.triu_indices(len(Ztr), len(Ztr), 1, device=Dtr.device).unbind()][:200000])
    Ktr = torch.exp(-Dtr / (h * med)).double()
    del Dtr
    Kte = torch.exp(-_sqdist(Zte, Ztr) / (h * med)).double()
    return Ktr, Kte


def hadamard_epistatic(Ka_tr: torch.Tensor, Ka_te: torch.Tensor):
    """Additive-by-additive kernel K_A o K_A rescaled to unit mean diagonal (training)."""
    d = torch.diagonal(Ka_tr).mean()
    Ktr = Ka_tr * Ka_tr
    f = d / torch.diagonal(Ktr).mean()
    Ktr.mul_(f)
    Kte = (Ka_te * Ka_te).mul_(f)
    return Ktr, Kte


def haplotype_onehot(X: np.ndarray, block_idx: np.ndarray, min_freq: float = 0.01, train_mask=None):
    """One-hot haplotype matrix (individuals x haplotype alleles) per block.

    Haplotypes are the distinct allele strings within a block; alleles with training
    frequency < min_freq are pooled into one 'rare' column per block.  Columns are defined on
    training individuals only (``train_mask``).
    """
    n = X.shape[0]
    if train_mask is None:
        train_mask = np.ones(n, bool)
    cols = []
    for row in block_idx:
        j = row[row >= 0]
        sub = X[:, j].astype(np.int8)
        # hash each haplotype string to an integer code
        weights = (3 ** np.arange(len(j), dtype=np.int64))
        code = (sub.astype(np.int64) * weights).sum(1)
        uniq, cnt = np.unique(code[train_mask], return_counts=True)
        common = uniq[cnt >= max(2, min_freq * train_mask.sum())]
        lookup = {c: i for i, c in enumerate(common)}
        lab = np.array([lookup.get(c, len(common)) for c in code])
        k = len(common) + 1
        oh = np.zeros((n, k), np.float32)
        oh[np.arange(n), lab] = 1.0
        if oh[train_mask, -1].sum() == 0:
            oh = oh[:, :-1]
        cols.append(oh)
    return np.concatenate(cols, 1)
