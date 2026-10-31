"""Kernel mixed models fitted by REML (spectral / FaST-LMM formulation) in torch.

y = Xb + g + e,  g ~ N(0, s2g K),  e ~ N(0, s2e I),  delta = s2e / s2g.
With K = U diag(s) U^T the REML likelihood is a 1-D function of log(delta).
"""

from __future__ import annotations

import math

import numpy as np
import torch


def vanraden_kernel(Z_a: torch.Tensor, Z_b: torch.Tensor | None = None, scale: float | None = None):
    """Linear kernel Z_a Z_b^T / scale for centred genotype matrices."""
    if Z_b is None:
        Z_b = Z_a
    if scale is None:
        scale = Z_a.shape[1]
    return (Z_a @ Z_b.T) / scale


def center_genotypes(X_train: torch.Tensor, X_other=None):
    """Centre (and return allele means) using training allele frequencies only."""
    mu = X_train.mean(0, keepdim=True)
    Zt = X_train - mu
    if X_other is None:
        return Zt, mu
    return Zt, X_other - mu, mu


class SpectralREML:
    """Single-kernel REML-BLUP with fixed-effect design (intercept by default)."""

    def __init__(self, log_delta_grid=(-12.0, 12.0, 97)):
        self.grid = log_delta_grid

    def fit(self, K: torch.Tensor, y: torch.Tensor, C: torch.Tensor | None = None, eig=None):
        dt = torch.float64
        K = K.to(dt)
        y = y.to(dt).reshape(-1)
        n = y.shape[0]
        X = torch.ones(n, 1, dtype=dt, device=y.device)
        if C is not None:
            X = torch.cat([X, C.to(dt)], 1)
        if eig is None:
            s, U = torch.linalg.eigh(K)
        else:
            s, U = eig
        s = s.clamp_min(0)
        self.s, self.U, self.X = s, U, X
        Uy = U.T @ y
        UX = U.T @ X
        q = X.shape[1]
        logdet_xtx = torch.logdet(X.T @ X)

        def reml(ld: float):
            d = s + math.exp(ld)
            w = 1.0 / d
            A = (UX * w[:, None]).T @ UX
            b = (UX * w[:, None]).T @ Uy
            beta = torch.linalg.solve(A, b)
            r = Uy - UX @ beta
            R = (r * r * w).sum()
            s2g = R / (n - q)
            ll = -0.5 * ((n - q) * torch.log(2 * math.pi * s2g) + torch.log(d).sum()
                         + torch.logdet(A) - logdet_xtx + (n - q))
            return float(ll), beta, float(s2g)

        lo, hi, m = self.grid
        lds = np.linspace(lo, hi, int(m))
        lls = [reml(ld)[0] for ld in lds]
        i = int(np.nanargmax(lls))
        a, b = lds[max(i - 1, 0)], lds[min(i + 1, len(lds) - 1)]
        # golden-section refinement
        g = (math.sqrt(5) - 1) / 2
        c, d_ = b - g * (b - a), a + g * (b - a)
        fc, fd = reml(c)[0], reml(d_)[0]
        for _ in range(40):
            if fc > fd:
                b, d_, fd = d_, c, fc
                c = b - g * (b - a)
                fc = reml(c)[0]
            else:
                a, c, fc = c, d_, fd
                d_ = a + g * (b - a)
                fd = reml(d_)[0]
        ld = (a + b) / 2
        ll, beta, s2g = reml(ld)
        self.log_delta, self.delta, self.beta, self.s2g = ld, math.exp(ld), beta, s2g
        self.s2e = s2g * self.delta
        self.loglik = ll
        resid = y - X @ beta
        self.alpha = U @ ((U.T @ resid) / (s + self.delta))  # (K + delta I)^-1 (y - Xb)
        self.h2 = 1.0 / (1.0 + self.delta * float(len(s)) / float(s.sum()))  # h2 on mean-diag scale
        return self

    def predict(self, K_test_train: torch.Tensor, C_test: torch.Tensor | None = None):
        K_test_train = K_test_train.to(torch.float64)
        m = K_test_train.shape[0]
        X = torch.ones(m, 1, dtype=torch.float64, device=K_test_train.device)
        if C_test is not None:
            X = torch.cat([X, C_test.to(torch.float64)], 1)
        return X @ self.beta + K_test_train @ self.alpha

    def fitted(self):
        """In-sample BLUP fitted values (fixed + random)."""
        K_alpha = self.U @ (self.s * (self.U.T @ self.alpha))
        return self.X @ self.beta + K_alpha


def select_epi_weight(Ka: torch.Tensor, Kaa: torch.Tensor, y: torch.Tensor,
                      weights=(0.0, 0.25, 0.5, 0.75, 1.0), n_sub: int = 4000, seed: int = 0):
    """REML choice of w in (1-w)K_A + wK_AA on a fixed random subsample of individuals.

    Used identically by the EGBLUP baseline and HapGAM-E so that HapGAM-E with M = 0 is
    exactly EGBLUP.  Returns (w, {w: loglik}).
    """
    n = Ka.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(n, n_sub)].to(Ka.device)
    A, AA, ys = Ka[idx][:, idx], Kaa[idx][:, idx], y[idx]
    table = {}
    for w in weights:
        table[w] = SpectralREML().fit((1 - w) * A + w * AA, ys).loglik
    return max(table, key=table.get), table
