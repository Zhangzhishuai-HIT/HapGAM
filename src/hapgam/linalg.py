"""Memory-lean genotype linear algebra.

Genotypes stay as int8 tensors (one byte per call); centred cross-products are accumulated
over column chunks in float64, so no float copy of the full n x p matrix is ever made.
"""

from __future__ import annotations

import torch

CHUNK = 4096


def col_stats(X: torch.Tensor, chunk: int = CHUNK):
    """Column means and standard deviations (float32, shape [1, p]) of an int8 genotype matrix."""
    p = X.shape[1]
    mu = torch.empty(1, p, dtype=torch.float32, device=X.device)
    sd = torch.empty(1, p, dtype=torch.float32, device=X.device)
    for c in range(0, p, chunk):
        A = X[:, c:c + chunk].float()
        mu[:, c:c + chunk] = A.mean(0, keepdim=True)
        sd[:, c:c + chunk] = A.std(0, keepdim=True)
    return mu, sd


def centered_cross(Xa: torch.Tensor, Xb: torch.Tensor | None, mu: torch.Tensor, chunk: int = CHUNK,
                   scale: float | None = None) -> torch.Tensor:
    """(Xa - mu)(Xb - mu)^T / scale (default scale = p), float64; Xb=None means Xb = Xa."""
    p = Xa.shape[1]
    nb = Xa.shape[0] if Xb is None else Xb.shape[0]
    out = torch.zeros(Xa.shape[0], nb, dtype=torch.float64, device=Xa.device)
    for c in range(0, p, chunk):
        A = Xa[:, c:c + chunk].float() - mu[:, c:c + chunk]
        B = A if Xb is None else Xb[:, c:c + chunk].float() - mu[:, c:c + chunk]
        out += (A @ B.T).double()
    return out / (p if scale is None else scale)


def centered_tmatvec(X: torch.Tensor, v: torch.Tensor, mu: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """(X - mu)^T v  (float32 vector of length p)."""
    p = X.shape[1]
    out = torch.empty(p, dtype=torch.float32, device=X.device)
    vf = v.float()
    for c in range(0, p, chunk):
        out[c:c + chunk] = (X[:, c:c + chunk].float() - mu[:, c:c + chunk]).T @ vf
    return out


def centered_cols(X: torch.Tensor, cols: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor | None = None):
    """Centred (optionally standardised) float32 copy of selected columns only."""
    Z = X[:, cols].float() - mu[:, cols]
    if sd is not None:
        Z = Z / sd[:, cols].clamp_min(1e-3)
    return Z


def centered_sqnorm(X: torch.Tensor, mu: torch.Tensor, chunk: int = CHUNK, scale: float | None = None):
    """Row-wise ||x_i - mu||^2 / scale (default p), float64."""
    p = X.shape[1]
    out = torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
    for c in range(0, p, chunk):
        A = X[:, c:c + chunk].float() - mu[:, c:c + chunk]
        out += (A * A).sum(1).double()
    return out / (p if scale is None else scale)


def kernel_pcs(Xtr: torch.Tensor, Xte: torch.Tensor, mu: torch.Tensor, k: int):
    """Top-k principal-component scores of centred genotypes via the n x n kernel.

    Training scores U_k sqrt(lambda_k); test genotypes are projected with the training loadings
    (equivalent to an SVD of the centred training matrix, up to signs).
    """
    p = Xtr.shape[1]
    G = centered_cross(Xtr, None, mu, scale=1.0)            # Z Z^T
    lam, U = torch.linalg.eigh(G)
    lam, U = lam.flip(0)[:k].clamp_min(1e-9), U.flip(1)[:, :k]
    tr = (U * lam.sqrt()).float()
    te = (centered_cross(Xte, Xtr, mu, scale=1.0) @ (U / lam.sqrt())).float()
    return tr, te
