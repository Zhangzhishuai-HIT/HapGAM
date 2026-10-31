"""HapGAM: genome-wide REML-BLUP plus gated haplotype-block subnetworks.

    y = mu + g + sum_{b in S} g_b(z_b) + e,     g ~ N(0, s2g K),  e ~ N(0, s2e I)

Block terms F(x) = sum_b g_b(z_b) are fitted by generalised least squares under the
genomic covariance V = K/delta + I, which is the profiled form of fitting them jointly with
ridge (rrBLUP) marker effects.  With all block outputs zero the model is exactly
REML-GBLUP/rrBLUP.

Block selection is hierarchical: (1) blocks are screened inside the training data by the
variance of their REML-rrBLUP contribution (and, optionally, by a marginal nonlinear fit to the
GLS residual); (2) only the top-M blocks receive a subnetwork; (3) subnetwork output weights
are zero-initialised and shrunk by weight decay, and training stops early on an inner
validation set.  M = 0 recovers GBLUP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .linalg import centered_cols, centered_cross, centered_tmatvec, col_stats
from .reml import SpectralREML, select_epi_weight


@dataclass
class HapGAMConfig:
    top_m: int = 32              # number of screened blocks receiving a subnetwork (0 = GBLUP)
    hidden: int = 8
    weight_decay: float = 1e-3   # L2 on block output weights (standardised y units)
    wd_first: float = 1e-4       # L2 on first-layer weights
    lr: float = 3e-3
    steps: int = 2000
    eval_every: int = 25
    patience: int = 8            # evaluations without improvement before early stop (with val)
    refit_delta: bool = True     # re-estimate the REML variance ratio on y - F after training
    gls: bool = True             # GLS loss under the genomic covariance (False: OLS)
    screen: str = "blup"         # block screening score: "blup" (additive) or "blup+nl"
    backbone: str = "A"          # genomic covariance: "A" (GBLUP) or "AE" (A + AxA, REML weight)
    seed: int = 0
    extra: dict = field(default_factory=dict)


class BlockMLP(nn.Module):
    """All selected block subnetworks evaluated together with einsum.

    Zb: [n, M, k] standardised dosages (zero padded).  Output: contributions [n, M].
    """

    def __init__(self, M: int, k: int, hidden: int, mask: torch.Tensor, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(M, k, hidden, generator=g) * math.sqrt(2.0 / k))
        self.b1 = nn.Parameter(torch.randn(M, hidden, generator=g) * 0.1)
        self.w2 = nn.Parameter(torch.zeros(M, hidden))   # zero init -> model == GBLUP
        self.register_buffer("mask", mask)              # [M, k] 1 for real SNPs

    def features(self, Zb: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("nmk,mkh->nmh", Zb, self.W1 * self.mask[:, :, None]) + self.b1
        return torch.nn.functional.gelu(h)

    def forward(self, Zb: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
        h = self.features(Zb)
        c = h.mean(0, keepdim=True) if center is None else center
        return ((h - c) * self.w2).sum(-1)


class BlockLinear(nn.Module):
    """Ablation: linear block terms (a learned, GWAS-assisted-GBLUP-like model)."""

    def __init__(self, M: int, k: int, mask: torch.Tensor):
        super().__init__()
        self.w2 = nn.Parameter(torch.zeros(M, k))
        self.W1 = nn.Parameter(torch.zeros(1))          # placeholders keep the optimiser code shared
        self.b1 = nn.Parameter(torch.zeros(1))
        self.register_buffer("mask", mask)

    def features(self, Zb: torch.Tensor) -> torch.Tensor:
        return Zb

    def forward(self, Zb: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
        c = Zb.mean(0, keepdim=True) if center is None else center
        return ((Zb - c) * (self.w2 * self.mask)).sum(-1)


def gather_blocks(Z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Z [n, p] -> [n, B, k] with zero padding where idx == -1."""
    safe = idx.clamp_min(0)
    out = Z[:, safe.reshape(-1)].reshape(Z.shape[0], *idx.shape)
    return out * (idx >= 0).to(Z.dtype)


def blup_block_scores(X: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Variance of each block's additive BLUP contribution Z_b beta_b (beta = Z^T alpha / p).

    X is the int8 training genotype matrix and mu its column means (Z = X - mu).
    """
    beta = centered_tmatvec(X, alpha, mu) / X.shape[1]                # [p]
    safe = idx.clamp_min(0)
    valid = (idx >= 0).float()
    B = idx.shape[0]
    scores = torch.empty(B, dtype=torch.float32, device=X.device)
    step = 128
    for s0 in range(0, B, step):
        sub = safe[s0:s0 + step]
        bb = beta[sub] * valid[s0:s0 + step]                           # [b, k]
        Zs = centered_cols(X, sub.reshape(-1), mu).reshape(X.shape[0], *sub.shape)
        scores[s0:s0 + step] = torch.einsum("nbk,bk->nb", Zs, bb).var(0)
    return scores


class Context:
    """Per-training-set quantities shared by every HapGAM configuration.

    Holds training genotype centring/scaling, the genomic kernel eigendecomposition, the null
    REML-GBLUP fit and cached block-screening scores (per block scheme).
    """

    def __init__(self, X: torch.Tensor, y: np.ndarray, device, backbone: str = "A",
                 epi_weights=(0.0, 0.25, 0.5, 0.75, 1.0), w_fixed: float | None = None):
        """X: training genotypes as an int8 tensor (kept by reference, never copied to float)."""
        self.dev = device
        self.X = X
        self.mu, sd = col_stats(X)
        self.sd = sd.clamp_min(1e-3)
        Ka = centered_cross(X, None, self.mu)
        self.y_mean, self.y_sd = float(np.mean(y)), float(np.std(y))
        self.ys = torch.tensor((y - self.y_mean) / self.y_sd, device=device, dtype=torch.float64)
        self.w_epi, self.epi_scale = 0.0, 1.0
        if backbone == "A":
            self.K = Ka
            self.null = SpectralREML().fit(self.K, self.ys)
        elif backbone == "AE":
            # additive + additive-by-additive kernel, weight chosen by REML likelihood
            d = torch.diagonal(Ka).mean()
            Kaa = Ka * Ka
            self.epi_scale = float(d / torch.diagonal(Kaa).mean())
            Kaa = Kaa * self.epi_scale
            w = w_fixed if w_fixed is not None else select_epi_weight(Ka, Kaa, self.ys, epi_weights)[0]
            self.w_epi = float(w)
            self.K = (1 - w) * Ka + w * Kaa
            self.null = SpectralREML().fit(self.K, self.ys)
            del Kaa
        else:
            raise ValueError(backbone)
        # block screening uses additive marker effects: the additive part of the backbone BLUP,
        # beta_A = (1 - w) Z^T alpha / p (pure additive REML fit only if w == 1)
        if self.w_epi < 1.0:
            self.alpha_A = (1.0 - self.w_epi) * self.null.alpha
        else:
            self.alpha_A = SpectralREML().fit(Ka, self.ys).alpha
        del Ka
        self._scores = {}

    def kernel_new(self, X_new: torch.Tensor) -> torch.Tensor:
        """Kernel between new genotypes (int8) and the training set, matching the backbone."""
        Ka = centered_cross(X_new, self.X, self.mu)
        if self.w_epi == 0.0:
            return Ka
        return (1 - self.w_epi) * Ka + self.w_epi * (Ka * Ka) * self.epi_scale

    def block_inputs(self, X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Standardised dosages gathered into [n, B, k] blocks, touching only block columns."""
        safe = idx.clamp_min(0).reshape(-1)
        Z = centered_cols(X, safe, self.mu, self.sd)
        return Z.reshape(X.shape[0], *idx.shape) * (idx >= 0).float()

    def scores(self, key, idx: torch.Tensor) -> torch.Tensor:
        if key not in self._scores:
            self._scores[key] = blup_block_scores(self.X, self.mu, self.alpha_A, idx)
        return self._scores[key]


class HapGAM:
    def __init__(self, block_idx: np.ndarray, cfg: HapGAMConfig, device, scheme_key=None):
        self.idx_all = torch.as_tensor(block_idx, device=device)
        self.cfg = cfg
        self.dev = device
        self.scheme_key = scheme_key if scheme_key is not None else id(block_idx)

    def fit(self, X: torch.Tensor, y: np.ndarray, X_val: torch.Tensor | None = None,
            y_val: np.ndarray | None = None, ctx: Context | None = None):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        dev = self.dev
        if ctx is None:
            ctx = Context(X, y, dev, backbone=cfg.backbone, w_fixed=cfg.extra.get("w_fixed"))
        self.ctx = ctx
        self.mu_x, self.sd_x = ctx.mu, ctx.sd
        self.y_mean, self.y_sd = ctx.y_mean, ctx.y_sd
        K, ys, null = ctx.K, ctx.ys, ctx.null
        self.null = null
        s, U = null.s, null.U
        delta = null.delta
        # ---- block screening (training data only) ----------------------------------
        self.block_scores = ctx.scores(self.scheme_key, self.idx_all)
        M = min(cfg.top_m, self.idx_all.shape[0])
        order = torch.argsort(self.block_scores, descending=True)
        self.selected = order[:M]
        self.idx = self.idx_all[self.selected]
        self.history, self.best_step, self.best_val = [], 0, np.nan
        if M == 0:
            self.net = None
            self._finish(K, ys, s, U, delta, None)
            return self
        Zb = ctx.block_inputs(X, self.idx)
        mask = (self.idx >= 0).float()
        if cfg.extra.get("linear"):
            self.net = BlockLinear(M, self.idx.shape[1], mask).to(dev)
        else:
            self.net = BlockMLP(M, self.idx.shape[1], cfg.hidden, mask, seed=cfg.seed).to(dev)
        opt = torch.optim.Adam([
            {"params": [self.net.W1, self.net.b1], "weight_decay": 0.0},
            {"params": [self.net.w2], "weight_decay": 0.0}], lr=cfg.lr)
        Uf = U.float()
        w = (1.0 / (s / delta + 1.0)).float() if cfg.gls else torch.ones_like(s).float()
        wsum = w.sum()
        ysf = ys.float()
        Zb_val = Kv = None
        if X_val is not None:
            Zb_val = ctx.block_inputs(X_val, self.idx)
            Kv = ctx.kernel_new(X_val)
        best = (-np.inf, 0, {k_: v.detach().clone() for k_, v in self.net.state_dict().items()})
        bad = 0
        for step in range(1, cfg.steps + 1):
            opt.zero_grad(set_to_none=True)
            F = self.net(Zb).sum(1)
            Ur = Uf.T @ (ysf - F)
            loss = (w * Ur * Ur).sum() / wsum
            loss = loss + cfg.weight_decay * (self.net.w2 ** 2).sum() + cfg.wd_first * (self.net.W1 ** 2).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            if Zb_val is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
                with torch.no_grad():
                    ctr = self.net.features(Zb).mean(0, keepdim=True)
                    Ftr = self.net(Zb, ctr).sum(1).double()
                    pv = self._predict_from(Zb_val, Kv, Ftr, ys, s, U, delta, ctr)
                    rv = float(np.corrcoef(pv, y_val)[0, 1])
                self.history.append((step, rv))
                if rv > best[0] + 1e-4:
                    best = (rv, step, {k_: v.detach().clone() for k_, v in self.net.state_dict().items()})
                    bad = 0
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break
        if Zb_val is not None:
            self.net.load_state_dict(best[2])
            self.best_step, self.best_val = best[1], best[0]
        else:
            self.best_step = cfg.steps
        self._finish(K, ys, s, U, delta, Zb)
        return self

    def _finish(self, K, ys, s, U, delta, Zb):
        if self.net is None:
            F = torch.zeros_like(ys)
            self.center = None
        else:
            with torch.no_grad():
                self.center = self.net.features(Zb).mean(0, keepdim=True)
                F = self.net(Zb, self.center).sum(1).double()
        resid = ys - F
        post = SpectralREML().fit(K, resid, eig=(s, U)) if self.cfg.refit_delta else None
        self.delta = post.delta if post is not None else delta
        self.mean_r = float(post.beta[0]) if post is not None else float(resid.mean())
        self.alpha = U @ ((U.T @ (resid - self.mean_r)) / (s + self.delta))
        self.F_train = F

    def _predict_from(self, Zb_new, K_new, F_train, ys, s, U, delta, center):
        resid = ys - F_train
        m = resid.mean()
        alpha = U @ ((U.T @ (resid - m)) / (s + delta))
        p = self.net(Zb_new, center).sum(1).double() + m + K_new @ alpha
        return (p * self.y_sd + self.y_mean).cpu().numpy()

    @torch.no_grad()
    def predict(self, X_new: torch.Tensor, return_parts: bool = False):
        Kn = self.ctx.kernel_new(X_new)
        lin = Kn @ self.alpha + self.mean_r
        if self.net is None:
            Fb = torch.zeros(X_new.shape[0], 0, device=X_new.device)
        else:
            Zb = self.ctx.block_inputs(X_new, self.idx)
            Fb = self.net(Zb, self.center)
        p = (Fb.sum(1).double() + lin) * self.y_sd + self.y_mean
        if return_parts:
            return p.cpu().numpy(), (Fb * self.y_sd).cpu().numpy(), (lin * self.y_sd).cpu().numpy()
        return p.cpu().numpy()
