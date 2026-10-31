"""Synthetic checks: nesting of GBLUP and recovery of a planted within-block epistatic effect."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hapgam.blocks import block_index, fixed_blocks  # noqa: E402
from hapgam.model import HapGAM, HapGAMConfig  # noqa: E402
from hapgam.reml import SpectralREML  # noqa: E402


def simulate(n=1500, p=480, seed=0):
    rng = np.random.default_rng(seed)
    # inbred-like haplotypes: blocks of 8 SNPs drawn from 16 founder haplotypes each, so a
    # haplotype function is in general NOT a linear function of the block's SNPs
    B = p // 8
    X = np.zeros((n, p))
    for b in range(B):
        founders = rng.integers(0, 2, size=(16, 8)) * 2
        X[:, b * 8:(b + 1) * 8] = founders[rng.integers(0, 16, size=n)]
    beta = rng.normal(0, 0.05, p)
    g = (X - X.mean(0)) @ beta
    # planted epistatic (XOR-like) effect in block 3: SNPs 24 and 25
    a, c = X[:, 24] > 0, X[:, 25] > 0
    epi = np.where(a ^ c, 1.0, -1.0) * g.std() * 1.0
    y = g + epi + rng.normal(0, g.std(), n)
    return X, y, np.repeat(np.arange(B), 8)


def test_nesting_and_recovery():
    X, y, blocks = simulate()
    _, _, idx = block_index(fixed_blocks(np.zeros(len(blocks), dtype=int), 8))
    n_tr = 1200
    Xt = torch.tensor(X, dtype=torch.float32)
    dev = torch.device("cpu")
    # (a) top_m = 0 -> exactly GBLUP
    big = HapGAM(idx, HapGAMConfig(top_m=0), dev).fit(Xt[:n_tr], y[:n_tr])
    Zc = Xt[:n_tr] - Xt[:n_tr].mean(0, keepdim=True)
    K = (Zc @ Zc.T / Zc.shape[1]).double()
    ys = (y[:n_tr] - y[:n_tr].mean()) / y[:n_tr].std()
    null = SpectralREML().fit(K, torch.tensor(ys))
    Kte = ((Xt[n_tr:] - Xt[:n_tr].mean(0, keepdim=True)) @ Zc.T / Zc.shape[1]).double()
    p_gblup = null.predict(Kte).numpy() * y[:n_tr].std() + y[:n_tr].mean()
    p_big = big.predict(Xt[n_tr:])
    assert np.allclose(p_big, p_gblup, atol=1e-6 * np.std(p_gblup) + 1e-8), np.abs(p_big - p_gblup).max()
    # (b) the planted block 3 is screened in, dominates the block terms, and accuracy improves
    val = np.zeros(n_tr, bool)
    val[np.random.default_rng(0).choice(n_tr, 150, replace=False)] = True
    m = HapGAM(idx, HapGAMConfig(top_m=8, steps=3000, hidden=8), dev).fit(
        Xt[:n_tr][~val], y[:n_tr][~val], Xt[:n_tr][val], y[:n_tr][val])
    p = m.predict(Xt[n_tr:])
    r_h = np.corrcoef(p, y[n_tr:])[0, 1]
    r_g = np.corrcoef(p_gblup, y[n_tr:])[0, 1]
    _, Fb, _ = m.predict(Xt[n_tr:], return_parts=True)
    top = int(m.selected[int(np.argmax(Fb.var(0)))])
    print(f"r_gblup={r_g:.3f} r_hapgam={r_h:.3f} selected={m.selected.tolist()} top_block={top}")
    assert top == 3
    assert r_h > r_g + 0.05


if __name__ == "__main__":
    test_nesting_and_recovery()
    print("OK")
