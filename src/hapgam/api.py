"""Uniform fit/predict entry points for every model.

Each function takes training genotypes ``Xtr`` (numpy int8, n x p), targets ``ytr``, group labels
for the shared inner CV, the inner-CV kind ("random" or "group"), and test genotypes ``Xte``;
it returns (predictions, info).  Hyper-parameters are always selected on the shared inner
folds of the training data only.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch

from . import data
from .blocks import block_index, fixed_blocks
from .kernels import additive, hadamard_epistatic, haplotype_onehot
from .linalg import centered_cross, centered_sqnorm, col_stats, kernel_pcs
from .metrics import pearson
from .model import Context, HapGAM, HapGAMConfig
from .reml import SpectralREML, select_epi_weight

HAPGAM_GRID = {"scheme": ["fixed8", "fixed16", "fixed32"], "top_m": [8, 16, 32, 64],
               "weight_decay": [1e-3, 1e-2]}
HAPGAM_BASE = dict(hidden=8, lr=5e-4, steps=4000, eval_every=10, patience=20)


def _t(a, dev):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)


def _i8(a, dev):
    """Genotypes as an int8 tensor (1 byte per call)."""
    return torch.as_tensor(np.asarray(a, dtype=np.int8), device=dev)


def schemes_for(chrom: np.ndarray, names=("fixed8", "fixed16", "fixed32")):
    return {n: block_index(fixed_blocks(chrom, int(n[5:])))[2] for n in names}


# ---------------------------------------------------------------------------------------
def gblup(Xtr, ytr, Xte, dev, **_):
    Xt, Xs = _i8(Xtr, dev), _i8(Xte, dev)
    mu, _ = col_stats(Xt)
    K, Kte = centered_cross(Xt, None, mu), centered_cross(Xs, Xt, mu)
    m = SpectralREML().fit(K, torch.tensor(ytr, device=dev))
    return m.predict(Kte).cpu().numpy(), {"delta": m.delta, "h2": m.h2}


def kernel_model(model, Xtr, ytr, Xte, dev, block_idx=None, **_):
    """rkhs | egblup | hapgblup with kernel choice by REML likelihood (training data only)."""
    Xt, Xs = _i8(Xtr, dev), _i8(Xte, dev)
    mu, _ = col_stats(Xt)
    Ka = centered_cross(Xt, None, mu)          # (Z Z^T)/p, Z = X - mu
    Kat = centered_cross(Xs, Xt, mu)
    if model == "rkhs":
        # one bandwidth at a time to bound memory
        y = torch.tensor(ytr, device=dev)
        best, table = None, {}
        p = Xt.shape[1]
        dtr = torch.diagonal(Ka).clone()
        dte = centered_sqnorm(Xs, mu)
        Dtr = (p * (dtr[:, None] + dtr[None, :] - 2 * Ka)).float()       # squared distances
        Dte = (p * (dte[:, None] + dtr[None, :] - 2 * Kat)).float().clamp_min(0)
        del Ka, Kat
        iu = torch.triu_indices(len(dtr), len(dtr), 1, device=Dtr.device)
        med = float(torch.median(Dtr[iu[0][:200000], iu[1][:200000]]))
        del iu
        for h in (0.25, 0.5, 1.0, 2.0, 4.0):
            Ktr, Kte = torch.exp(-Dtr.double() / (h * med)), torch.exp(-Dte.double() / (h * med))
            m = SpectralREML().fit(Ktr, y)
            table[f"h={h}"] = m.loglik
            if best is None or m.loglik > best[0]:
                best = (m.loglik, f"h={h}", m.predict(Kte).cpu().numpy())
            del Ktr, Kte, m
        return best[2], {"chosen": best[1], "loglik": table}
    elif model == "egblup":
        Kaa, Kaat = hadamard_epistatic(Ka, Kat)
        ys = torch.tensor((ytr - ytr.mean()) / ytr.std(), device=dev)
        w, table = select_epi_weight(Ka, Kaa, ys)          # same rule as HapGAM-E
        K = Ka.mul_(1 - w).add_(Kaa, alpha=w)              # in place: (1-w)K_A + wK_AA
        del Kaa
        Kte = Kat.mul_(1 - w).add_(Kaat, alpha=w)
        del Kaat
        m = SpectralREML().fit(K, torch.tensor(ytr, device=dev))
        return m.predict(Kte).cpu().numpy(), {"chosen": f"w={w}", "loglik_sub": table}
    elif model == "hapgblup":
        Xn = np.concatenate([Xtr, Xte])
        mask = np.zeros(len(Xn), bool)
        mask[: len(Xtr)] = True
        H = torch.tensor(haplotype_onehot(Xn, block_idx, train_mask=mask), device=dev)
        H = H - H[: len(Xtr)].mean(0, keepdim=True)
        Kh, Kht = additive(H[: len(Xtr)], H[len(Xtr):])
        del H
        da, dh = float(Ka.diagonal().mean()), float(Kh.diagonal().mean())
        Ka.div_(da); Kat.div_(da); Kh.div_(dh); Kht.div_(dh)
        y = torch.tensor(ytr, device=dev)
        best, table = None, {}
        for w in (0.0, 0.5, 0.75, 1.0):                    # one combined kernel at a time
            m = SpectralREML().fit((1 - w) * Ka + w * Kh, y)
            table[f"w={w}"] = m.loglik
            if best is None or m.loglik > best[0]:
                best = (m.loglik, f"w={w}", m.predict((1 - w) * Kat + w * Kht).cpu().numpy())
            del m
        return best[2], {"chosen": best[1], "loglik": table}
    raise ValueError(model)


# ---------------------------------------------------------------------------------------
def _es_split(n, rng, frac=0.1):
    es = np.zeros(n, bool)
    es[rng.choice(n, max(int(frac * n), 50), replace=False)] = True
    return es


def hapgam(Xtr, ytr, groups, inner_kind, Xte, dev, schemes, grid=None, base=None, seed=0,
           linear=False, return_model=False, backbone="A", **_):
    """HapGAM with inner-CV selection over (block scheme, top-M, weight decay); M=0 = GBLUP."""
    grid = grid or HAPGAM_GRID
    base = {**HAPGAM_BASE, **(base or {})}
    if linear:
        base["extra"] = {"linear": True}
    base["backbone"] = backbone
    rng = np.random.default_rng(seed)
    X = _i8(Xtr, dev)
    w_fixed = None
    if backbone == "AE":
        # epistatic-kernel weight chosen once by REML on the outer training set (as in EGBLUP)
        w_fixed = Context(X, ytr, dev, backbone="AE").w_epi
        torch.cuda.empty_cache()
    base["extra"] = {**base.get("extra", {}), "w_fixed": w_fixed}
    configs = [{"scheme": grid["scheme"][0], "top_m": 0, "weight_decay": 0.0}]
    keys = list(grid)
    configs += [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    rows = []
    for f, va in enumerate(data.inner_folds(groups, inner_kind)):
        tr = np.setdiff1d(np.arange(len(ytr)), va)
        es = _es_split(len(tr), rng)
        fit_i, es_i = tr[~es], tr[es]
        ti = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
        ctx = Context(X[ti(fit_i)], ytr[fit_i], dev, backbone=backbone, w_fixed=w_fixed)
        for c in configs:
            cfg = HapGAMConfig(**base, top_m=c["top_m"], weight_decay=c["weight_decay"], seed=seed)
            m = HapGAM(schemes[c["scheme"]], cfg, dev, scheme_key=c["scheme"]).fit(
                X[ti(fit_i)], ytr[fit_i], X[ti(es_i)], ytr[es_i], ctx=ctx)
            rows.append({**c, "inner_fold": f, "r": pearson(ytr[va], m.predict(X[ti(va)])),
                         "best_step": m.best_step})
            del m
        del ctx
        torch.cuda.empty_cache()
    tune = pd.DataFrame(rows)
    agg = tune.groupby(["scheme", "top_m", "weight_decay"], as_index=False).agg(r=("r", "mean"))
    agg = agg.sort_values(["r", "top_m"], ascending=[False, True])
    best = agg.iloc[0].to_dict()
    cfg = HapGAMConfig(**base, top_m=int(best["top_m"]), weight_decay=float(best["weight_decay"]), seed=seed)
    idx = schemes[best["scheme"]]
    ti = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    steps = cfg.steps
    if cfg.top_m > 0:
        es = _es_split(len(ytr), rng)
        m1 = HapGAM(idx, cfg, dev, scheme_key=best["scheme"]).fit(
            X[ti(np.flatnonzero(~es))], ytr[~es], X[ti(np.flatnonzero(es))], ytr[es])
        steps = max(m1.best_step, 1)
        del m1
    m = HapGAM(idx, HapGAMConfig(**{**cfg.__dict__, "steps": steps}), dev, scheme_key=best["scheme"]).fit(X, ytr)
    Xs = _i8(Xte, dev)
    pred, Fb, lin = m.predict(Xs, return_parts=True)
    p_null = m.null.predict(m.ctx.kernel_new(Xs)).cpu().numpy() * m.y_sd + m.y_mean
    info = {"chosen": best, "steps": int(steps), "tuning": tune, "w_epi": float(m.ctx.w_epi),
            "inner_r_gblup": float(agg.loc[agg["top_m"] == 0, "r"].iloc[0]),
            "selected_blocks": m.selected.cpu().numpy(),
            "block_scores": m.block_scores.cpu().numpy(), "scheme_idx": idx,
            "test_block_contrib": Fb, "test_linear": lin, "pred_gblup_same_fit": p_null}
    if return_model:
        info["model"] = m
    else:
        del m
    del X, Xs
    torch.cuda.empty_cache()
    return pred, info


# ---------------------------------------------------------------------------------------
def trees(model, Xtr, ytr, groups, inner_kind, Xte, threads=16, trials=30, device="cpu", **_):
    from .trees import fit_lightgbm, fit_xgboost, tune_lightgbm, tune_xgboost

    folds = data.inner_folds(groups, inner_kind)
    if model == "lightgbm":
        params, rounds, study = tune_lightgbm(Xtr, ytr, folds, n_trials=trials, n_threads=threads)
        pred = fit_lightgbm(Xtr, ytr, params, rounds).predict(Xte)
    else:
        import xgboost as xgb

        params, rounds, study = tune_xgboost(Xtr, ytr, folds, n_trials=trials, device=device, nthread=threads)
        pred = fit_xgboost(Xtr, ytr, params, rounds).predict(xgb.DMatrix(Xte))
    return pred, {"params": params, "rounds": rounds, "inner_r": study.best_value, "trials": len(study.trials)}


# ---------------------------------------------------------------------------------------
NET_GRID = {
    "dnngp": {"dropout": [0.1, 0.3, 0.5], "lr": [1e-3, 1e-2], "n_pc": [256]},
    "mlp": {"dropout": [0.1, 0.3, 0.5], "lr": [1e-4, 1e-3], "wd": [1e-4]},
}


def nets(model, Xtr, ytr, groups, inner_kind, Xte, dev, seeds=5, **_):
    """DNNGP replica / MLP: grid on the shared inner CV, then a seed ensemble on all data."""
    from .nets import DNNGP, MLP, predict_net, train_net

    if model == "dnngp":
        X, Xs = _i8(Xtr, dev), _i8(Xte, dev)          # PCs from the kernel; no float copy needed
    else:
        X, Xs = _t(Xtr, dev), _t(Xte, dev)
    y = torch.tensor(ytr, dtype=torch.float32, device=dev)
    keys = list(NET_GRID[model])

    pc_cache = {}   # DNNGP PCs depend only on the inner fold and k, not on dropout/lr: compute once per fold

    def prep(A, B, hp, key=None):
        if model == "dnngp":
            k = min(hp["n_pc"], A.shape[0] - 1, A.shape[1])
            if key is None or (key, k) not in pc_cache:
                mu, _ = col_stats(A)
                pcs = kernel_pcs(A, B, mu, k)
                if key is None:
                    return (*pcs, lambda: DNNGP(k, hp["dropout"]))
                pc_cache[(key, k)] = pcs
            return (*pc_cache[(key, k)], lambda: DNNGP(k, hp["dropout"]))
        mu = A.mean(0, keepdim=True)
        A, B = A - mu, B - mu
        return A, B, (lambda: MLP(A.shape[1], dropout=hp["dropout"]))

    folds = data.inner_folds(groups, inner_kind)
    results = []
    for combo in itertools.product(*NET_GRID[model].values()):
        hp = dict(zip(keys, combo))
        rs, eps = [], []
        for i, va in enumerate(folds):
            tri = torch.as_tensor(np.setdiff1d(np.arange(len(ytr)), va), device=dev)
            vai = torch.as_tensor(va, device=dev)
            A, B, make = prep(X[tri], X[vai], hp, key=i)
            net, ep, r = train_net(make, A, y[tri], B, y[vai], lr=hp["lr"], wd=hp.get("wd", 0.0), device=dev)
            rs.append(r)
            eps.append(ep)
            del net, A, B
        results.append((float(np.mean(rs)), hp, int(np.mean(eps))))
    score, hp, ep = max(results, key=lambda t: t[0])
    A, B, make = prep(X, Xs, hp)
    preds = []
    for seed in range(seeds):
        net, _, _ = train_net(make, A, y, None, None, lr=hp["lr"], wd=hp.get("wd", 0.0), seed=seed,
                              device=dev, fixed_epochs=max(ep, 1))
        preds.append(predict_net(net, B))
        del net
    del X, Xs, A, B
    torch.cuda.empty_cache()
    return np.mean(preds, 0), {"hp": hp, "epochs": ep, "inner_r": score, "seed_preds": np.array(preds)}
