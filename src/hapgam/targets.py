"""Fold-internal phenotype adjustment for GRIN evaluation batches.

GRIN accessions were evaluated in batches (``method_name``) that are organised by
maturity group, so batch means mix environmental and genetic differences.  Within each
training fold we fit the record-level mixed model

    y_rj = mu + b_j + g_i(r) + e_r,   g ~ N(0, s2g K)

with batch effects b_j as fixed effects and the genomic relationship K linking
accessions across batches; the estimated environmental batch effects are then removed
from training and test records.  Only training records enter the fit.  Test records in
batches without training records receive NaN (excluded from pooled metrics).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data
from .paths import PROCESSED
from .linalg import centered_cross, col_stats
from .reml import SpectralREML

TARGET_DIR = PROCESSED / "targets"


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def debatch_effects(trait: str, train_acc: set, device) -> tuple[dict, dict]:
    acc = data.grin_accessions().set_index("accession")
    rec = data.grin_records()
    rec = rec[(rec["trait"] == trait) & rec["accession"].isin(train_acc)].reset_index(drop=True)
    batches = sorted(rec["batch"].unique())
    if trait in data.UNCENTERED or len(batches) == 1:
        return {b: 0.0 for b in batches}, {"method": "none"}
    rows = acc.loc[rec["accession"], "row"].to_numpy()
    G = data.grin_dosage()
    uniq, inv = np.unique(rows, return_inverse=True)
    Xu = torch.as_tensor(np.asarray(G[uniq], dtype=np.int8), device=device)
    mu, _ = col_stats(Xu)
    Ku = centered_cross(Xu, None, mu)
    del Xu
    inv_t = torch.as_tensor(inv, device=device)
    K = Ku[inv_t][:, inv_t]
    del Ku
    code = pd.Categorical(rec["batch"], categories=batches).codes.astype(np.int64)
    C = torch.zeros(len(rec), len(batches) - 1, dtype=torch.float64, device=device)
    nz = code > 0
    C[torch.as_tensor(np.flatnonzero(nz), device=device), torch.as_tensor(code[nz] - 1, device=device)] = 1.0
    y = torch.tensor(rec["value"].to_numpy(float), device=device)
    m = SpectralREML().fit(K, y, C)
    beta = m.beta.cpu().numpy()
    eff = {batches[0]: 0.0}
    for j, b in enumerate(batches[1:]):
        eff[b] = float(beta[j + 1])
    info = {"method": "gblup_fixed_batch", "h2": m.h2, "delta": m.delta, "n_records": int(len(rec)),
            "n_batches": len(batches)}
    del K, C, m
    torch.cuda.empty_cache()
    return eff, info


def build_targets(trait: str, train_acc: set, test_acc: set, device) -> dict:
    acc = data.grin_accessions().set_index("accession")
    rec = data.grin_records()
    rec = rec[rec["trait"] == trait]
    eff, info = debatch_effects(trait, train_acc, device)
    rec = rec.assign(adj=rec["value"] - rec["batch"].map(eff))   # NaN if batch unseen in training
    tr = rec[rec["accession"].isin(train_acc)]
    te = rec[rec["accession"].isin(test_acc)]
    ytr = tr.groupby("accession")["adj"].mean()
    yte = te.groupby("accession")["adj"].mean()  # NaN if all records are in unseen batches
    out = {
        "train_rows": acc.loc[ytr.index, "row"].to_numpy(np.int64),
        "y_train": ytr.to_numpy(float),
        "train_groups": acc.loc[ytr.index, "region"].to_numpy(str),
        "test_rows": acc.loc[yte.index, "row"].to_numpy(np.int64),
        "y_test": yte.to_numpy(float),
        "rec_row": acc.loc[te["accession"], "row"].to_numpy(np.int64),
        "rec_batch": te["batch"].to_numpy(str),
        "rec_value": te["value"].to_numpy(float),
        "rec_adj": te["adj"].to_numpy(float),
        "effects_batch": np.array(list(eff.keys()), dtype=str),
        "effects_value": np.array(list(eff.values()), dtype=float),
    }
    out["info"] = np.array([str(info)])
    return out


def target_path(scenario: str, trait: str, fold: str) -> Path:
    return TARGET_DIR / scenario / trait / f"{safe(fold)}.npz"


def load_targets(scenario: str, trait: str, fold: str) -> data.TraitData:
    z = np.load(target_path(scenario, trait, fold), allow_pickle=False)
    recs = pd.DataFrame({"row": z["rec_row"], "batch": z["rec_batch"], "c": z["rec_value"]})
    return data.TraitData(trait, z["train_rows"], z["test_rows"], z["y_train"], z["y_test"], recs,
                          z["train_groups"], dict(zip(z["effects_batch"], z["effects_value"])))
