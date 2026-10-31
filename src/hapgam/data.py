"""Dataset access and fold-internal target construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import PROCESSED, SPLITS

PRIMARY_TRAITS = ["maturity_doy", "flowering_doy", "height_cm", "seed_weight_g"]
SECONDARY_TRAITS = ["yield_mg_ha", "protein_pct", "oil_pct", "lodging_score", "maturity_group"]
UNCENTERED = {"maturity_group"}  # already a standardised classification


@lru_cache(maxsize=1)
def grin_accessions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "grin" / "accessions.tsv", sep="\t", keep_default_na=False,
                       na_values={"mg": [""], "latitude": [""], "longitude": [""]})


@lru_cache(maxsize=1)
def grin_records() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "grin" / "records.tsv.gz", sep="\t")


@lru_cache(maxsize=1)
def grin_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS / "grin_splits.tsv", sep="\t", keep_default_na=False)


@lru_cache(maxsize=1)
def grin_dosage() -> np.ndarray:
    return np.load(PROCESSED / "grin" / "dosage.npy", mmap_mode="r")


@lru_cache(maxsize=1)
def grin_snps() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "grin" / "snps.tsv", sep="\t")


@dataclass
class TraitData:
    """Accession-level targets for one trait and one train/test partition."""

    trait: str
    train_rows: np.ndarray      # row indices into the GRIN dosage matrix
    test_rows: np.ndarray
    y_train: np.ndarray         # batch-centred accession means (training batch means)
    y_test: np.ndarray
    test_records: pd.DataFrame  # record-level test data: row, batch, centred value
    train_groups: np.ndarray    # region labels of training accessions (for grouped inner CV)
    batch_means: dict


def build_trait_data(trait: str, train_acc: set, test_acc: set) -> TraitData:
    """Centre records by training-fold batch means and average per accession.

    Batch means are computed from training records only. Test records in a batch
    that has no training records are centred by the test batch's own mean (this only
    affects RMSE; Pearson r within batch is invariant to the offset).
    """
    acc = grin_accessions().set_index("accession")
    rec = grin_records()
    rec = rec[rec["trait"] == trait]
    tr = rec[rec["accession"].isin(train_acc)]
    te = rec[rec["accession"].isin(test_acc)]
    if trait in UNCENTERED:
        means = {b: 0.0 for b in rec["batch"].unique()}
    else:
        means = tr.groupby("batch")["value"].mean().to_dict()
        for b, m in te.groupby("batch")["value"].mean().items():
            means.setdefault(b, m)
    tr = tr.assign(c=tr["value"] - tr["batch"].map(means))
    te = te.assign(c=te["value"] - te["batch"].map(means))
    ytr = tr.groupby("accession")["c"].mean()
    yte = te.groupby("accession")["c"].mean()
    train_rows = acc.loc[ytr.index, "row"].to_numpy()
    test_rows = acc.loc[yte.index, "row"].to_numpy()
    te = te.assign(row=acc.loc[te["accession"], "row"].to_numpy())
    return TraitData(trait, train_rows, test_rows, ytr.to_numpy(float), yte.to_numpy(float),
                     te[["accession", "row", "batch", "c"]].reset_index(drop=True),
                     acc.loc[ytr.index, "region"].to_numpy(), means)


def scenario_partitions(scenario: str):
    """Yield (fold_name, train_accessions, test_accessions) for a GRIN scenario."""
    s = grin_splits()
    if scenario == "dev_random":
        d = s[s["set"] == "dev"]
        for k in range(5):
            yield f"f{k}", set(d.loc[d["dev_fold"] != k, "accession"]), set(d.loc[d["dev_fold"] == k, "accession"])
    elif scenario == "dev_region":
        d = s[s["set"] == "dev"]
        for reg in sorted(r for r in d["dev_region_fold"].unique() if r != "train_only"):
            yield reg, set(d.loc[d["dev_region_fold"] != reg, "accession"]), set(d.loc[d["dev_region_fold"] == reg, "accession"])
    elif scenario == "C1":
        yield "lockbox_random", set(s.loc[s["set"] == "dev", "accession"]), set(s.loc[s["set"] == "lockbox_random", "accession"])
    elif scenario == "C2":
        yield "lockbox_japan", set(s.loc[s["set"] != "lockbox_japan", "accession"]), set(s.loc[s["set"] == "lockbox_japan", "accession"])
    elif scenario == "C3":
        for reg in sorted(r for r in s["region_fold"].unique() if r != "train_only"):
            yield reg, set(s.loc[s["region_fold"] != reg, "accession"]), set(s.loc[s["region_fold"] == reg, "accession"])
    elif scenario == "C4":
        for g in sorted(r for r in s["mg_fold"].unique() if r != "train_only"):
            yield g, set(s.loc[s["mg_fold"] != g, "accession"]), set(s.loc[s["mg_fold"] == g, "accession"])
    elif scenario == "grin_all":
        yield "all", set(s["accession"]), set()
    else:
        raise ValueError(scenario)


INNER_KIND = {"dev_random": "random", "C1": "random", "dev_region": "group", "C2": "group",
              "C3": "group", "C4": "group", "soynam_random": "random", "soynam_lofo": "group",
              "grin_all": "group", "grin2soynam": "group", "grin2nust": "group", "pc_random": "random"}


def inner_folds(groups: np.ndarray, kind: str, k: int = 3, seed: int = 0) -> list[np.ndarray]:
    """Validation index sets for inner CV, shared by every model.

    random: k random folds.  group: whole groups (regions, MG classes or families) are
    assigned to k folds, largest first, each to the currently smallest fold; groups named
    Other/Unknown/train_only always stay in training.
    """
    n = len(groups)
    rng = np.random.default_rng(seed)
    if kind == "random":
        lab = rng.permutation(np.arange(n) % k)
        return [np.flatnonzero(lab == i) for i in range(k)]
    if kind == "group":
        groups = np.asarray(groups).astype(str)
        uniq, counts = np.unique(groups, return_counts=True)
        keep = [(c, g) for g, c in zip(uniq, counts) if g not in ("Other", "Unknown", "train_only")]
        keep.sort(reverse=True)
        k = min(k, len(keep))
        sizes = np.zeros(k)
        members = [[] for _ in range(k)]
        for c, g in keep:
            j = int(np.argmin(sizes))
            members[j].append(g)
            sizes[j] += c
        return [np.flatnonzero(np.isin(groups, m)) for m in members]
    raise ValueError(kind)
