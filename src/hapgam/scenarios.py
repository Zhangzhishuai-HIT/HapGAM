"""Prediction tasks for every scenario (GRIN folds and external SoyNAM / NUST transfer).

A Task holds numpy genotypes (int8, GRIN ALT-dosage coding), targets, the inner-CV grouping
of the training set, test identifiers and the grouping used for the within-group metric
(GRIN evaluation batch, SoyNAM family or NUST MG test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .paths import INTERIM, PROCESSED, SPLITS
from .targets import load_targets, target_path

GRIN_SCENARIOS = ["dev_random", "dev_region", "C1", "C2", "C3", "C4"]
SOYNAM_SCENARIOS = ["soynam_random", "soynam_lofo"]
TRANSFER_SCENARIOS = ["grin2soynam", "grin2nust"]

SOYNAM_TRAIT = {"maturity_doy": "R8", "flowering_doy": "flower", "height_cm": "height",
                "seed_weight_g": "size", "yield_mg_ha": "yield", "protein_pct": "protein",
                "oil_pct": "oil", "lodging_score": "lodging"}
NUST_TRAIT = {"maturity_doy": "Maturity", "height_cm": "Height", "seed_weight_g": "SeedSize",
              "yield_mg_ha": "YieldBuA", "protein_pct": "Protein", "oil_pct": "Oil",
              "lodging_score": "Lodging"}


@dataclass
class Task:
    scenario: str
    trait: str
    fold: str
    Xtr: np.ndarray
    ytr: np.ndarray
    groups: np.ndarray
    inner_kind: str
    Xte: np.ndarray
    yte: np.ndarray
    test_ids: np.ndarray
    test_groups: np.ndarray            # group per test individual for within-group r
    chrom: np.ndarray                  # chromosome of each SNP column (for block schemes)
    records: pd.DataFrame | None = None  # GRIN record-level test data (within-batch r)
    meta: dict = field(default_factory=dict)
    cols: np.ndarray | None = None     # GRIN SNP index of each genotype column


# ----------------------------------------------------------------------------- GRIN
def grin_task(scenario, trait, fold):
    td = load_targets(scenario, trait, fold)
    G = data.grin_dosage()
    acc = data.grin_accessions()
    # the batch of each test accession for grouping = its first record's batch
    rec = td.test_records
    first = rec.groupby("row")["batch"].first()
    return Task(scenario, trait, fold, np.asarray(G[td.train_rows]), td.y_train, td.train_groups,
                data.INNER_KIND[scenario], np.asarray(G[td.test_rows]), td.y_test,
                acc.loc[td.test_rows, "accession"].to_numpy(), first.reindex(td.test_rows).to_numpy(),
                data.grin_snps()["chrom"].to_numpy(), records=rec,
                cols=np.arange(G.shape[1]))


# ----------------------------------------------------------------------------- SoyNAM
@lru_cache(maxsize=1)
def soynam_geno():
    X = np.load(PROCESSED / "soynam" / "dosage_grin_coded.npy", mmap_mode="r")
    idx = np.load(PROCESSED / "soynam" / "grin_snp_index.npy")
    lines = pd.read_csv(PROCESSED / "soynam" / "lines.tsv", sep="\t")["line"].to_numpy()
    return X, idx, lines


@lru_cache(maxsize=1)
def soynam_pheno() -> pd.DataFrame:
    qa = pd.read_csv(INTERIM / "soynam" / "data_line_qa.tsv", sep="\t")
    raw = pd.read_csv(INTERIM / "soynam" / "data_line_raw.tsv", sep="\t")
    keep = ["environ", "strain", "family", "height", "R8", "lodging", "yield", "protein", "oil", "size"]
    long = qa[keep].melt(id_vars=["environ", "strain", "family"], var_name="trait", value_name="value")
    fl = raw[["environ", "strain", "family", "flower"]].dropna()
    cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    m, d = np.divmod(fl["flower"].astype(int), 100)
    ok = (m >= 5) & (m <= 10) & (d >= 1) & (d <= 31)
    fl = fl[ok].assign(trait="flower", value=(cum[m[ok] - 1] + d[ok]).astype(float))[
        ["environ", "strain", "family", "trait", "value"]]
    long = pd.concat([long, fl], ignore_index=True).dropna(subset=["value"])
    long["line"] = long["strain"].str.replace("-", "N", n=1, regex=False)
    return long


def soynam_splits() -> pd.DataFrame:
    path = SPLITS / "soynam_splits.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    _, _, lines = soynam_geno()
    ph = soynam_pheno()
    fam = ph.drop_duplicates("line").set_index("line")["family"]
    lines = [l for l in lines if l in fam.index]
    rng = np.random.default_rng(20261020)
    s = pd.DataFrame({"line": lines, "family": fam.loc[lines].astype(int).to_numpy()})
    s["random_fold"] = rng.permutation(np.arange(len(s)) % 5)
    fams = s.groupby("family").size().sort_values(ascending=False)
    load = np.zeros(5)
    assign = {}
    for f, c in fams.items():
        j = int(np.argmin(load))
        assign[f] = j
        load[j] += c
    s["family_fold"] = s["family"].map(assign)
    s.to_csv(path, sep="\t", index=False)
    return s


def twoway_env_effects(df: pd.DataFrame, n_iter: int = 30) -> dict:
    """Environment effects from y = env + line (both fixed) by alternating centring."""
    y = df["value"].to_numpy(float)
    env = pd.factorize(df["environ"])
    line = pd.factorize(df["line"])
    e = np.zeros(len(env[1]))
    g = np.zeros(len(line[1]))
    for _ in range(n_iter):
        g = np.bincount(line[0], y - e[env[0]], len(g)) / np.bincount(line[0], minlength=len(g))
        e = np.bincount(env[0], y - g[line[0]], len(e)) / np.bincount(env[0], minlength=len(e))
    e -= e.mean()
    return dict(zip(env[1], e))


def soynam_targets(trait_col: str, train_lines: set | None):
    """Line means of env-adjusted records; env effects from training lines (or all lines)."""
    ph = soynam_pheno()
    ph = ph[ph["trait"] == trait_col]
    fitset = ph if train_lines is None else ph[ph["line"].isin(train_lines)]
    eff = twoway_env_effects(fitset)
    ph = ph.assign(adj=ph["value"] - ph["environ"].map(eff))
    return ph.dropna(subset=["adj"]).groupby("line").agg(y=("adj", "mean"), family=("family", "first"))


def _soynam_matrix(lines, snp_cols):
    X, idx, all_lines = soynam_geno()
    pos = {l: i for i, l in enumerate(all_lines)}
    M = np.asarray(X[[pos[l] for l in lines]][:, snp_cols]).astype(np.int8)
    return M


def _impute(Xtr, Xte):
    """Mean-impute missing (-1) with training allele means, rounded to the nearest code."""
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    Xtr[Xtr < 0] = np.nan
    Xte[Xte < 0] = np.nan
    mu = np.nanmean(Xtr, 0)
    mu = np.where(np.isnan(mu), 0, mu)
    Xtr = np.where(np.isnan(Xtr), np.round(mu), Xtr)
    Xte = np.where(np.isnan(Xte), np.round(mu), Xte)
    return Xtr.astype(np.int8), Xte.astype(np.int8)


def soynam_task(scenario, trait, fold):
    col = SOYNAM_TRAIT[trait]
    s = soynam_splits()
    key = "random_fold" if scenario == "soynam_random" else "family_fold"
    k = int(fold[1:])
    train_lines = set(s.loc[s[key] != k, "line"])
    test_lines = set(s.loc[s[key] == k, "line"])
    tg = soynam_targets(col, train_lines)
    _, idx, _ = soynam_geno()
    X = np.load(PROCESSED / "soynam" / "dosage_grin_coded.npy", mmap_mode="r")
    # SNPs polymorphic in SoyNAM training lines (MAF >= 0.01), phenotype-free
    tr = [l for l in tg.index if l in train_lines]
    te = [l for l in tg.index if l in test_lines]
    Xtr_all = _soynam_matrix(tr, idx)
    called = Xtr_all >= 0
    p = np.where(called, Xtr_all, 0).sum(0) / np.maximum(2 * called.sum(0), 1)
    poly = (np.minimum(p, 1 - p) >= 0.01) & (called.mean(0) >= 0.9)
    cols = idx[poly]
    Xtr, Xte = _impute(Xtr_all[:, poly], _soynam_matrix(te, cols))
    chrom = data.grin_snps()["chrom"].to_numpy()[cols]
    groups = tg.loc[tr, "family"].astype(str).to_numpy()
    kind = "random" if scenario == "soynam_random" else "group"
    return Task(scenario, trait, fold, Xtr, tg.loc[tr, "y"].to_numpy(float), groups, kind, Xte,
                tg.loc[te, "y"].to_numpy(float), np.array(te), tg.loc[te, "family"].astype(str).to_numpy(),
                chrom, meta={"n_snps": int(len(cols))}, cols=cols)


# ----------------------------------------------------------------------------- transfer
def grin_all_targets(trait):
    """GRIN accessions with targets de-batched on all GRIN records (scenario 'grin_all')."""
    return load_targets("grin_all", trait, "all")


def transfer_task(scenario, trait, fold="all"):
    td = grin_all_targets(trait)
    G = data.grin_dosage()
    if scenario == "grin2soynam":
        _, cols, _ = soynam_geno()
        tg = soynam_targets(SOYNAM_TRAIT[trait], None)
        tg = tg[tg.index.isin(set(soynam_splits()["line"]))]
        lines = list(tg.index)
        Xte_raw = _soynam_matrix(lines, cols)
        yte = tg["y"].to_numpy(float)
        tgroups = tg["family"].astype(str).to_numpy()
        ids = np.array(lines)
    elif scenario == "grin2nust":
        cols = np.load(PROCESSED / "nust" / "grin_snp_index.npy")
        b = pd.read_csv(PROCESSED / "nust" / "blues.tsv", sep="\t")
        b = b.dropna(subset=[NUST_TRAIT[trait]])
        Xn = np.load(PROCESSED / "nust" / "dosage_grin_coded.npy", mmap_mode="r")
        Xte_raw = np.asarray(Xn[b["row"].to_numpy()][:, cols]).astype(np.int8)
        yte = b[NUST_TRAIT[trait]].to_numpy(float)
        tgroups = b["MG_factor"].to_numpy(str)
        ids = b["GermplasmId"].to_numpy(str)
    else:
        raise ValueError(scenario)
    Xtr = np.asarray(G[td.train_rows][:, cols]).astype(np.int8)
    Xtr, Xte = _impute(Xtr, Xte_raw)
    chrom = data.grin_snps()["chrom"].to_numpy()[cols]
    return Task(scenario, trait, fold, Xtr, td.y_train, td.train_groups, "group", Xte, yte, ids, tgroups,
                chrom, meta={"n_snps": int(len(cols))}, cols=cols)


# ----------------------------------------------------------------------------- positive controls
# binary descriptors with known causal genes (W1, T, Dt1, I/R); dev accessions, dev random folds
PC_TRAITS = {
    "pc_flower_purple": ("flower_color", {"P", "Dp", "Lp"}, {"W", "Nw"}),
    "pc_pubescence_tawny": ("pubescence_color", {"T", "Lt"}, {"G", "Ng"}),
    "pc_indeterminate": ("stem_termination", {"N"}, {"D"}),
    "pc_seedcoat_pigmented": ("seed_coat_color", {"Bl", "Br", "Rbr", "Gnbr"}, {"Y"}),
}


def pc_task(scenario, trait, fold):
    col, pos, neg = PC_TRAITS[trait]
    cat = pd.read_csv(PROCESSED / "grin" / "categorical.tsv", sep="\t", keep_default_na=False)
    acc = data.grin_accessions().set_index("accession")
    s = data.grin_splits()
    d = s[s["set"] == "dev"]
    k = int(fold[1:])
    lab = cat.set_index("accession")[col]
    yv = lab.map(lambda v: 1.0 if v in pos else (0.0 if v in neg else np.nan)).dropna()
    tr = [a for a in d.loc[d["dev_fold"] != k, "accession"] if a in yv.index]
    te = [a for a in d.loc[d["dev_fold"] == k, "accession"] if a in yv.index]
    G = data.grin_dosage()
    rtr, rte = acc.loc[tr, "row"].to_numpy(), acc.loc[te, "row"].to_numpy()
    return Task(scenario, trait, fold, np.asarray(G[rtr]), yv.loc[tr].to_numpy(float), acc.loc[tr, "region"].to_numpy(),
                "random", np.asarray(G[rte]), yv.loc[te].to_numpy(float), np.array(te),
                acc.loc[te, "region"].to_numpy(), data.grin_snps()["chrom"].to_numpy(), cols=np.arange(G.shape[1]))


# ----------------------------------------------------------------------------- dispatch
def folds(scenario, trait):
    if scenario in GRIN_SCENARIOS:
        return [f for f, _, _ in data.scenario_partitions(scenario)]
    if scenario in SOYNAM_SCENARIOS or scenario == "pc_random":
        return [f"f{k}" for k in range(5)]
    return ["all"]


def get_task(scenario, trait, fold) -> Task:
    if scenario in GRIN_SCENARIOS:
        return grin_task(scenario, trait, fold)
    if scenario in SOYNAM_SCENARIOS:
        return soynam_task(scenario, trait, fold)
    if scenario == "pc_random":
        return pc_task(scenario, trait, fold)
    return transfer_task(scenario, trait, fold)


def traits_for(scenario):
    if scenario in GRIN_SCENARIOS:
        return data.PRIMARY_TRAITS + data.SECONDARY_TRAITS
    if scenario in ("soynam_random", "soynam_lofo", "grin2soynam"):
        return list(SOYNAM_TRAIT)
    if scenario == "pc_random":
        return list(PC_TRAITS)
    return list(NUST_TRAIT)
