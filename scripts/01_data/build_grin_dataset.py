"""Build the GRIN x SoySNP50K analysis dataset (Glycine max only).

Outputs (data/processed/grin/):
  accessions.tsv     one row per genotyped G. max accession with phenotype data
  records.tsv.gz     cleaned quantitative records (accession, trait, batch, value)
  categorical.tsv    accession x categorical descriptors (modal value)
  geno_qc.json       genotype QC summary
  snps.tsv           SNPs retained after phenotype-free QC
  dosage.npy         int8 [n_accessions, n_snps] (0/2 homozygous, 1 het), mode-imputed
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.grin import load_all, region_of  # noqa: E402
from hapgam.paths import INTERIM, PROCESSED, RAW  # noqa: E402

# plausible ranges (records outside are treated as data-entry errors)
RANGES = {
    "maturity_doy": (180, 350), "flowering_doy": (120, 300), "height_cm": (5, 250),
    "seed_weight_g": (1, 60), "yield_mg_ha": (0.05, 8), "oil_pct": (5, 30),
    "protein_pct": (25, 62), "lodging_score": (1, 5), "maturity_group": (-2, 10),
    "stem_term_score": (1, 5),
}
MAD_K = 6.0
MIN_BATCH = 20


def mg_group(mg: float) -> str:
    if np.isnan(mg):
        return "NA"
    if mg <= 0:
        return "MG000-0"
    if mg <= 2:
        return "MGI-II"
    if mg <= 4:
        return "MGIII-IV"
    if mg <= 6:
        return "MGV-VI"
    return "MGVII-X"


def main():
    out = PROCESSED / "grin"
    out.mkdir(parents=True, exist_ok=True)
    sam = pd.read_csv(INTERIM / "soysnp50k" / "samples.tsv", sep="\t")
    sam["vcf_row"] = np.arange(len(sam))
    df = load_all(RAW / "grin_descriptors")
    df = df[(df["taxon"] == "Glycine max") & df["accession"].isin(set(sam["accession"]))].copy()

    # ---- quantitative records -------------------------------------------------
    q = df[df["kind"] == "quantitative"].dropna(subset=["value"]).copy()
    log = {}
    keep = np.ones(len(q), bool)
    for t, (lo, hi) in RANGES.items():
        m = (q["trait"] == t).to_numpy()
        bad = m & ~q["value"].between(lo, hi).to_numpy()
        log[f"{t}_out_of_range"] = int(bad.sum())
        keep &= ~bad
    q = q[keep]
    # robust within-batch outliers
    med = q.groupby(["trait", "batch"])["value"].transform("median")
    mad = q.groupby(["trait", "batch"])["value"].transform(lambda s: (s - s.median()).abs().median())
    z = (q["value"] - med) / (1.4826 * mad.replace(0, np.nan))
    out_mask = z.abs() > MAD_K
    log["mad_outliers"] = q.loc[out_mask, "trait"].value_counts().to_dict()
    q = q[~out_mask.fillna(False)]
    # drop tiny batches
    bn = q.groupby(["trait", "batch"])["value"].transform("size")
    log["small_batch_records"] = q.loc[bn < MIN_BATCH, "trait"].value_counts().to_dict()
    q = q[bn >= MIN_BATCH]
    # average duplicate (accession, trait, batch) records
    q = q.groupby(["accession", "trait", "batch"], as_index=False)["value"].mean()

    # ---- accession metadata ---------------------------------------------------
    def mode(s):
        s = s[s.astype(str) != ""]
        return s.value_counts().index[0] if len(s) else ""

    meta = df.groupby("accession").agg(
        origin=("origin", mode), source_type=("source_type", mode),
        latitude=("latitude", "median"), longitude=("longitude", "median"),
        label=("accession_label", "first"), plant_name=("plant_name", mode)).reset_index()
    mg = q[q["trait"] == "maturity_group"].groupby("accession")["value"].median()
    meta["mg"] = meta["accession"].map(mg)
    meta["mg_group"] = meta["mg"].map(mg_group)
    meta["region"] = meta["origin"].map(region_of)
    meta = meta.merge(sam[["accession", "vcf_id", "vcf_row"]], on="accession", how="inner")
    meta = meta[meta["accession"].isin(set(q["accession"]))].sort_values("vcf_row").reset_index(drop=True)
    meta["row"] = np.arange(len(meta))

    # ---- categorical descriptors ----------------------------------------------
    c = df[df["kind"] == "categorical"]
    cat = c.groupby(["accession", "trait"])["raw_value"].agg(mode).unstack()
    cat = cat.reindex(meta["accession"]).reset_index()

    # ---- genotypes: phenotype-free QC on the analysis accessions ----------------
    dos_all = np.load(INTERIM / "soysnp50k" / "dosage.npy", mmap_mode="r")
    var = pd.read_csv(INTERIM / "soysnp50k" / "variants.tsv", sep="\t")
    X = np.asarray(dos_all[meta["vcf_row"].to_numpy()])
    called = X >= 0
    call_rate = called.mean(0)
    alt = np.where(called, X, 0).sum(0) / np.maximum(2 * called.sum(0), 1)
    maf = np.minimum(alt, 1 - alt)
    snp_keep = (call_rate >= 0.95) & (maf >= 0.01)
    X = X[:, snp_keep]
    var = var[snp_keep].reset_index(drop=True)
    # mode imputation (inbred panel: impute the major homozygote)
    major = np.where(alt[snp_keep] > 0.5, 2, 0).astype(np.int8)
    miss_r, miss_c = np.nonzero(X < 0)
    X[miss_r, miss_c] = major[miss_c]
    sample_het = (X == 1).mean(1)

    np.save(out / "dosage.npy", np.ascontiguousarray(X))
    var.to_csv(out / "snps.tsv", sep="\t", index=False)
    meta["het_rate"] = sample_het
    meta.to_csv(out / "accessions.tsv", sep="\t", index=False)
    q.to_csv(out / "records.tsv.gz", sep="\t", index=False)
    cat.to_csv(out / "categorical.tsv", sep="\t", index=False)
    summary = {
        "n_accessions": int(len(meta)), "n_snps_raw": int(len(snp_keep)), "n_snps_qc": int(snp_keep.sum()),
        "snp_filter": "call_rate>=0.95 & MAF>=0.01 on analysis accessions (phenotype-free)",
        "imputed_genotypes": int(len(miss_r)), "cleaning": log,
        "records_per_trait": q["trait"].value_counts().to_dict(),
        "accessions_per_trait": q.groupby("trait")["accession"].nunique().to_dict(),
        "regions": meta["region"].value_counts().to_dict(),
        "mg_groups": meta["mg_group"].value_counts().to_dict(),
    }
    (out / "geno_qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
