"""Build the external (confirmatory) datasets aligned to the GRIN SNP coding.

NUST  (Northern Uniform Soybean Tests): BLUEs per line x MG test, 5,158-SNP VCF (Wm82.a2).
SoyNAM: 5,176 RILs, imputed Wm82.a2 genotypes (Song_NAM_2021b, 0/1 relative to IA3023),
        phenotypes from the SoyNAM R package (QA version; flowering from the raw version).

Genotypes are re-coded to the GRIN ALT-dosage coding (0/1/2) on the GRIN QC SNPs; SNPs whose
alleles cannot be matched are dropped.  Outputs in data/processed/{nust,soynam}/.
usage: build_external_datasets.py nust|soynam
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import data  # noqa: E402
from hapgam.paths import INTERIM, PROCESSED, RAW  # noqa: E402
from hapgam.vcf import read_vcf  # noqa: E402

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def grin_snp_key():
    s = data.grin_snps().copy()
    s["key"] = s["chrom"] + ":" + s["pos"].astype(str)
    return s


def align(ref, alt, a0, a1):
    """Return +1 if (a0,a1)==(ref,alt), -1 if swapped, 0 if unmatched (incl. strand flips)."""
    if (a0, a1) == (ref, alt):
        return 1
    if (a0, a1) == (alt, ref):
        return -1
    if (COMP.get(a0), COMP.get(a1)) == (ref, alt) and {ref, alt} not in ({"A", "T"}, {"C", "G"}):
        return 1
    if (COMP.get(a0), COMP.get(a1)) == (alt, ref) and {ref, alt} not in ({"A", "T"}, {"C", "G"}):
        return -1
    return 0


def build_nust():
    out = PROCESSED / "nust"
    out.mkdir(parents=True, exist_ok=True)
    dos, var, samples = read_vcf(RAW / "nust" / "data" / "NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz")
    var["chrom"] = "Gm" + var["chrom"].astype(str).str.zfill(2)
    var["key"] = var["chrom"] + ":" + var["pos"].astype(str)
    g = grin_snp_key()
    gi = dict(zip(g["key"], range(len(g))))
    X = np.full((len(samples), len(g)), -1, np.int8)
    used, sign = [], []
    for j, r in var.iterrows():
        k = gi.get(r["key"])
        if k is None:
            continue
        sgn = align(g.at[k, "ref"], g.at[k, "alt"], r["ref"], r["alt"])
        if sgn == 0:
            continue
        col = dos[:, j].astype(np.int16)
        col = np.where(col < 0, -1, col if sgn == 1 else 2 - col)
        X[:, k] = col
        used.append(k)
    keep = np.array(sorted(set(used)))
    blue = pd.read_csv(RAW / "nust" / "results" / "BLUE_NUST_1993_2020_GermplasmId_Output_Table.csv")
    blue = blue[blue["GermplasmId_name"].isin(samples)].copy()
    sidx = {s: i for i, s in enumerate(samples)}
    blue["row"] = blue["GermplasmId_name"].map(sidx)
    np.save(out / "dosage_grin_coded.npy", X)
    np.save(out / "grin_snp_index.npy", keep)
    blue.to_csv(out / "blues.tsv", sep="\t", index=False)
    pd.DataFrame({"sample": samples}).to_csv(out / "samples.tsv", sep="\t", index=False)
    fc = pd.read_csv(RAW / "nust" / "data" / "NUST_Flower_Pubescence_Color_Input_Data.csv")
    fc.to_csv(out / "flower_pubescence.tsv", sep="\t", index=False)
    info = {"n_samples": len(samples), "nust_snps": int(len(var)), "matched_grin_snps": int(len(keep)),
            "blue_rows_genotyped": int(len(blue)), "lines": int(blue["GermplasmId_name"].nunique()),
            "missing_rate_matched": float((X[:, keep] < 0).mean())}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


def build_soynam():
    out = PROCESSED / "soynam"
    out.mkdir(parents=True, exist_ok=True)
    path = (Path("/home/user/zhangzhishuai/myhermes/down_data/LIS_glycine_max_diversity")
            / "Wm82.gnm2.div.Song_NAM_2021b" / "glyma.Wm82.gnm2.div.Song_NAM_2021b.imputed_5176RILs_423kSNPs_01.txt.gz")
    g = grin_snp_key()
    gi = dict(zip(g["key"], range(len(g))))
    lut = {"0/0": 0, "1/1": 2, "0/1": 1, "1/0": 1}
    cols, keep_k, n_seen = [], [], 0
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        lines = header[8:]                      # first sample column is IA3023 (parent), then RILs
        for line in fh:
            n_seen += 1
            f = line.rstrip("\n").split("\t")
            key = f[0].replace("glyma.Wm82.gnm2.", "") + ":" + f[1]
            k = gi.get(key)
            if k is None:
                continue
            sgn = align(g.at[k, "ref"], g.at[k, "alt"], f[3], f[4])   # a0 = IA3023 allele
            if sgn == 0:
                continue
            c = np.fromiter((lut.get(x, -1) for x in f[8:]), dtype=np.int16, count=len(lines))
            c = np.where(c < 0, -1, c if sgn == 1 else 2 - c)
            cols.append(c.astype(np.int8))
            keep_k.append(k)
    X = np.full((len(lines), len(g)), -1, np.int8)
    X[:, keep_k] = np.stack(cols, 1)
    np.save(out / "dosage_grin_coded.npy", X)
    np.save(out / "grin_snp_index.npy", np.array(sorted(keep_k)))
    pd.DataFrame({"line": lines}).to_csv(out / "lines.tsv", sep="\t", index=False)
    info = {"n_lines": len(lines), "snps_in_file": n_seen, "matched_grin_snps": len(keep_k),
            "missing_rate_matched": float((X[:, keep_k] < 0).mean())}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    {"nust": build_nust, "soynam": build_soynam}[sys.argv[1]]()
