"""Parse the SoySNP50K (Wm82.a2) VCF into an int8 dosage matrix.

Outputs (data/interim/soysnp50k/):
  dosage.npy      int8 [n_samples, n_variants], ALT dosage 0/1/2, -1 = missing
  variants.tsv    chrom, pos, id, ref, alt (chromosomes Gm01-Gm20 only)
  samples.tsv     vcf_id, accession (canonical id)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.grin import canonical_id  # noqa: E402
from hapgam.paths import INTERIM, RAW  # noqa: E402
from hapgam.vcf import read_vcf  # noqa: E402


def main():
    vcf = RAW / "soysnp50k_gnm2" / "glyma.Wm82.gnm2.div.Song_Hyten_2015.vcf.gz"
    out = INTERIM / "soysnp50k"
    out.mkdir(parents=True, exist_ok=True)
    dosage, variants, samples = read_vcf(vcf, chrom_filter=lambda c: ".Gm" in c)
    variants["chrom"] = variants["chrom"].str.replace("glyma.Wm82.gnm2.", "", regex=False)
    order = np.lexsort((variants["pos"].to_numpy(), variants["chrom"].to_numpy()))
    variants = variants.iloc[order].reset_index(drop=True)
    dosage = np.ascontiguousarray(dosage[:, order])
    sam = pd.DataFrame({"vcf_id": samples, "accession": [canonical_id(s) for s in samples]})
    np.save(out / "dosage.npy", dosage)
    variants.to_csv(out / "variants.tsv", sep="\t", index=False)
    sam.to_csv(out / "samples.tsv", sep="\t", index=False)
    miss = (dosage < 0).mean()
    het = (dosage == 1).sum() / max((dosage >= 0).sum(), 1)
    print(f"samples={dosage.shape[0]} variants={dosage.shape[1]} missing={miss:.4f} het={het:.4f}")
    print(f"duplicated canonical ids: {sam['accession'].duplicated().sum()}")


if __name__ == "__main__":
    main()
