"""Minimal VCF reader producing an int8 ALT-dosage matrix (samples x variants)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

_GT = {
    "0/0": 0, "0|0": 0,
    "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1,
    "1/1": 2, "1|1": 2,
}


def read_vcf(path: Path, chrom_filter=None, max_variants: int | None = None):
    """Return (dosage int8 [n_samples, n_variants], variants DataFrame, sample list).

    Missing or multi-allelic genotype codes are stored as -1.
    ``chrom_filter`` is an optional callable on the CHROM string.
    """
    samples: list[str] = []
    rows: list[np.ndarray] = []
    meta: list[tuple] = []
    lut: dict[str, int] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chrom_filter is not None and not chrom_filter(chrom):
                continue
            fmt = fields[8].split(":")
            gi = fmt.index("GT") if "GT" in fmt else 0
            calls = fields[9:]
            if len(fmt) > 1:
                calls = [c.split(":")[gi] for c in calls]
            codes = np.fromiter((lut.setdefault(c, _GT.get(c, -1)) for c in calls),
                                dtype=np.int8, count=len(calls))
            rows.append(codes)
            meta.append((chrom, int(fields[1]), fields[2], fields[3], fields[4]))
            if max_variants is not None and len(rows) >= max_variants:
                break
    dosage = np.ascontiguousarray(np.vstack(rows).T)
    variants = pd.DataFrame(meta, columns=["chrom", "pos", "id", "ref", "alt"])
    return dosage, variants, samples
