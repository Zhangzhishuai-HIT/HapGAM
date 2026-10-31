"""Shared experiment runner: partitions, targets, prediction files, metrics."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .metrics import summary
from .paths import ROOT, RUNS
from .targets import safe


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / "hapgam").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_dir(scenario: str, model: str, trait: str) -> Path:
    d = RUNS / scenario / model / trait
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fold(out: Path, fold: str, td: data.TraitData, pred: np.ndarray, info: dict):
    acc = data.grin_accessions()
    df = pd.DataFrame({"row": td.test_rows, "accession": acc.loc[td.test_rows, "accession"].to_numpy(),
                       "y": td.y_test, "pred": pred})
    df.to_csv(out / f"{safe(fold)}.pred.tsv", sep="\t", index=False)
    m = summary(td.y_test, pred, td.test_records, td.test_rows)
    m.update({"fold": fold, "n_train": int(len(td.train_rows)), "code": code_hash(),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    m.update(info)
    (out / f"{safe(fold)}.json").write_text(json.dumps(m, indent=1, default=float))
    return m
