"""Create the frozen GRIN split design.

  lockbox_japan   all accessions from Japan: unseen-region lockbox, never used in development
  lockbox_random  15% of the remaining accessions (stratified by region x MG group)
  dev             the rest; development uses only these accessions

Columns written to splits/grin_splits.tsv:
  set, dev_fold (5-fold random within dev), dev_region_fold (region name within dev,
  'train_only' for Other/Unknown), region (for the confirmatory leave-region-out rotation),
  mg_fold (for the confirmatory leave-MG-group-out rotation).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import PROCESSED, SPLITS  # noqa: E402

SEED = 20261020
LOCKBOX_FRAC = 0.15
MAIN_REGIONS = ["China", "Korea", "Japan", "S/SE Asia", "Americas", "Europe", "Russia/C.Asia"]


def main():
    acc = pd.read_csv(PROCESSED / "grin" / "accessions.tsv", sep="\t", keep_default_na=False, na_values={"mg": [""], "latitude": [""], "longitude": [""]})
    rng = np.random.default_rng(SEED)
    s = pd.DataFrame({"accession": acc["accession"], "row": acc["row"], "region": acc["region"],
                      "mg_group": acc["mg_group"]})
    s["set"] = "dev"
    s.loc[s["region"] == "Japan", "set"] = "lockbox_japan"
    # stratified random lockbox among non-Japan accessions
    pool = s[s["set"] == "dev"]
    for _, g in pool.groupby(["region", "mg_group"]):
        k = int(round(LOCKBOX_FRAC * len(g)))
        if k:
            pick = rng.choice(g.index.to_numpy(), size=k, replace=False)
            s.loc[pick, "set"] = "lockbox_random"
    # 5-fold random CV within dev (stratified by region x MG group)
    s["dev_fold"] = -1
    dev = s[s["set"] == "dev"]
    for _, g in dev.groupby(["region", "mg_group"]):
        idx = rng.permutation(g.index.to_numpy())
        s.loc[idx, "dev_fold"] = np.arange(len(idx)) % 5
    s["dev_region_fold"] = np.where(s["set"] != "dev", "",
                                    np.where(s["region"].isin(MAIN_REGIONS), s["region"], "train_only"))
    s["region_fold"] = np.where(s["region"].isin(MAIN_REGIONS), s["region"], "train_only")
    s["mg_fold"] = np.where(s["mg_group"] == "NA", "train_only", s["mg_group"])
    SPLITS.mkdir(parents=True, exist_ok=True)
    path = SPLITS / "grin_splits.tsv"
    if path.exists():
        raise SystemExit(f"{path} exists; splits are frozen and must not be regenerated")
    s.to_csv(path, sep="\t", index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (SPLITS / "grin_splits.sha256").write_text(f"{digest}  grin_splits.tsv\n")
    print(s["set"].value_counts().to_dict())
    print(s[s["set"] == "dev"]["dev_region_fold"].value_counts().to_dict())
    print(pd.crosstab(s["set"], s["mg_group"]))
    print("sha256", digest)


if __name__ == "__main__":
    main()
