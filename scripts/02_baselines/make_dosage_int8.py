"""Headerless int8 copy of the GRIN dosage matrix, read with readBin by bglr_fit.R for GRIN folds.

Writes data/processed/grin/dosage.int8 (the data bytes of dosage.npy, C order) and dosage.int8.json
({"n", "p", "dtype", "order"}).
usage: make_dosage_int8.py [out_dir]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import PROCESSED  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else PROCESSED / "grin"
d = np.load(PROCESSED / "grin" / "dosage.npy", mmap_mode="r")
np.ascontiguousarray(d, dtype=np.int8).tofile(out / "dosage.int8")
(out / "dosage.int8.json").write_text(json.dumps({"n": int(d.shape[0]), "p": int(d.shape[1]), "dtype": "int8", "order": "C"}))
print(f"wrote {out / 'dosage.int8'} ({d.shape[0]} x {d.shape[1]})")
