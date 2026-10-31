"""Freeze code and configuration before the confirmatory analyses (C1-C7).

Writes docs/FREEZE.json with SHA-256 digests of every source, script, job and split file plus the
pre-registration, the chosen primary model and the HapGAM grid, then removes write permission
from everything that produces predictions (src/, scripts/run_model.py, scripts/01_data/,
scripts/02_baselines/, splits/).  Analysis and plotting scripts (scripts/04_analysis/) stay
editable; they do not affect predictions and are hashed again at submission.  Confirmatory outputs
record the code hash (runner.code_hash), which must match the frozen hash.

usage: freeze.py --primary hapgam_e --note "..."
"""

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hapgam import api  # noqa: E402
from hapgam.runner import code_hash  # noqa: E402


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    out = ROOT / "docs" / "FREEZE.json"
    if out.exists():
        raise SystemExit("already frozen: docs/FREEZE.json exists")
    files = {}
    for sub in ("src", "scripts", "jobs", "splits"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[str(p.relative_to(ROOT))] = digest(p)
    files["docs/PREREGISTRATION.md"] = digest(ROOT / "docs" / "PREREGISTRATION.md")
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "primary_model": args.primary,
           "runner_code_hash": code_hash(), "hapgam_grid": api.HAPGAM_GRID, "hapgam_base": api.HAPGAM_BASE,
           "note": args.note, "files": files}
    out.write_text(json.dumps(rec, indent=1))
    locked = [ROOT / "scripts" / "run_model.py"]
    for sub in ("src", "scripts/01_data", "scripts/02_baselines", "splits"):
        locked += [p for p in (ROOT / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for p in locked:
        os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({k: v for k, v in rec.items() if k != "files"}, indent=1))
    print(f"{len(files)} files hashed -> {out}")


if __name__ == "__main__":
    main()
