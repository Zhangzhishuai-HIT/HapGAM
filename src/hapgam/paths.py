"""Project paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "splits"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
