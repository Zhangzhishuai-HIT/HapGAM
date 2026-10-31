"""Cross-scenario summaries quoted in the text.

1. Safety: fold-level differences HapGAM-E − EGBLUP (and − GBLUP, − LightGBM) over all confirmatory
   scenarios and traits (share of folds with a gain, largest loss).
2. Computing time per fit (median seconds by model and scenario family, from the result JSONs).
3. Model × scenario coverage (number of completed trait × fold cells; results/tables/coverage.tsv).

usage: summary_stats.py [--coverage-only]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS, ROOT, RUNS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import MODELS  # noqa: E402

CONF = ["C1", "C2", "C3", "C4", "soynam_random", "soynam_lofo", "grin2soynam", "grin2nust"]

def load_all():
    rows = []
    for f in RUNS.glob("*/*/*/*.json"):
        sc, m, t = f.parts[-4], f.parts[-3], f.parts[-2]
        if sc.startswith("_") or m.startswith("_"):
            continue
        d = json.loads(f.read_text())
        rows.append({"scenario": sc, "model": m, "trait": t, "fold": d.get("fold", f.stem), "pearson": d.get("pearson"),
                     "within": d.get("within_group_r"), "seconds": d.get("seconds"), "n_train": d.get("n_train")})
    return pd.DataFrame(rows)


def list_all():
    """Scenario, model, trait and fold of every result file, without reading the files (for the coverage table)."""
    rows = [{"scenario": f.parts[-4], "model": f.parts[-3], "trait": f.parts[-2], "fold": f.stem}
            for f in RUNS.glob("*/*/*/*.json") if not (f.parts[-4].startswith("_") or f.parts[-3].startswith("_"))]
    return pd.DataFrame(rows)


def coverage(df, out):
    cols = ["dev_region"] + CONF + ["pc_random"]
    cov = df[df["scenario"].isin(cols)].groupby(["model", "scenario"]).size().unstack(fill_value=0)
    cov = cov.reindex(index=[m for m in MODELS if m in cov.index and m not in ("hapgam", "hapgam_linear")],
                      columns=[s for s in cols if s in cov.columns])
    cov.to_csv(out / "coverage.tsv", sep="\t")
    print(cov.to_string())


def main():
    out = RESULTS / "tables"
    if "--coverage-only" in sys.argv:
        coverage(list_all(), out)
        return
    df = load_all()
    # 1. safety
    rows = []
    for base in ["gblup", "egblup", "lightgbm", "bayesb"]:
        sub = df[df["scenario"].isin(CONF)]
        val = np.where(sub["scenario"].str.startswith("grin2"), sub["within"], sub["pearson"])
        sub = sub.assign(v=val)
        w = sub.pivot_table(index=["scenario", "trait", "fold"], columns="model", values="v")
        if base not in w or "hapgam_e" not in w:
            continue
        d = (w["hapgam_e"] - w[base]).dropna()
        rows.append({"baseline": base, "fold_comparisons": len(d), "share_gain": float((d > 0).mean()),
                     "median_diff": float(d.median()), "largest_loss": float(d.min()), "largest_gain": float(d.max()),
                     "worst": "/".join(map(str, d.idxmin()))})
    safety = pd.DataFrame(rows)
    safety.to_csv(out / "safety_hapgam_e.tsv", sep="\t", index=False)
    print(safety.round(4).to_string(index=False))
    # 2. time
    t = df.dropna(subset=["seconds"]).copy()
    t["family"] = np.where(t["scenario"].str.startswith("soynam"), "SoyNAM",
                           np.where(t["scenario"].str.startswith("grin2"), "transfer", "GRIN"))
    tt = t.groupby(["model", "family"])["seconds"].median().unstack().round(0)
    tt.to_csv(out / "fit_seconds.tsv", sep="\t")
    print(tt.to_string())
    # 3. coverage
    coverage(df, out)


if __name__ == "__main__":
    main()
