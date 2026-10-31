"""Collect fold metrics from runs/<scenario>/<model>/<trait>/*.json into tables.

usage: summarize.py --scenario dev_region [--metric pearson] [--models gblup lightgbm ...]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS, RUNS  # noqa: E402


def collect(scenario: str) -> pd.DataFrame:
    rows = []
    for f in (RUNS / scenario).glob("*/*/*.json"):
        model, trait = f.parts[-3], f.parts[-2]
        if model.startswith("_"):
            continue
        d = json.loads(f.read_text())
        rows.append({"scenario": scenario, "model": model, "trait": trait, "fold": d["fold"], "n": d["n"],
                     "pearson": d["pearson"], "spearman": d["spearman"], "rmse": d["rmse"],
                     "top20": d["top20"], "within_batch_r": d.get("within_batch_r"),
                     "within_group_r": d.get("within_group_r")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--metric", default="pearson")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--matched", action="store_true", help="only folds where every listed model finished")
    args = ap.parse_args()
    df = collect(args.scenario)
    if args.models:
        df = df[df["model"].isin(args.models)]
    if args.matched:
        n_models = df["model"].nunique()
        ok = df.groupby(["trait", "fold"])["model"].transform("nunique") == n_models
        df = df[ok]
    if df.empty:
        print("no results")
        return
    if not (args.models or args.matched):
        RESULTS.joinpath("metrics").mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS / "metrics" / f"{args.scenario}_folds.tsv", sep="\t", index=False)
    tab = df.pivot_table(index="trait", columns="model", values=args.metric, aggfunc="mean")
    cnt = df.pivot_table(index="trait", columns="model", values=args.metric, aggfunc="count")
    pd.set_option("display.width", 200)
    print(f"== {args.scenario}: mean {args.metric} over folds")
    print(tab.round(4).to_string())
    print("folds per cell:")
    print(cnt.to_string())


if __name__ == "__main__":
    main()
