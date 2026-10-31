"""Paired comparison of a focal model against baselines, per trait.

For each trait the fold-level test predictions of the focal model and each baseline are
aligned by test individual.  The statistic is the mean over folds of Pearson r (pooled metric)
or of the mean within-group r (families / MG tests / GRIN batches).  Uncertainty: 2,000
bootstrap replicates resampling test individuals within each fold (pooled metric) or whole
groups within each fold (within-group metric).  Two-sided bootstrap p-values are Holm-adjusted
across traits.

usage: compare_models.py --scenario C3 --focal hapgam_e [--baselines gblup egblup lightgbm ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam.paths import RESULTS, RUNS  # noqa: E402

B_REPS = 2000
# random k-fold scenarios: also report the corrected resampled t-test of Nadeau and Bengio (2003)
RANDOM_KFOLD = {"soynam_random", "dev_random"}


def _r(y, p):
    y = y - y.mean()
    p = p - p.mean()
    d = np.sqrt((y * y).sum() * (p * p).sum())
    return float((y * p).sum() / d) if d > 0 else np.nan


def load_preds(scenario, model, trait):
    d = RUNS / scenario / model / trait
    out = {}
    for f in sorted(d.glob("*.pred.tsv")):
        df = pd.read_csv(f, sep="\t", keep_default_na=False, na_values={"y": [""], "pred": [""]})
        out[f.name[: -len(".pred.tsv")]] = df
    return out


def fold_stat(y, pa, pb, groups, within):
    if not within:
        return _r(y, pa) - _r(y, pb), _r(y, pa), _r(y, pb)
    ra, rb = [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 20:
            ra.append(_r(y[m], pa[m]))
            rb.append(_r(y[m], pb[m]))
    ra, rb = np.array(ra), np.array(rb)
    ok = np.isfinite(ra) & np.isfinite(rb)
    return float(np.mean(ra[ok] - rb[ok])), float(np.mean(ra[ok])), float(np.mean(rb[ok]))


def compare(scenario, focal, base, trait, within=False, seed=0):
    A, Bm = load_preds(scenario, focal, trait), load_preds(scenario, base, trait)
    folds = sorted(set(A) & set(Bm))
    if not folds:
        return None
    rng = np.random.default_rng(seed)
    data = []
    for f in folds:
        a = A[f].set_index("id")
        b = Bm[f].set_index("id")
        ids = a.index.intersection(b.index)
        y = a.loc[ids, "y"].to_numpy(float)
        ok = np.isfinite(y)
        data.append((y[ok], a.loc[ids, "pred"].to_numpy(float)[ok], b.loc[ids, "pred"].to_numpy(float)[ok],
                     a.loc[ids, "group"].astype(str).to_numpy()[ok]))
    fold_stats = [fold_stat(y, pa, pb, g, within) for y, pa, pb, g in data]
    diff = float(np.mean([s_[0] for s_ in fold_stats]))
    ra = float(np.mean([s_[1] for s_ in fold_stats]))
    rb = float(np.mean([s_[2] for s_ in fold_stats]))
    boots = np.empty(B_REPS)
    for i in range(B_REPS):
        vals = []
        for y, pa, pb, g in data:
            if within:
                ug = np.unique(g)
                pick = rng.choice(ug, len(ug), replace=True)
                idx = np.concatenate([np.flatnonzero(g == u) for u in pick])
                # relabel resampled groups so duplicates stay separate groups
                lab = np.concatenate([np.full((g == u).sum(), j) for j, u in enumerate(pick)])
                vals.append(fold_stat(y[idx], pa[idx], pb[idx], lab, True)[0])
            else:
                idx = rng.integers(0, len(y), len(y))
                vals.append(_r(y[idx], pa[idx]) - _r(y[idx], pb[idx]))
        boots[i] = np.mean(vals)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    # predictions identical to numerical precision (e.g. HapGAM-E with M = 0 vs EGBLUP): no difference to test
    same = max(np.max(np.abs(pa - pb)) / max(np.std(pb), 1e-12) for _, pa, pb, _ in data) < 1e-6
    if same:
        p = 1.0
    pos = int(sum(s_[0] > 0 for s_ in fold_stats))
    out = {"trait": trait, "baseline": base, "folds": len(folds), "r_focal": ra, "r_base": rb, "diff": diff,
           "ci_lo": float(lo), "ci_hi": float(hi), "p_boot": float(max(p, 1.0 / B_REPS)),
           "folds_positive": pos}
    if scenario in RANDOM_KFOLD and len(folds) > 2:
        d = np.array([s_[0] for s_ in fold_stats])
        n_te = np.array([len(x[0]) for x in data], float)
        ratio = float(np.mean(n_te / (n_te.sum() - n_te)))
        var = (1.0 / len(d) + ratio) * d.var(ddof=1)
        t = d.mean() / np.sqrt(var) if var > 0 else np.nan
        out["nb_t"] = float(t)
        out["nb_p"] = float(2 * stats.t.sf(abs(t), len(d) - 1)) if np.isfinite(t) else np.nan
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--focal", required=True)
    ap.add_argument("--baselines", nargs="+", default=None)
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--within", action="store_true", help="use the within-group metric")
    args = ap.parse_args()
    root = RUNS / args.scenario
    baselines = args.baselines or sorted(p.name for p in root.iterdir()
                                         if p.is_dir() and not p.name.startswith("_") and p.name != args.focal)
    traits = args.traits or sorted(p.name for p in (root / args.focal).iterdir() if p.is_dir())
    rows = []
    for b in baselines:
        res = [compare(args.scenario, args.focal, b, t, args.within) for t in traits]
        res = [r for r in res if r is not None]
        if not res:
            continue
        adj = holm([r["p_boot"] for r in res])
        # the corrected resampled t-test (random k-fold scenarios) is Holm-adjusted across traits as well (protocol 7)
        nb = holm([r["nb_p"] for r in res]) if all("nb_p" in r and np.isfinite(r["nb_p"]) for r in res) else None
        for k, (r, a) in enumerate(zip(res, adj)):
            r["p_holm"] = float(a)
            if nb is not None:
                r["nb_p_holm"] = float(nb[k])
            rows.append(r)
    df = pd.DataFrame(rows)
    out = RESULTS / "tables" / f"compare_{args.scenario}_{args.focal}{'_within' if args.within else ''}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
