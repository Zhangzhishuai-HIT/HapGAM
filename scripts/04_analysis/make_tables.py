"""Manuscript tables from fold-level results (runs/<scenario>/<model>/<trait>/<fold>.json).

Writes results/tables/all_metrics.tsv (every scenario x model x trait) and Markdown + TSV tables
for the manuscript under manuscript/tables/:
  table2_test_sets   accuracy in the two GRIN test sets (C2 Japanese, C1 random), all traits
  table3_grouped     C3 and across-population prediction (C4, C5), primary traits: mean r and groups won by HapGAM
  tableS_<scenario>  every reported model x trait for one scenario (supplementary)
  tableS_compare_<scenario>  bootstrap comparisons of HapGAM (from compare_models.py output)
all_metrics.tsv covers every model and scenario that was run; the manuscript tables report the models in REPORTED and
the scenarios in REPORTED_SCEN (the other runs accompany the code).

Cells show the mean over outer folds of Pearson r; '–' marks a model that was not run or has not
finished every fold of that trait (incomplete cells are never averaged into a table).

usage: make_tables.py [--from-tsv]
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapgam import scenarios  # noqa: E402
from hapgam.metrics import auc  # noqa: E402
from hapgam.paths import RESULTS, ROOT, RUNS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import collect  # noqa: E402

MODELS = {"gblup": "GBLUP", "egblup": "EGBLUP", "rkhs": "RKHS", "hapgblup": "HapGBLUP", "bayesb": "BayesB",
          "lightgbm": "LightGBM", "xgboost": "XGBoost", "mlp": "MLP", "dnngp": "DNNGP", "hapgam": "HapGAM-A",
          "hapgam_linear": "HapGAM-A linear", "hapgam_e_linear": "HapGAM-linear", "hapgam_e": "HapGAM",
          "ens_hapgam_e_lightgbm": "HapGAM + LightGBM",
          "ens_gblup_lightgbm_bayesb": "GBLUP + LightGBM + BayesB"}
# models listed in the tables
REPORTED = [m for m in MODELS if m in ("gblup", "egblup", "bayesb", "lightgbm", "hapgam_e_linear", "hapgam_e",
                                       "ens_hapgam_e_lightgbm", "ens_gblup_lightgbm_bayesb")]
SINGLES = ["gblup", "egblup", "bayesb", "lightgbm", "hapgam_e"]
TRAITS = {"maturity_doy": "Maturity date", "flowering_doy": "Flowering date", "height_cm": "Plant height",
          "seed_weight_g": "100-seed weight", "yield_mg_ha": "Seed yield", "protein_pct": "Protein",
          "oil_pct": "Oil", "lodging_score": "Lodging", "maturity_group": "Maturity group"}
PRIMARY = ["maturity_doy", "flowering_doy", "height_cm", "seed_weight_g"]
SCEN = {"dev_region": "Training-set cross-validation", "dev_random": "Training set, random five-fold",
        "C1": "C1 random test set", "C2": "C2 Japanese test set", "C3": "C3 leave-one-region-out",
        "C4": "Leave-one-maturity-group-out (not reported)", "soynam_random": "SoyNAM random five-fold (not reported)",
        "soynam_lofo": "SoyNAM leave-families-out (not reported)", "grin2soynam": "C4 GRIN → SoyNAM",
        "grin2nust": "C5 GRIN → NUST"}
# scenarios reported in the manuscript (internal key C4, leave-one-maturity-group-out, and SoyNAM cross-validation are not)
REPORTED_SCEN = ["dev_region", "C1", "C2", "C3", "grin2soynam", "grin2nust"]
# within-group metric per scenario: GRIN evaluation batches; SoyNAM families; NUST MG trials
WITHIN = {"grin2soynam": "within_group_r", "grin2nust": "within_group_r", "soynam_random": "within_group_r",
          "soynam_lofo": "within_group_r"}
OUT = ROOT / "manuscript" / "tables"


@lru_cache(maxsize=None)
def folds_of(scenario, trait):
    """Outer folds of a scenario (cached: GRIN fold lists are derived from the frozen split file)."""
    return tuple(scenarios.folds(scenario, trait))


def complete_cells(df, scenario):
    """Keep (model, trait) cells that have every outer fold of the scenario."""
    keep = []
    for (m, t), g in df.groupby(["model", "trait"]):
        try:
            need = set(folds_of(scenario, t))
        except Exception:
            need = set(g["fold"])
        if need <= set(g["fold"]):
            keep.append(g[g["fold"].isin(need)])
    return pd.concat(keep) if keep else df.iloc[:0]


def long_table():
    rows = []
    for sc in SCEN:
        if not (RUNS / sc).exists():
            continue
        df = collect(sc)
        if df.empty:
            continue
        df = complete_cells(df, sc)
        wcol = WITHIN.get(sc, "within_batch_r")
        for (m, t), g in df.groupby(["model", "trait"]):
            rows.append({"scenario": sc, "model": m, "trait": t, "folds": len(g), "n_test": int(g["n"].sum()),
                         "pearson": g["pearson"].mean(), "pearson_sd": g["pearson"].std(ddof=1) if len(g) > 1 else np.nan,
                         "pearson_min": g["pearson"].min(), "spearman": g["spearman"].mean(),
                         "top20": g["top20"].mean(), "within_r": pd.to_numeric(g[wcol], errors="coerce").mean()})
    return pd.DataFrame(rows)


def fmt(v, best=False, nd=3):
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:.{nd}f}"
    return f"**{s}**" if best else s


def sgn(v, nd=3):
    """+0.012 / -0.004 with 0.000 instead of a signed zero."""
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:+.{nd}f}"
    return f"{0:.{nd}f}" if float(s) == 0 else s


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "–"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}" if p < 0.01 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def fmt_p_perm(p):
    """Permutation P (5,000 permutations): '< 0.0002' when no permutation reached the observed value."""
    return "<\u00a00.0002" if p == 0 else (f"{p:.2f}" if p >= 0.1 else f"{p:.2g}")


def group_row(label, ncol):
    return [f"*{label}*"] + [""] * (ncol - 1)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def wide(long, sc, models, traits, metric="pearson"):
    sub = long[(long["scenario"] == sc) & long["model"].isin(models) & long["trait"].isin(traits)]
    return sub.pivot_table(index="trait", columns="model", values=metric).reindex(index=traits, columns=models)


def table2(long):
    models = SINGLES
    header = ["Trait"] + [MODELS[m] for m in models]
    rows, tsv = [], []
    for sc in ["C2", "C1"]:
        w = wide(long, sc, models, list(TRAITS))
        rows.append([f"*{SCEN[sc]}*"] + [""] * len(models))
        for t in TRAITS:
            vals = w.loc[t].to_numpy(float)
            best = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
            rows.append([TRAITS[t]] + [fmt(v, np.isfinite(v) and fmt(v) == fmt(best)) for v in vals])
            tsv.append([sc, t] + list(vals))
    (OUT / "table2_test_sets.md").write_text(md_table(header, rows))


def groups_won(sc, trait, focal, base):
    a = {p.stem: json.loads(p.read_text())["pearson"] for p in (RUNS / sc / focal / trait).glob("*.json")}
    b = {p.stem: json.loads(p.read_text())["pearson"] for p in (RUNS / sc / base / trait).glob("*.json")}
    common = sorted(set(a) & set(b))
    return sum(a[f] > b[f] for f in common), len(common)


def table3(long):
    models = SINGLES
    header = ["Trait"] + [MODELS[m] for m in models] + ["Wins vs EGBLUP", "Wins vs LightGBM"]
    rows, tsv = [], []
    for sc in ["C3", "grin2soynam", "grin2nust"]:
        traits = [t for t in PRIMARY if t in scenarios.traits_for(sc)]
        metric = "within_r" if sc.startswith("grin2") else "pearson"
        w = wide(long, sc, models, traits, metric)
        rows.append(group_row(SCEN[sc], len(header)))
        for t in traits:
            vals = w.loc[t].to_numpy(float)
            best = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
            won = []
            for base in ["egblup", "lightgbm"]:
                k, n = groups_won(sc, t, "hapgam_e", base)
                complete = np.isfinite(vals[models.index(base)]) and np.isfinite(vals[models.index("hapgam_e")])
                won.append(f"{k}/{n}" if n > 1 and complete else "–")
            rows.append([TRAITS[t]] + [fmt(v, np.isfinite(v) and fmt(v) == fmt(best)) for v in vals] + won)
            tsv.append([sc, t, metric] + list(vals) + won)
    (OUT / "table3_grouped.md").write_text(md_table(header, rows))


def table_s(long):
    """Every reported model (rows) x trait (columns) for one scenario; bold = most accurate per trait.
    C1 and C2 are in Table 2."""
    for sc in [s for s in REPORTED_SCEN if s not in ("C1", "C2")]:
        sub = long[long["scenario"] == sc]
        if sub.empty:
            continue
        models = [m for m in REPORTED if m in set(sub["model"])]
        traits = [t for t in TRAITS if t in set(sub["trait"])]
        w = wide(long, sc, models, traits)
        f = sub.pivot_table(index="trait", columns="model", values="folds").reindex(index=traits, columns=models)
        best = w.max(axis=1)
        rows = []
        for m in models:
            if m.startswith("ens_") and not any(r[0] == "*Ensembles*" for r in rows):
                rows.append(group_row("Ensembles", len(traits) + 1))
            rows.append([MODELS[m]] + [fmt(w.at[t, m], np.isfinite(w.at[t, m]) and fmt(w.at[t, m]) == fmt(best[t]))
                                       for t in traits])
        nfold = int(np.nanmax(f.to_numpy())) if np.isfinite(f.to_numpy(float)).any() else 0
        (OUT / f"tableS_{sc}.md").write_text(md_table(["Model"] + [TRAITS[t] for t in traits], rows))
        print(f"{sc}: {len(models)} models x {len(traits)} traits (max folds {nfold})")


def table_s_compare():
    """Paired comparisons of HapGAM-E (or the ensemble) with each baseline, grouped by baseline, traits in fixed order."""
    for f in sorted((RESULTS / "tables").glob("compare_*_hapgam_e*.tsv")):
        if "_ens_" in f.stem:          # ensemble comparisons are summarised in tableS_ensembles_compare
            continue
        if f.stem.replace("compare_", "").split("_hapgam_e")[0] not in REPORTED_SCEN:
            continue
        df = pd.read_csv(f, sep="\t")
        if df.empty:
            continue
        multi = df["folds"].max() > 1
        nb = "nb_p" in df.columns
        header = ["Trait", "*r* focal", "*r* other model", "Difference [95% CI]", "*P* (Holm)"]
        header += ["*P* (corrected *t*, Holm)"] if nb else []
        header += ["Folds better"] if multi else []
        rows = []
        for base in [m for m in REPORTED if m in set(df["baseline"])]:
            rows.append(group_row(f"vs {MODELS[base]}", len(header)))
            sub = df[df["baseline"] == base].set_index("trait")
            for t in [t for t in TRAITS if t in sub.index]:
                r = sub.loc[t]
                row = [TRAITS[t], fmt(r["r_focal"]), fmt(r["r_base"]),
                       f"{sgn(r['diff'])}\u00a0[{sgn(r['ci_lo'])},\u00a0{sgn(r['ci_hi'])}]", fmt_p(r["p_holm"])]
                row += [fmt_p(r["nb_p_holm"] if "nb_p_holm" in r else r["nb_p"])] if nb else []
                row += [f"{int(r['folds_positive'])}/{int(r['folds'])}"] if multi else []
                rows.append(row)
        name = f.stem.replace("compare_", "tableS_compare_")
        focal = "HapGAM + LightGBM" if "_ens_" in f.stem else "HapGAM"
        header = [h.replace("focal", focal) for h in header]
        (OUT / f"{name}.md").write_text(md_table(header, rows))


def table_s_extra(long):
    """Hyper-parameter choices, Mendelian traits, enrichment (C2) and ensembles."""
    names = {"dev_region": "Training-set cross-validation", "C1": "C1", "C2": "C2", "C3": "C3",
             "grin2soynam": "C4", "grin2nust": "C5", "pc_random": "Mendelian traits"}
    f = RESULTS / "tables" / "hapgam_choices_hapgam_e_summary.tsv"
    if f.exists():
        c = pd.read_csv(f, sep="\t", index_col=0)
        order = [k for k in names if k in c.index]
        rows = [[names[k], str(int(c.at[k, "folds"])), str(int(c.at[k, "kernel_model_M0"])),
                 " / ".join(str(int(c.at[k, m])) for m in ["M8", "M16", "M32", "M64"]),
                 " / ".join(str(int(c.at[k, b])) for b in ["blk8", "blk16", "blk32"]),
                 f"{c.at[k, 'w_epi_mean']:.2f}", f"{c.at[k, 'inner_gain_mean']:+.3f}"] for k in order]
        (OUT / "tableS_choices.md").write_text(md_table(
            ["Scenario", "Fits", "*M* = 0", "*M* = 8 / 16 / 32 / 64", "Block size 8 / 16 / 32", "Mean *w*",
             "Inner gain"], rows))
    # positive controls on the folds run by HapGAM-E: (a) accuracy of the single models, models as rows (BayesB results
    # collected from BGLR carry no AUC, which is computed here from the predictions); (b) enrichment of the selected
    # blocks near the causal gene
    pcs = {"pc_flower_purple": ("Purple flower", "W1"), "pc_pubescence_tawny": ("Tawny pubescence", "T"),
           "pc_indeterminate": ("Indeterminate stem", "Dt1"), "pc_seedcoat_pigmented": ("Pigmented seed coat", "I")}
    enr = RESULTS / "tables" / "blocks_enrichment_pc_random_hapgam_e.tsv"
    if enr.exists():
        e = pd.read_csv(enr, sep="\t")
        pc_models = ["gblup", "egblup", "bayesb", "lightgbm", "hapgam_e_linear", "hapgam_e"]

        def pc_auc(m, t, f_, j):
            if "auc" in j:
                return j["auc"]
            p = pd.read_csv(RUNS / "pc_random" / m / t / f"{f_}.pred.tsv", sep="\t")
            return auc(p["y"].to_numpy(float), p["pred"].to_numpy(float))

        acc = {}
        for t in pcs:
            folds = sorted(p_.stem for p_ in (RUNS / "pc_random" / "hapgam_e" / t).glob("*.json"))
            for m in pc_models:
                js = [json.loads((RUNS / "pc_random" / m / t / f"{f_}.json").read_text()) for f_ in folds]
                acc[(m, t, "r")] = np.mean([j["pearson"] for j in js])
                acc[(m, t, "auc")] = np.mean([pc_auc(m, t, f_, j) for f_, j in zip(folds, js)])
        header = ["Model"] + [f"{lab} (*{gene}*)" for lab, gene in pcs.values()]
        rows = []
        for met, lab in [("r", "Pearson correlation with the 0/1 phenotype"), ("auc", "Area under the ROC curve")]:
            rows.append(group_row(lab, len(header)))
            best = {t: max(acc[(m, t, met)] for m in pc_models) for t in pcs}
            rows += [[MODELS[m]] + [fmt(acc[(m, t, met)], fmt(acc[(m, t, met)]) == fmt(best[t])) for t in pcs]
                     for m in pc_models]
        (OUT / "tableS_positive_controls.md").write_text(md_table(header, rows))
        rows = []
        for t, (lab, gene) in pcs.items():
            ee = e[e["trait"] == t]
            rows.append([lab, f"*{gene}*", f"{100 * ee['share_near_gene'].mean():.1f}%",
                         f"{100 * ee['expected_share'].mean():.2f}%",
                         fmt_p_perm(0.0) if (ee["p_enrich"] == 0).all() else fmt_p_perm(ee["p_enrich"].max())])
        (OUT / "tableS_positive_controls_blocks.md").write_text(md_table(
            ["Trait", "Causal gene", "Selected blocks near a known gene", "Expected for random blocks", "*P*"], rows))
    enr = RESULTS / "tables" / "blocks_enrichment_C2_hapgam_e.tsv"
    if enr.exists():
        e = pd.read_csv(enr, sep="\t")
        rows = [[TRAITS.get(r["trait"], r["trait"]), str(int(r["n_selected"])), f"{100 * r['share_near_gene']:.1f}%",
                 f"{100 * r['expected_share']:.2f}%", fmt_p_perm(r["p_enrich"]),
                 ", ".join(f"*{g}*" if g != "-" else "–" for g in str(r["top5_near_gene"]).split(","))]
                for _, r in e.iterrows()]
        (OUT / "tableS_enrichment_C2.md").write_text(md_table(
            ["Trait", "Blocks", "Near gene", "Expected", "*P*", "Nearest gene of the five most important blocks"],
            rows))
    # ensembles
    rows = []
    singles = SINGLES + ["hapgam_e_linear"]
    for sc in REPORTED_SCEN:
        sub = long[long["scenario"] == sc]
        if sub.empty or "ens_hapgam_e_lightgbm" not in set(sub["model"]):
            continue
        met = "within_r" if sc.startswith("grin2") else "pearson"
        tab = sub.pivot_table(index="trait", columns="model", values=met)
        for t in [t for t in TRAITS if t in tab.index]:
            row = tab.loc[t]
            cand = row[[m for m in singles if m in row.index]].dropna()
            if cand.empty or not np.isfinite(row.get("ens_hapgam_e_lightgbm", np.nan)):
                continue
            if not any(r[0] == f"*{SCEN.get(sc, sc)}*" for r in rows):
                rows.append(group_row(SCEN.get(sc, sc), 7))
            rows.append([TRAITS[t], MODELS[cand.idxmax()], fmt(cand.max()),
                         fmt(row.get("hapgam_e", np.nan)), fmt(row.get("lightgbm", np.nan)),
                         fmt(row.get("ens_hapgam_e_lightgbm", np.nan), row.get("ens_hapgam_e_lightgbm", -1) >
                             max(cand.max(), row.get("ens_gblup_lightgbm_bayesb", -np.inf))),
                         fmt(row.get("ens_gblup_lightgbm_bayesb", np.nan))])
    if rows:
        (OUT / "tableS_ensembles.md").write_text(md_table(
            ["Trait", "Best single model", "*r* best single", "*r* HapGAM", "*r* LightGBM",
             "*r* HapGAM + LightGBM", "*r* GBLUP + LightGBM + BayesB"], rows))


def table_s_ens_compare():
    """S12b: paired comparisons of the HapGAM-E + LightGBM ensemble with HapGAM-E, LightGBM and the baseline
    ensemble in every scenario (difference in r and Holm-adjusted bootstrap P)."""
    bases = ["hapgam_e", "lightgbm", "ens_gblup_lightgbm_bayesb"]
    header = ["Trait"] + [x for b in bases for x in (f"Δ*r* vs {MODELS[b]}", "*P*")]
    rows = []
    for sc in REPORTED_SCEN:
        f = RESULTS / "tables" / (f"compare_{sc}_ens_hapgam_e_lightgbm_within.tsv" if sc.startswith("grin2")
                                  else f"compare_{sc}_ens_hapgam_e_lightgbm.tsv")
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t").set_index(["trait", "baseline"])
        rows.append(group_row(SCEN[sc], len(header)))
        for t in [t for t in TRAITS if (t, "hapgam_e") in d.index]:
            row = [TRAITS[t]]
            for b in bases:
                if (t, b) in d.index:
                    r = d.loc[(t, b)]
                    row += [sgn(r["diff"]), fmt_p(r["p_holm"])]
                else:
                    row += ["–", "–"]
            rows.append(row)
    (OUT / "tableS_ensembles_compare.md").write_text(md_table(header, rows))


def table_s_transfer():
    """S12e: HapGAM-E minus GBLUP / EGBLUP / LightGBM in C6 and C7, all target lines vs excluding SoyNAM families whose
    founder parent and NUST lines that are GRIN accessions; S12f: C7 trials in which HapGAM-E was more accurate."""
    f = RESULTS / "tables" / "transfer_sensitivity.tsv"
    if f.exists():
        d = pd.read_csv(f, sep="\t")
        rows = []
        for sc in ["grin2soynam", "grin2nust"]:
            rows.append(group_row(SCEN[sc], 7))
            sub = d[d["scenario"] == sc]
            for t in [t for t in TRAITS if t in set(sub["trait"])]:
                a = sub[sub["trait"] == t].set_index("model")
                row = [TRAITS[t]]
                for col in ["within_all", "within_excl"]:
                    row += [sgn(a.at["hapgam_e", col] - a.at[b, col]) for b in ["gblup", "egblup", "lightgbm"]]
                rows.append(row)
        (OUT / "tableS_transfer_sensitivity.md").write_text(md_table(
            ["Trait", "vs GBLUP, all", "vs EGBLUP, all", "vs LightGBM, all",
             "vs GBLUP, excluding", "vs EGBLUP, excluding", "vs LightGBM, excluding"], rows))
    f = RESULTS / "tables" / "nust_trials_won.tsv"
    if f.exists():
        w = pd.read_csv(f, sep="\t").pivot(index="trait", columns="baseline", values="trials_won")
        bases = ["gblup", "egblup", "bayesb", "lightgbm"]
        rows = [[TRAITS[t]] + [w.at[t, b] for b in bases] for t in TRAITS if t in w.index]
        (OUT / "tableS_nust_trials.md").write_text(md_table(["Trait"] + [f"vs {MODELS[b]}" for b in bases], rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--from-tsv" in sys.argv:      # re-render the tables from an existing all_metrics.tsv (10 decimals)
        long = pd.read_csv(RESULTS / "tables" / "all_metrics.tsv", sep="\t")
    else:
        long = long_table()
        (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
        long.to_csv(RESULTS / "tables" / "all_metrics.tsv", sep="\t", index=False, float_format="%.10f")
    table2(long)
    table3(long)
    table_s(long)
    table_s_compare()
    table_s_extra(long)
    table_s_ens_compare()
    table_s_transfer()
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
