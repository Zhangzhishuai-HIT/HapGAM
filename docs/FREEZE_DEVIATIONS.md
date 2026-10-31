# Changes to frozen files after the freeze

The code and configuration used for the confirmatory analyses were frozen on 2026-10-21 at 01:59
(`docs/FREEZE.json`: SHA-256 of 72 files; package code hash `440783eb7e4ba983`, recorded in every result file
as `code`). The frozen files were then set read-only. This file lists every later change to a frozen file.
No change altered a HapGAM model, a split, a target, a metric or a hyper-parameter grid.

## Prediction code

| File | Time | Change | Effect on results |
|---|---|---|---|
| `src/hapgam/api.py` | 2026-10-21 16:47 | `nets()`: for DNNGP, the kernel principal components of each inner fold are computed once and reused for all six dropout × learning-rate combinations, instead of being recomputed for every combination (diff: `freeze_deviations/api_py_dnngp_pc_cache.diff`). | None: the principal components depend only on the inner fold, so the computation is identical. Needed because DNNGP exceeded the time limits of the CPU cluster (protocol record 6). |

Replacing the current `api.py` by the frozen version (reconstructed by reversing the diff; SHA-256
`0a98ea9f92bb5da5a68ac10e627229e1818442323d3ddf53ea27862c2b068b47`, identical to `FREEZE.json`) restores the
frozen package hash `440783eb7e4ba983`, which shows that no other package file changed. Runs started after
16:47 carry the package hash `989587bae25cd9a9`; for every model other than DNNGP they execute exactly the frozen
code path. Result files from the development scenarios (`dev_region`, `dev_random`) that carry other hashes
were produced during development, before the freeze.

## Analysis, plotting and job files (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/compare_models.py` | added the corrected resampled t-test (Nadeau and Bengio 2003) for random k-fold scenarios, as specified in section 7 of the protocol |
| `scripts/04_analysis/summarize.py` | also collects the within-group correlation |
| `scripts/04_analysis/block_maps.py` | positive-control traits mapped to their descriptor genes in `known_genes.tsv` (needed to relate positive-control blocks to *W1*, *T*, *Dt1* and *I*); enrichment test unchanged |
| `scripts/04_analysis/gain_vs_architecture.py` | position of the figure annotation |
| `scripts/04_analysis/fig_blockmap.py`, `fig_haplotype_effects.py`, `fig_results.py`, `fig_schematic.py` | figure layout, labels and panel options |
| `jobs/lists/dev_trees_dnngp.txt` | development random-split LightGBM and DNNGP tasks removed (protocol record 4, addendum) |
| `docs/PREREGISTRATION.md` | dated records appended (records 3–7); `docs/PREREGISTRATION.sha256` updated after each append |

New analysis scripts written after the freeze (`ensembles.py`, `make_tables.py`, `fig_main.py`,
`summary_stats.py`, `hapgam_choices.py`, `additive_r2.py`, `soynam_unseen_haplotypes.py`, `make_si_prereg.py`,
`finalize.sh`, `figstyle.py`, `figures.sh`) only read the stored predictions, fold outputs and frozen splits.

### 2026-10-22 figure and table revision (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/figstyle.py` (new), `figures.sh` (new) | shared figure style (final print size, Helvetica-type font, fixed model colours checked for colour-vision deficiency); one script that redraws every figure |
| `scripts/04_analysis/fig_schematic.py`, `fig_main.py`, `fig_blockmap.py` | figures redrawn; Fig. 3 now shows the C2 and C1 comparisons side by side; Fig. S1 is a heatmap of all models (it previously averaged HapGAM-A linear over two of six folds for maturity group); gene labels of block maps no longer overlap |
| `scripts/04_analysis/fig_haplotype_effects.py` | adds the post hoc check of haplotype effects in the test set: mean partial residual (y − ŷ) − mean(y − ŷ) + f_b of the carriers of each haplotype, and the carrier-weighted correlation r_w with the learned effects (Methods, Block interpretation) |
| `scripts/04_analysis/make_tables.py` | `all_metrics.tsv` written with six instead of four decimals (four decimals caused double rounding in numbers derived from it, e.g. 0.7325 → 0.733 for a value of 0.73246); supplementary tables transposed (models × traits) and grouped; comparison of the HapGAM-E + LightGBM ensemble added as Table S12b; HapGAM-A linear (partial) omitted from the supplementary accuracy tables |
| `scripts/04_analysis/summary_stats.py`, `make_si_prereg.py` | short column labels of the coverage table; formula markup in Supplementary Note 3 |
| `manuscript/build/build.js` | three-line (booktabs) tables, content-based column widths, display equations, one figure per page |

### 2026-10-22 statistical reporting revision (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/compare_models.py` | bootstrap *P* capped at 1 and set to 1 when the two models' predictions are identical to numerical precision (e.g. HapGAM-E with *M* = 0 vs EGBLUP); corrected resampled *t*-test Holm-adjusted across traits (`nb_p_holm`), as specified in section 7 of the protocol |
| `scripts/04_analysis/patch_compare_tables.py` (new) | applies the two corrections above to the existing comparison tables without re-running the bootstrap (changed: one C6 row with identical predictions, *P* 0.16 → 1; Holm values of two C6 rows; `nb_p_holm` added for C5 random) |
| `scripts/04_analysis/transfer_sensitivity.py` (new) | C6/C7 sensitivity analysis excluding SoyNAM families whose founder parent and NUST cultivars that are GRIN accessions, and C7 per-trial wins |
| `scripts/04_analysis/make_tables.py` | RMSE in the lockbox table, development comparisons (Table S6b), transfer sensitivity and C7 trials (Tables S12e, f), signed zeros printed as 0.000, percentages for enrichment shares, number-free table file names |

### 2026-10-22 corrections after the fourth manuscript review (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/fig_schematic.py` (frozen) | Fig. 1a label "M = 0 returns GBLUP or EGBLUP exactly" → "M = 0 recovers GBLUP or EGBLUP" (wording aligned with the text, which states that M = 0 reproduces EGBLUP to numerical precision) |
| `scripts/04_analysis/fig_main.py` | Fig. 2b x-axis ticks at 0.05–0.20 and Fig. S1 colour bar moved left (a tick label and the colour-bar title touched the image edge) |
| `scripts/04_analysis/make_tables.py` | `all_metrics.tsv` written with ten decimals (six decimals still caused double rounding for values within 5 × 10⁻⁷ of a rounding boundary, e.g. 0.5315000543 → 0.531500 → 0.531; four table cells changed by 0.001); option `--from-tsv` re-renders the tables from that file; permutation *P* values ≥ 0.1 printed with two decimals ("1.00") |
| `scripts/04_analysis/make_si_prereg.py` | the removal of signed zeros in Supplementary Note 3 is restricted to "+0.000"/"−0.000" not followed by a further digit (it had turned "+0.0003" into "0.0003") |
| `scripts/04_analysis/finalize.sh` | also runs `patch_compare_tables.py` and `transfer_sensitivity.py` |
| `scripts/05_release/make_release_staged.sh` (new) | builds the same two archives as `make_release.sh` in steps of less than a minute |

The HapGAM-E block screening in the frozen code uses the additive part of the EGBLUP solution, (1 − w)Zᵀα/p, rather
than a separate additive REML-GBLUP fit as worded in protocol amendment 2 (identical for HapGAM-A and when w = 0;
a separate additive fit is used when w = 1). This is not a change to the code; the manuscript (Methods) now describes
the implemented rule and notes the difference from the amendment's wording.

### 2026-10-22 corrections after the fifth manuscript review (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/make_tables.py` | Table S14 split into (a) accuracy and (b) block enrichment; (a) now includes LightGBM, which the positive-control task list (`jobs/lists/pc.txt`, frozen) specified and which had been run but omitted from the table; bold marks every model whose printed value equals the printed maximum (previously: within 5 × 10⁻⁴ of the maximum, which bolded or left plain some cells that print the same value; three cells changed) |
| `scripts/04_analysis/summary_stats.py` | coverage table (Table S3) includes the positive controls; option `--coverage-only` |
| `scripts/04_analysis/fig_main.py` | Fig. S1 uses the same bold rule as the tables (no cell changed) |
| `manuscript/build/build.js`, `build_refs.py` | nested bullet lists; tables of up to 25 rows kept on one page; footer paragraph excluded from line numbering; reference title capitalisation (Northern Uniform Soybean Tests) |
| `scripts/04_analysis/make_tables.py` (round 6) | Table S14b column labels ("Causal gene", "Selected blocks near a known gene"); permutation *P* "< 0.0002" with a non-breaking space |

### 2026-10-22 corrections after the seventh manuscript review (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/hapgam_choices.py` | summary means of *w* and of the inner-CV gain stored unrounded (rounding to 3 and 4 decimals before the table rounded again turned 0.3452 into 0.34 and 0.00147 into +0.002 in Table S7) |
| `scripts/04_analysis/gain_vs_architecture.py` (frozen, not changed) | its docstring says the concentration index uses 16-SNP blocks; the code uses the block scores stored with each fitted model, i.e. the block size chosen by inner CV in that fold, which is what the manuscript describes |
| `docs/DATA_CARD.md`, `docs/README_release.md` | missing-genotype rate after QC 0.02% (not 0.03%, the rate before QC); inner-CV folds described as implemented (regions for GRIN grouped scenarios and transfer, families for SoyNAM leave-families-out); only the change to prediction code has a diff file |

### 2026-10-22 corrections after the eighth manuscript review (no effect on predictions)

| File | Change |
|---|---|
| `scripts/01_data/extract_soynam_cran.R` (new) | writes `data/interim/soynam/data_{line,check}_{qa,raw}.tsv` from the CRAN package SoyNAM 1.6.2; reproduces the files used in the analyses byte for byte (MD5 checked) |
| `scripts/02_baselines/make_dosage_int8.py` (new) | writes `data/processed/grin/dosage.int8(.json)`, the headerless copy of `dosage.npy` read by `bglr_fit.R` (identical to the file used: MD5 of the data bytes checked) |
| `scripts/04_analysis/finalize.sh`, `figures.sh`, `scripts/05_release/make_release*.sh`, `tests/test_reml_vs_rrblup.py` | no absolute paths (repository root from the script location; `PY` and `RSCRIPT` from the environment) |
| `scripts/05_release/make_release*.sh` | the code archive now also contains the six frozen GPU job scripts and `jobs/xiyou/xiyou_runner.py`, so that every frozen file can be checked against `docs/FREEZE.json`; only `jobs/xiyou/sync_from_xiyou.sh` (file transfer to a cloud server) is left out |
| `scripts/04_analysis/make_tables.py` | Table S7 header symbols in italics |
| `docs/README_release.md` | input folder names under `data/raw/`, the machine-specific paths in three frozen scripts, the SoyNAM extraction and `dosage.int8` steps, and the full list of target scenarios |

### 2026-10-22 condensed submission version (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/fig_schematic.py`, `fig_blockmap.py` (frozen) | figure labels "random/Japanese lockbox" → "random/Japanese test set"; bar labels of Fig. 1b broken after the first word |
| `scripts/04_analysis/fig_main.py` | the same label change in Figs. 3 and 4 |
| `scripts/04_analysis/make_tables.py` | scenario labels "test set"; Table 2 file renamed `table2_test_sets.md`; tables not used by the manuscript (C1/C2 accuracy, development random split, ensemble comparison detail) and the `.tsv` copies of the tables are no longer written; Table S4 with trait rows |
| `scripts/04_analysis/make_si_prereg.py` | the translation is a subsection of Supplementary Note 3 (after the status, deviations and post hoc analyses) |

### 2026-10-22 final submission version: single model name, field-standard terms (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/figstyle.py`, `fig_main.py`, `fig_schematic.py` (frozen), `fig_blockmap.py` (frozen) | display labels: HapGAM-E → HapGAM, HapGAM-E linear → HapGAM-linear; the variant with an additive kernel only (internally `hapgam`) is not shown; Fig. 1b without the timeline, titles "Evaluation design" / "Evaluation scenarios", "training set" instead of "development set"; Fig. 5a title "Mendelian traits" |
| `scripts/04_analysis/make_tables.py`, `summary_stats.py` | the same labels; rows and comparisons of the additive-kernel variant omitted; scenario label "Training set, leave-one-region-out"; "Mendelian traits"; "*r* other model"; headers of the table on random initialisations |

### 2026-10-22 corrections after the tenth and eleventh manuscript reviews (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/fig_schematic.py` (frozen) | Fig. 1a label "Block screening (training data only)" instead of "(training set only)": screening uses the training data of each analysis |
| `scripts/04_analysis/make_tables.py` | Table S2 column "Reported value" |

### 2026-10-23 copy-editing of the final version (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/fig_main.py` | Fig. 4 panel title "HapGAM − best kernel method" (the manuscript defines "the kernel model" as HapGAM without block networks) |

### 2026-10-25 narrower reported scope (no effect on predictions)

The article reports five single models (HapGAM, GBLUP, EGBLUP, BayesB, LightGBM), HapGAM-linear and the two ensembles, and the scenarios dev_region, C1, C2, C3, grin2soynam and grin2nust (article C4, C5). All runs stay in `runs/` and `results/tables/all_metrics.tsv`.

| File | Change |
|---|---|
| `scripts/04_analysis/fig_schematic.py` (frozen) | Fig. 1b lists the five reported scenarios (C1–C5) with wider row spacing |
| `scripts/04_analysis/fig_blockmap.py`, `fig_haplotype_effects.py` (frozen) | output names and presets of the supplementary figures renumbered: `figS2_blockmap_dev_region` (preset `figS2`), `figS3_haplotypes_C2_extra` (preset `figS3`); content unchanged |
| `scripts/04_analysis/fig_main.py` | Fig. 4 with the five reported scenarios and models (left panel: better of GBLUP and EGBLUP); Fig. S1 shows C3 only (`figS1_C3_regions`); the all-model heatmap of the training-set cross-validation is no longer drawn |
| `scripts/04_analysis/make_tables.py`, `summary_stats.py`, `figures.sh` | manuscript tables restricted to the reported models and scenarios; secondary-metric, initialisation and coverage tables no longer written to the manuscript |

### 2026-10-29 to 2026-10-31 completed runs, labels and figure font (frozen code unchanged; existing predictions unchanged)

The reported models were run, with the frozen code, in the scenario × trait combinations still missing; comparison tables gained the new rows (fixed bootstrap seed).

| File | Change |
|---|---|
| `jobs/lists/pc_complete.tsv` (new) | Mendelian traits (`pc_random`) in all five folds |
| `jobs/lists/linear_complete.tsv`, `jobs/lists/bglr_complete.txt` (new) | HapGAM-linear in C3, grin2soynam, grin2nust and `pc_random` (98 fits); BayesB in C3 for the five traits not yet run and in `pc_random` (55 fits) |
| `scripts/04_analysis/finalize.sh` | also summarises `pc_random` |
| `scripts/04_analysis/make_tables.py` | Mendelian-trait table with BayesB and HapGAM-linear (AUC computed from the predictions for the BGLR results); HapGAM-linear counted as a single model in the ensemble table; scenario label "Training-set cross-validation"; Table S4 column "Fits" |
| `scripts/04_analysis/fig_main.py` | HapGAM-linear among the single models of Fig. 4 (right panel); Fig. 3 panel title "C2: Japanese test set (2,931 accessions)" |
| `scripts/04_analysis/figstyle.py` | figure text in Times New Roman (author request), with STIX and DejaVu Serif for glyphs that Times New Roman lacks (∝); all figures redrawn |
| `jobs/slurm_build_render.sh` | also renders the Chinese reading copy |

### 2026-10-31 public code repository (no effect on predictions)

| File | Change |
|---|---|
| `scripts/04_analysis/check_dois.py` (frozen) | the contact address sent to Crossref is read from the environment variable `CROSSREF_MAILTO` instead of being written in the script |
| `scripts/05_release/push_github.sh` (new) | copies a code archive into the clone of the GitHub repository, commits and pushes |
| `scripts/05_release/make_release.sh`, `make_release_staged.sh` | the code archive (the content of the GitHub repository) no longer contains input data (data splits, known-gene tables) and now contains `LICENSE` (MIT) |
| `docs/README_release.md`, `docs/DATA_CARD.md` | the repository holds code only; the data splits are regenerated byte for byte by `make_grin_splits.py` and `hapgam.scenarios.soynam_splits()` (checked on 2026-10-31; SHA-256 in the README); format of `known_genes.tsv`; licence; `push_github.sh` |
