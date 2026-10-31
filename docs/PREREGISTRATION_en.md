# HapGAM pre-registration (English translation)

Translation of `docs/PREREGISTRATION.md` (Chinese original; SHA-256 in `docs/PREREGISTRATION.sha256`). The original is authoritative. The protocol was written on 2026-10-20 before any model was trained; later changes were only appended as dated amendments or records.

## 1. Hypotheses

- **H1 (primary).** For traits with a few large-effect loci on a polygenic background, the accuracy of HapGAM in unseen-region (grouped holdout) scenarios is not lower than max(GBLUP, LightGBM), and higher for some traits.
- **H2.** HapGAM outperforms GBLUP in cross-population zero-shot transfer (GRIN → SoyNAM, GRIN → NUST).
- **H3.** Under random splits HapGAM is on a par with the strongest baseline (95% CI of the difference covers 0 or is positive); for polygenic traits such as yield it is on a par with GBLUP.
- **H4 (interpretability).** The gated non-linear blocks are enriched near known large-effect genes (E1–E4, Dt1, Dt2, W1, T, …), and the blocks selected for positive-control traits coincide with the known gene positions.

## 2. Data and splits (frozen: `splits/grin_splits.tsv`, SHA-256 in the same folder)

- GRIN × SoySNP50K (Wm82.a2): 17,407 *G. max* accessions, 38,016 SNPs (phenotype-free QC: call rate ≥ 0.95, MAF ≥ 0.01, mode imputation).
- `lockbox_japan` (2,931 accessions from Japan): unseen-region lockbox, not used during development.
- `lockbox_random` (2,175 accessions, 15% of non-Japanese accessions stratified by region × maturity group): random lockbox, not used during development.
- `dev` (12,301 accessions): the only material used during development (random five-fold; leave-one-region-out over China, Korea, South/Southeast Asia, Americas, Europe, Russia/Central Asia).
- SoyNAM (5,176 RILs, 40 families) and NUST (about 2,500 lines): results not inspected during development; run once after the freeze.

## 3. Traits and targets

- Primary traits (4): maturity date (day of year), flowering date (day of year), plant height (cm), 100-seed weight (g).
- Secondary traits: seed yield, protein, oil, lodging, maturity group (MG).
- Targets: centred within evaluation batches (GRIN `method_name`), with batch means estimated from training records only; multiple records per accession averaged. Maturity group is already standardised and not centred. (Replaced by amendment 1.)
- Positive controls (categorical): flower colour (W1), pubescence colour (T), stem termination (Dt1/Dt2), pod colour (L1/L2), seed-coat colour (I).

## 4. Metrics

- **Primary metric:** Pearson r between predictions and batch-centred observations of test accessions (accession level).
- Secondary metrics: mean within-batch Pearson r (batches with ≥ 20 test records), Spearman correlation, RMSE, top-20% selection coincidence.
- Categorical controls: AUC.

## 5. Scenarios (primary in bold)

| Code | Training → test | Role |
|---|---|---|
| C1 | dev → lockbox_random | random holdout, confirmatory |
| **C2** | dev + lockbox_random → **lockbox_japan** | **unseen region, confirmatory (primary test of H1)** |
| C3 | all material, leave-one-region-out (7 regions) | grouped holdout; regions other than Japan were exposed during development |
| C4 | all material, leave-one-maturity-group-out (5 groups) | extrapolation, secondary |
| C5 | SoyNAM random five-fold and leave-families-out (families grouped into 5 folds) | new population, confirmatory |
| **C6** | **GRIN → SoyNAM zero-shot** (within-family r) | **cross-population, confirmatory (H2)** |
| C7 | GRIN → NUST zero-shot | cross-population, confirmatory |

## 6. Models

HapGAM: ŷ = μ + Xβ + Σ_b γ_b g_b(x_b) (+ optional sparse block interactions) + cᵀPC. The linear part is initialised with REML-rrBLUP and keeps the same ridge penalty; g_b are small MLPs with zero-initialised output layers; γ is shrunk by a group-sparsity penalty. Hyper-parameters (block scale, penalty strength, hidden units, learning rate, number of epochs) are selected only by inner cross-validation within the training set.

Baselines: GBLUP/rrBLUP (REML), BayesB and BayesC (BGLR), RKHS (Gaussian kernel, bandwidth grid), EGBLUP (additive + additive × additive kernel), haplotype GBLUP, LightGBM, XGBoost, MLP, a DNNGP-type CNN, and a stacked ensemble (GBLUP + LightGBM + BayesB). Tuning budget: tree models and HapGAM use the same inner-CV scheme and the same maximum number of search trials; linear mixed models use REML.

## 7. Statistics

- Paired differences (HapGAM − baseline): for C1/C2, 95% CIs from 2,000 bootstrap replicates of test accessions; for C3/C4, summary over held-out groups with per-group differences; for repeated k-fold, the Nadeau–Bengio corrected t-test; Holm correction across traits.
- Deep models: five random seeds, mean ± SD and the seed-averaged ensemble.

## 8. Go/No-Go (development, dev inner CV)

If the mean gain of HapGAM over max(GBLUP, LightGBM) in grouped-holdout r over the four primary traits is < 0.01 and its bootstrap 95% CI includes 0, no further complexity is added, and the study turns to (i) demonstrating complementarity as an ensemble member, (ii) interpretability and transfer as the main contributions, and (iii) reporting the results as they are.

## 9. Lockbox rules

- The lockboxes (lockbox_random, lockbox_japan, SoyNAM, NUST) are each run only once, after code and configuration have been frozen; code hashes are recorded at the freeze.
- No model changes after any lockbox result; re-testing requires new, untouched data.

## Amendments and records

### Amendment 1 (2026-10-20, before any HapGAM training; based only on GBLUP on dev folds and descriptive statistics)
- **Change:** targets changed from within-batch centring to fold-internal mixed-model batch adjustment: y = μ + b_batch + g + e fitted on training records (g with genomic relationship matrix K, batch fixed); estimated batch effects subtracted from training and test records, then averaged per accession. Test records from batches absent from training are excluded from pooled metrics (used only for within-batch r).
- **Reason:** GRIN evaluation batches are organised by maturity group (e.g. 2KY81 contains only MG IV–V, within-batch SD of maturity 2.9 days); centring removed the genetic differences between maturity groups, giving GBLUP r of only 0.44–0.48 for maturity in dev random five-fold CV and heritability ≈ 0.42. Genomic relationships connect accessions across batches, so batch effects absorb only environmental differences.
- **Consequences:** all models share the same cached targets (`data/processed/targets/`); lockboxes untouched. GBLUP results under the original targets are archived in `runs/_superseded/` and not used for any conclusion.

### Amendment 2 (2026-10-20, development; dev inner/outer folds only, lockboxes untouched)
- **Change:** in addition to the GBLUP backbone (K_A), an EGBLUP backbone with covariance (1 − w)K_A + w·K_AA (K_AA the Hadamard square of K_A scaled to mean diagonal), w ∈ {0, 0.25, 0.5, 0.75, 1} chosen by REML likelihood on the training set; w = 0 is the original HapGAM and M = 0 is EGBLUP. Block screening always uses the additive REML-GBLUP marker effects. Both backbones are reported (HapGAM-A, HapGAM-E), with HapGAM-linear (linear within blocks) as ablation.
- **Reason:** in the dev region holdout EGBLUP was clearly better than GBLUP for plant height (0.610 vs 0.567); following the principle that the architecture should contain the strongest baseline as a special case, HapGAM should nest EGBLUP.
- **Consequences:** primary endpoints, metrics and splits unchanged; the primary model (HapGAM-A or HapGAM-E) is fixed before the freeze using dev results only; code hashes recorded at the freeze.

### Decision rules (2026-10-20 22:47, before dev results were complete)
- **Primary model:** compare HapGAM-A and HapGAM-E on dev_region by the mean Pearson r over the four primary traits (mean over folds, then over traits); choose the higher; if the difference is < 0.002 choose HapGAM-E (it nests the strongest kernel baseline, EGBLUP). The other variant is reported as secondary.
- **Confirmatory grid:** if computing resources do not allow the full grid (3 block sizes × M ∈ {8, 16, 32, 64} × 2 weight decays + M = 0) in all confirmatory scenarios, the grid may only be reduced to the configurations actually chosen by the inner CV in the dev_region outer folds (always keeping M = 0), decided on dev results only and before any confirmatory run. The tuning budget of tree models is not reduced (25 trials).
- **Go/No-Go:** as in section 8, against max(GBLUP, LightGBM); the difference to EGBLUP is also reported.

### Record 3 (2026-10-21 04:20, after the freeze; no change to any model or primary endpoint)
- **Go/No-Go (interim):** on dev_region, mean gain of HapGAM-E over max(GBLUP, LightGBM) over the four primary traits +0.0016 (bootstrap 95% CI −0.0022 to +0.0059; 6 of 24 LightGBM folds still missing) → NO-GO under section 8 (final value to be recomputed and reported when LightGBM is complete).
- **Following section 8 (i) and (ii):** the claim of the paper becomes "consistent gains of HapGAM over linear mixed models (GBLUP/EGBLUP/RKHS) + interpretability/mechanism + complementarity with tree models"; no claim of general superiority over LightGBM.
- **Added (pre-specified, equal weights, nothing fitted) ensemble analyses:** HapGAM-E + LightGBM equal-weight average; baseline ensemble GBLUP + LightGBM + BayesB equal-weight average (the stacked ensemble of section 6 with equal weights so that no test information enters the weights). Both are computed from the predictions of the frozen models and reported in all scenarios. At the time of writing, the only confirmatory LightGBM results seen were C2 maturity and flowering date.

### Record 4 (2026-10-21 04:55, after the freeze; only baseline coverage adjusted, no related results seen)
- Because of limited computing resources (CPU queue full, only GPU 2 available), confirmatory baseline coverage was fixed as follows, before the corresponding results were produced:
  - C3/C4: GBLUP, EGBLUP, RKHS, haplotype GBLUP, LightGBM and HapGAM-E for all nine traits; BayesB for the four primary traits only; MLP, DNNGP, HapGAM-A and HapGAM-E linear not run.
  - XGBoost: C1 and C2 only (last in the queue; reported as far as completed); BayesC not run (duplicates BayesB).
  - C3/C4 jobs split by held-out group; no change to any model or configuration.
- Addendum (same day 05:00): LightGBM and DNNGP no longer run on dev_random (development random split, used only to check the target definition; C1 is the confirmatory random split).

### Record 5 (2026-10-21 05:55, after the freeze; documents coverage decisions made when the confirmatory jobs were scheduled, about 02:00–03:30, but not written into this file earlier; none of the omitted analyses had any results)
- C5 SoyNAM random five-fold: only GBLUP, EGBLUP, BayesB, LightGBM and HapGAM-E; leave-families-out additionally RKHS, haplotype GBLUP, DNNGP, HapGAM-A and HapGAM-E linear.
- C6/C7 zero-shot transfer: GBLUP, EGBLUP, RKHS, BayesB, LightGBM and HapGAM-E.
- MLP only in dev_region, C1 and C2.
- Positive controls: flower colour (W1), pubescence colour (T), stem termination (Dt1) and seed-coat colour (I) only; pod colour (L1/L2), listed in section 3, was omitted when the positive-control task was implemented and was not run.
- The model specification evolved during development (dev data only): the group-sparsity gates and PC covariates of section 6 were replaced by REML-BLUP block screening with the number of blocks M (including M = 0) chosen by inner CV; the final specification was fixed at the freeze (`docs/FREEZE.json`).
- Addendum (same day 06:25): the HapGAM-A linear ablation (of the secondary variant) is not run for the remaining development traits; only the ablation of the primary model, HapGAM-E linear, is reported.

### Record 6 (2026-10-21 17:15, after the freeze; no change to any model or primary endpoint; no related results had been produced)
- Section 7 requires "deep models: five random seeds, mean ± SD and the seed-averaged ensemble". MLP and DNNGP follow this rule (five seeds per fold; the primary result is the seed-averaged ensemble; per-seed r stored as seed_r). HapGAM's block networks are also neural networks but, as frozen, were fitted with a single seed (seed 0). In the spirit of section 7 we add, post hoc: HapGAM-E refitted in C2 (all nine traits) with seeds 1–4 (frozen configuration unchanged; result directories hapgam_e_s1 … s4), reporting the mean ± SD of the five single-seed accuracies and the accuracy of the five-seed average prediction. Confirmatory conclusions remain based on the frozen single-seed model (seed 0); this is a supplementary analysis.
- Scheduling only (no change of method): DNNGP exceeded the CPU time limits and was moved to GPU 2; the kernel principal components of each inner fold are now computed once instead of once per hyper-parameter combination (identical results). XGBoost was requeued because its original time limit was too short (now 24 h); method unchanged.
- Addendum (same day 17:45): the DNNGP principal-component caching above changed the frozen file `src/hapgam/api.py` (the only prediction-code file changed after the freeze). The frozen version was reconstructed exactly (SHA-256 identical to FREEZE.json) and the diff is in `docs/freeze_deviations/`; every post-freeze change to any file (including analysis and plotting scripts) is listed in `docs/FREEZE_DEVIATIONS.md`.

### Record 7 (2026-10-21 19:20, after the freeze and after the lockboxes were opened; no change to any model, analysis or primary endpoint)
- Final Go/No-Go (dev_region, all 24 LightGBM folds of the primary traits complete): mean gain of HapGAM-E over max(GBLUP, LightGBM) +0.0036 (bootstrap 95% CI +0.0003 to +0.0073; maturity date +0.018, flowering date +0.000, plant height −0.003, 100-seed weight −0.000). The gain is below 0.01 but the CI excludes 0, so the NO-GO condition of section 8 (gain < 0.01 and CI including 0) is no longer met; the formal decision is GO.
- The interim decision (record 3; 18 of 24 folds, +0.0016, CI −0.0022 to +0.0059) was NO-GO. Both computations were made after the code freeze (01:59), so the Go/No-Go did not influence any model or confirmatory analysis; the equal-weight ensembles added after the interim decision are reported as planned. The paper reports both values: the gain over max(GBLUP, LightGBM) is statistically greater than zero but far below the pre-specified practical margin of 0.01.
