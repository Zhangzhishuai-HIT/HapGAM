# HapGAM: haplotype-block effects added to genomic BLUP

Code for the article "Haplotype-block effects added to genomic BLUP improve prediction of major-gene traits
across soybean germplasm origins". Input data and results are not included; all inputs are public (see Data).

## Model

HapGAM fits

    y = μ + g + Σ_{b ∈ S} f_b(z_b) + e,    g ~ N(0, σ²_g K),  e ~ N(0, σ²_e I)

with the EGBLUP covariance K = (1 − w) K_A + w K_AA (w ∈ {0, 0.25, 0.5, 0.75, 1} chosen by REML; w = 0 gives
GBLUP), where z_b are the standardised SNP dosages of haplotype block b, and each f_b is a one-hidden-layer network
(8 GELU units, zero-initialised output weights). The genomic effect is profiled out analytically, so the block
networks are fitted with the generalised-least-squares loss Σ_i (Uᵀ(y − μ − F))_i² / (s_i/δ + 1). Blocks are
screened by the variance of their REML-BLUP contribution; the number of blocks M ∈ {0, 8, 16, 32, 64}
(M = 0 is the kernel model), the block size (8, 16 or 32 SNPs) and the weight decay are chosen by inner
cross-validation (random folds for random splits; folds of whole regions, or of SoyNAM families, for grouped splits
and across-population prediction). Implementation: `src/hapgam/model.py`, `src/hapgam/api.py`.

## Names used in the code

The code, the run directories and the project records in `docs/` use internal names:

| Internal name | In the article |
|---|---|
| `hapgam_e`, `hapgam_e_linear` | HapGAM, HapGAM-linear |
| `hapgam` | HapGAM with the additive kernel only (*w* = 0); not reported |
| `ens_hapgam_e_lightgbm`, `ens_gblup_lightgbm_bayesb` | HapGAM + LightGBM and GBLUP + LightGBM + BayesB ensembles |
| `dev`, `lockbox_random`, `lockbox_japan` (column `set` of `splits/grin_splits.tsv`) | training set, random test set, Japanese test set |
| `dev_random`, `dev_region` | random and leave-one-region-out cross-validation within the training set |
| `C1`, `C2`, `C3` | C1–C3 |
| `grin2soynam`, `grin2nust` | C4 and C5 (across-population prediction) |
| `pc_random` | Mendelian traits |
| `C4`, `soynam_random`, `soynam_lofo` | cross-validation over maturity groups and within SoyNAM; evaluated, not reported in the article |
| `rkhs`, `hapgblup`, `xgboost`, `mlp`, `dnngp` | RKHS regression, haplotype GBLUP, XGBoost, fully connected network, DNNGP; evaluated, not reported in the article |

## Layout

| Path | Content |
|---|---|
| `src/hapgam/` | Python package: data access, REML, kernels, HapGAM model, other models, metrics |
| `scripts/01_data/` | data preparation from the public sources (genotypes, GRIN descriptors, SoyNAM, NUST, targets) |
| `scripts/run_model.py` | generic runner: any model × any scenario × trait × fold |
| `scripts/02_baselines/` | BGLR (BayesB) input preparation, R script and collection; older per-model runners |
| `scripts/04_analysis/` | comparisons, ensembles, enrichment, sensitivity analyses, tables and figures (`finalize.sh` runs everything; `figures.sh` redraws the figures with the shared style in `figstyle.py`) |
| `scripts/05_release/` | builds the code archive (the content of this repository) and a results archive (`make_release.sh`; `make_release_staged.sh` does the same in short steps); `push_github.sh` commits the code archive to this repository |
| `jobs/` | SLURM array scripts, task lists and the task runner used for the reported runs (`jobs/xiyou/xiyou_runner.py` ran 121 tasks, RKHS and haplotype GBLUP in C3 and C4 and one LightGBM task, on a CPU server without a scheduler; a script that only copied files between the cluster and that server is not included) |
| `envs/` | conda environments (`environment_hapgam.yml`: Python 3.11, PyTorch 2.5.1, LightGBM 4.7, XGBoost 3.0, Optuna 5.0; `environment_hapgam_r.yml`: R 4.3, BGLR 1.1.4) |
| `docs/` | `DATA_CARD.md` (data versions and record counts), `raw_data_manifests/` (downloaded files with URL, date and SHA-256) and project records: the analysis plan (`PREREGISTRATION.md`, in Chinese; English translation `PREREGISTRATION_en.md`; SHA-256 in `PREREGISTRATION.sha256`), `FREEZE.json` (SHA-256 of the code, configuration and splits fixed before the test sets were evaluated), `FREEZE_DEVIATIONS.md` and `freeze_deviations/` (later changes to these files) |
| `tests/` | unit tests (REML vs rrBLUP, exact nesting of EGBLUP, synthetic recovery) |

## Installation

    mamba env create -f envs/environment_hapgam.yml      # Python models
    mamba env create -f envs/environment_hapgam_r.yml    # BayesB (BGLR)

A GPU is optional: all models also run on CPU (`CUDA_VISIBLE_DEVICES=""`).

## Data

All inputs are public (versions and record counts in `docs/DATA_CARD.md`). Place them, or symbolic links to them,
under `data/raw/` with the names that the scripts expect:

| `data/raw/…` | Content |
|---|---|
| `soysnp50k_gnm2/` | LIS `Wm82.gnm2.div.Song_Hyten_2015` (SoySNP50K genotypes, `glyma.Wm82.gnm2.div.Song_Hyten_2015.vcf.gz`) |
| `grin_descriptors/` | GRIN-Global soybean descriptor tables (https://npgsweb.ars-grin.gov; downloaded by `download_grin_descriptors.py`) |
| `soynam_cran/` | the extracted CRAN package SoyNAM 1.6.2 (the directory that contains `data/soybase.RData`) |
| `nust/` | a copy of https://github.com/UMN-Lorenz-Group/GenomicAnalysisNUST |

The block maps and enrichment tests read the known genes of Supplementary Table S2 from
`data/processed/known_genes.tsv` (not included; tab-separated, columns `gene_id`, `locus`, `product`, `traits` with
comma-separated trait keys, and `chrom`, `start`, `end` from the Wm82.a2 gene models, LIS `Wm82.gnm2.ann1.RVB6`).
`docs/raw_data_manifests/` lists every downloaded file
with its source URL, download date and SHA-256. The SoyNAM genotypes (LIS `Wm82.gnm2.div.Song_NAM_2021b`) are read from the path set in
`scripts/01_data/build_external_datasets.py` (line 89). That script, `scripts/01_data/download_grin_descriptors.py`
(output folder, line 23) and `scripts/02_baselines/bglr_fit.R` (line 21) contain absolute paths of the original
machine; they were left unchanged so that the code matches `docs/FREEZE.json`, so edit these lines before use. The job scripts in `jobs/` (SLURM scripts, GPU launchers and `jobs/xiyou/xiyou_runner.py`) and the `prefix:`
lines of `envs/*.yml` also contain paths of the original machines. Then run

    python scripts/01_data/parse_soysnp50k.py
    python scripts/01_data/download_grin_descriptors.py 51009 51010 51011 51015 51016 51019 51020 51050 51051 51055 \
        51057 51058 51061 51062 51066 51067 51166   # descriptor IDs; then link data/raw/grin_descriptors to the output
    python scripts/01_data/build_grin_dataset.py
    python scripts/01_data/make_grin_splits.py   # data split of the GRIN accessions
    python scripts/01_data/build_external_datasets.py nust
    python scripts/01_data/build_external_datasets.py soynam
    Rscript scripts/01_data/extract_soynam_cran.R data/raw/soynam_cran data/interim/soynam   # SoyNAM phenotype tables
    python scripts/01_data/make_targets.py --scenarios dev_random dev_region C1 C2 C3 C4 grin_all
    python scripts/02_baselines/make_dosage_int8.py   # genotype file read by bglr_fit.R (BayesB on GRIN folds)

These steps regenerate, byte for byte, the data splits used in the article: `make_grin_splits.py` writes
`splits/grin_splits.tsv` (SHA-256 `d97c6a715c078bc38a3b640cd1b373f473c436bf4186bc67f5636adc9e7cea8d`; it does not overwrite an existing
file), and `hapgam.scenarios.soynam_splits()` writes `splits/soynam_splits.tsv` on first use (SHA-256
`e66974e4b02a01edff435164f03edced2f2d8432e23a799b8cdff1ab7baf99e8`).

## Reproducing the analyses

Scenarios and models are named as in the table above; the other models are `gblup`, `egblup`, `rkhs`, `hapgblup`,
`lightgbm`, `xgboost`, `mlp` and `dnngp` (BayesB runs through BGLR, below).

    python scripts/run_model.py --model hapgam_e --scenario C2 --traits height_cm --threads 8
    python scripts/run_model.py --model lightgbm --scenario C3 --traits maturity_doy --folds Korea

BayesB:

    python scripts/02_baselines/bglr_prepare.py prepare --scenario C2
    Rscript scripts/02_baselines/bglr_fit.R work/bglr/C2/height_cm/lockbox_japan BayesB 12000 2000
    python scripts/02_baselines/bglr_prepare.py collect --scenario C2 --model BayesB

Each run writes `runs/<scenario>/<model>/<trait>/<fold>.json` (metrics, chosen hyper-parameters, code hash)
and `<fold>.pred.tsv` (test predictions); HapGAM also writes the selected blocks (`.blocks.tsv`), the inner-CV
grid (`.tuning.tsv`) and the per-block test contributions (`.parts.npz`). `bash scripts/04_analysis/finalize.sh`
regenerates the ensembles, bootstrap comparisons, enrichment tests, the sensitivity analysis of across-population
prediction, tables and figures from `runs/`; `python scripts/04_analysis/patch_compare_tables.py` applies two later
corrections of `compare_models.py` to existing comparison tables (bootstrap *P* set to 1 when two models' predictions
are identical; Holm adjustment of the corrected resampled *t*-test).

## Citation and licence

Please cite the article (reference to be added on publication). Licence: MIT (`LICENSE`).
