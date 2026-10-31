# Data card (HapGAM)

All raw inputs are public and were downloaded by direct connection into
`/home/user/zhangzhishuai/myhermes/down_data/<source>/` with `MANIFEST.tsv` entries (file, URL, date, SHA-256).
HapGAM reads them only through read-only symbolic links in `data/raw/`.

## Sources

| Data | Content used | Source (accessed 2026-10) | Local folder |
|---|---|---|---|
| SoySNP50K, USDA Soybean Germplasm Collection | 20,087 accessions × 41,439 SNPs (Gm01–Gm20), Wm82.a2 | LIS `Wm82.gnm2.div.Song_Hyten_2015` (Song et al. 2015) | `LIS_glycine_max_diversity/` |
| GRIN-Global descriptors (crop 51) | maturity date, flowering date, height, 100-seed weight, yield, oil, protein, lodging, maturity group; flower, pubescence, stem termination, seed-coat colour | npgsweb.ars-grin.gov descriptor downloads | `GRIN_soybean_descriptors/` |
| SoyNAM phenotypes | QA data (height, R8, lodging, yield, protein, oil, seed size) and raw flowering dates | CRAN package SoyNAM 1.6.2 (Diers et al. 2018) | `SoyNAM_CRAN/` |
| SoyNAM genotypes | 5,176 RILs, 423,419 imputed SNPs, Wm82.a2 | LIS `Wm82.gnm2.div.Song_NAM_2021b` (Chen et al. 2022) | `LIS_glycine_max_diversity/` |
| NUST | BLUEs (1993–2020) and genotypes (5,158 SNPs, Wm82.a2) | GitHub UMN-Lorenz-Group/GenomicAnalysisNUST (Wartha et al. 2025) | `GitHub_UMN-Lorenz-Group_GenomicAnalysisNUST/` |
| Wm82.a2 gene models | gene coordinates for known loci | LIS `Wm82.gnm2.ann1.RVB6` | `LIS_genomes_annotations/` |

## GRIN analysis set (Glycine max with SoySNP50K genotypes)

- 17,407 accessions; 38,016 SNPs after phenotype-free QC (call rate ≥ 0.95, MAF ≥ 0.01; 0.02% missing set to the major homozygote; heterozygosity 0.02%).
- Regions of origin: China 6,204; Korea 3,632; Japan 2,931; South/Southeast Asia 1,408; Americas 1,108; Europe 1,022; Russia/Central Asia 745; other 243; unknown 114.
- Maturity groups: 000–0 1,759; I–II 3,505; III–IV 5,761; V–VI 4,046; VII–X 1,805; missing 531.

| Trait | Records | Accessions | Evaluation batches | Mean | SD |
|---|---:|---:|---:|---:|---:|
| Maturity date (day of year) | 17,548 | 16,968 | 32 | 270.9 | 16.9 |
| Flowering date (day of year) | 17,535 | 16,976 | 32 | 197.5 | 15.9 |
| Plant height (cm) | 17,469 | 16,889 | 32 | 86.7 | 29.4 |
| 100-seed weight (g) | 17,564 | 16,985 | 32 | 15.4 | 5.2 |
| Seed yield (Mg/ha) | 17,412 | 16,864 | 31 | 2.00 | 0.76 |
| Protein (%) | 22,134 | 16,519 | 30 | 44.3 | 2.8 |
| Oil (%) | 22,132 | 16,517 | 30 | 18.3 | 2.1 |
| Lodging score (1–5) | 17,448 | 16,875 | 31 | 2.87 | 0.95 |
| Maturity group (000 = −2 … X = 10) | 17,439 | 16,876 | 32 | 3.55 | 2.29 |

Cleaning: 17 height and 6 yield records outside plausible ranges and 131 records > 6 robust SD from their batch median were removed; batches with < 20 records were dropped; duplicate accession × batch records were averaged.

Data split (`splits/grin_splits.tsv`, written by `scripts/01_data/make_grin_splits.py`; column `set`): training set 12,301 (`dev`); random test set 2,175 (`lockbox_random`); Japanese test set 2,931 (`lockbox_japan`).

## SoyNAM

5,173 RILs from 39 families with genotypes and phenotypes (`splits/soynam_splits.tsv`). 21,309 SNPs matched the GRIN SNP set by position and allele (4.4% missing, mean-imputed within the training data).

| Trait | Records | RILs | Environments |
|---|---:|---:|---:|
| Days to maturity (R8) | 49,313 | 5,171 | 15 |
| Flowering date | 12,296 | 4,770 | 6 |
| Plant height | 49,348 | 5,171 | 15 |
| Lodging | 49,011 | 5,171 | 15 |
| Seed yield | 56,917 | 5,171 | 18 |
| Protein, oil | 44,950 | 5,171 | 10 |
| Seed size (100-seed weight) | 44,949 | 5,171 | 10 |

## NUST

2,637 line × maturity-group-trial BLUEs for 2,511 genotyped lines (MG 00: 63, 0: 360, I: 490, II: 699, III: 661, IV: 364); seed yield 2,633 BLUEs (2,507 lines), protein and oil 2,605 (2,482 lines). 5,061 of 5,158 SNPs matched the GRIN SNP set (0.11% missing).
