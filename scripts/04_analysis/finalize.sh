#!/bin/bash
# Regenerate every derived result used by the manuscript from runs/ (idempotent; safe to re-run).
# usage: bash scripts/04_analysis/finalize.sh
set -uo pipefail
cd "$(dirname "$0")/../.."                      # repository root
PY=${PY:-python}                                # python of the hapgam environment
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
A=scripts/04_analysis
echo "== BayesB collection"
for sc in dev_region C1 C2 C3 C4 soynam_random soynam_lofo grin2soynam grin2nust pc_random; do
  $PY scripts/02_baselines/bglr_prepare.py collect --scenario $sc 2>/dev/null | grep -vc missing | sed "s/^/  $sc: /"
done
echo "== ensembles"
$PY $A/ensembles.py --scenarios dev_region C1 C2 C3 C4 soynam_random soynam_lofo grin2soynam grin2nust
echo "== bootstrap comparisons"
B="gblup egblup rkhs hapgblup bayesb lightgbm xgboost mlp dnngp hapgam hapgam_e_linear ens_gblup_lightgbm_bayesb"
for sc in dev_region C1 C2 C3 C4 soynam_random soynam_lofo; do
  $PY $A/compare_models.py --scenario $sc --focal hapgam_e --baselines $B > /dev/null 2>&1 && echo "  $sc done"
done
for sc in grin2soynam grin2nust; do
  $PY $A/compare_models.py --scenario $sc --focal hapgam_e --baselines $B --within > /dev/null 2>&1 && echo "  $sc (within) done"
done
for sc in dev_region C1 C2 C3 C4 soynam_random soynam_lofo; do
  $PY $A/compare_models.py --scenario $sc --focal ens_hapgam_e_lightgbm --baselines hapgam_e lightgbm ens_gblup_lightgbm_bayesb > /dev/null 2>&1
done
for sc in grin2soynam grin2nust; do
  $PY $A/compare_models.py --scenario $sc --focal ens_hapgam_e_lightgbm --baselines hapgam_e lightgbm ens_gblup_lightgbm_bayesb --within > /dev/null 2>&1
done
$PY $A/patch_compare_tables.py > /dev/null   # idempotent; a no-op for tables written by the current compare_models.py
echo "== Go/No-Go"
$PY $A/go_nogo.py --focal hapgam_e 2>&1 | tail -8
echo "== choices, block maps, architecture"
$PY $A/hapgam_choices.py --model hapgam_e > /dev/null
Q="maturity_doy flowering_doy height_cm seed_weight_g maturity_group yield_mg_ha protein_pct oil_pct lodging_score"
$PY $A/block_maps.py --scenario pc_random --model hapgam_e --traits pc_flower_purple pc_pubescence_tawny pc_indeterminate pc_seedcoat_pigmented --perm 5000 > /dev/null 2>&1
$PY $A/block_maps.py --scenario dev_region --model hapgam_e --traits $Q --perm 5000 > /dev/null 2>&1
$PY $A/block_maps.py --scenario C2 --model hapgam_e --traits $Q --perm 5000 > /dev/null 2>&1
$PY $A/block_maps.py --scenario C3 --model hapgam_e --traits maturity_doy flowering_doy height_cm seed_weight_g --perm 5000 > /dev/null 2>&1
$PY $A/additive_r2.py --scenario C2 --model hapgam_e --fold lockbox_japan 2>&1 | head -1
$PY $A/soynam_unseen_haplotypes.py 2>&1 | tail -1
$PY $A/gain_vs_architecture.py --scenario dev_region --focal hapgam_e --base egblup 2>&1 | grep Spearman
$PY $A/transfer_sensitivity.py > /dev/null 2>&1 && echo "  transfer sensitivity done"
echo "== tables and summaries"
$PY $A/make_tables.py 2>&1 | tail -1
$PY $A/summary_stats.py > /dev/null 2>&1 && echo "  summary_stats done"
echo "== figures"
bash $A/figures.sh
echo "finalize done $(date +%T)"
