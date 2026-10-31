#!/bin/bash
# Redraw every manuscript figure from runs/ and results/tables (figures/*.pdf, vector, and *.png, 600 dpi).
# usage: bash scripts/04_analysis/figures.sh   (on a compute node: sbatch -J figs jobs/slurm_figures.sh)
set -uo pipefail
cd "$(dirname "$0")/../.."                      # repository root
PY=${PY:-python}                                # python of the hapgam environment
A=scripts/04_analysis
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=""
$PY $A/fig_schematic.py 2>&1 | grep png
for f in fig2 fig3 fig4 figS1; do $PY $A/fig_main.py $f 2>&1 | grep png; done
$PY $A/fig_blockmap.py --preset fig5 2>&1 | grep png
$PY $A/fig_blockmap.py --preset figS2 2>&1 | grep png
$PY $A/fig_haplotype_effects.py --preset fig6 2>&1 | grep png
$PY $A/fig_haplotype_effects.py --preset figS3 2>&1 | grep png
