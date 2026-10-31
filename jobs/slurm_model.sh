#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Generic CPU job: any model x scenario via scripts/run_model.py in the hapgam mamba env.
# usage: sbatch -J <name> jobs/slurm_model.sh <model> <scenario> "<traits>" ["<folds>"]
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
MODEL=$1; SC=$2; TRAITS=$3; FOLDS=${4:-}
ARGS="--model $MODEL --scenario $SC --traits $TRAITS --threads 16"
[ -n "$FOLDS" ] && ARGS="$ARGS --folds $FOLDS"
$PY scripts/run_model.py $ARGS
