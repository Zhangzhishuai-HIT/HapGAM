#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%A_%a.log
# Array driver: line N of <task_list> = "<model> <scenario> <trait> [folds...]"
# usage: sbatch -J <name> --array=1-N%K jobs/slurm_array.sh <task_list>
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
TH=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
read -r MODEL SC TRAIT FOLDS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
ARGS="--model $MODEL --scenario $SC --traits $TRAIT --threads $TH"
[ -n "${FOLDS:-}" ] && ARGS="$ARGS --folds $FOLDS"
echo "task $SLURM_ARRAY_TASK_ID: $ARGS on $(hostname)"
$PY scripts/run_model.py $ARGS
