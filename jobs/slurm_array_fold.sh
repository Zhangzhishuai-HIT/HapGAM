#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%A_%a.log
# Array driver, one fold per task: line N of <task_list> = "<model>\t<scenario>\t<trait>\t<fold>[\t<extra args>]"
# (tab-separated so that fold names with spaces or slashes, e.g. "S/SE Asia", stay one argument).
# usage: sbatch -J <name> --array=1-N%K jobs/slurm_array_fold.sh <task_list>
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
TH=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
IFS=$'\t' read -r MODEL SC TRAIT FOLD EXTRA <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "task $SLURM_ARRAY_TASK_ID: $MODEL $SC $TRAIT [$FOLD] ${EXTRA:-} on $(hostname)"
$PY scripts/run_model.py --model "$MODEL" --scenario "$SC" --traits "$TRAIT" --folds "$FOLD" --threads "$TH" ${EXTRA:-}
