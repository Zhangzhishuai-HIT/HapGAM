#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%A_%a.log
# line N of <list> = "<scenario> <trait>"
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
read -r SC TRAIT <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python scripts/01_data/make_targets.py --scenarios $SC --traits $TRAIT
