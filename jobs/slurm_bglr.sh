#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%A_%a.log
# usage: sbatch -J bglr_dev_region --array=1-N jobs/slurm_bglr.sh <list_file> <model> <nIter> <burnIn>
# <list_file> holds one fold-input directory per line.
set -euo pipefail
LIST=$1; MODEL=$2; NITER=${3:-12000}; BURN=${4:-2000}
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -f "$DIR/$MODEL.pred.txt" ] && { echo "exists $DIR"; exit 0; }
export OMP_NUM_THREADS=1
/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam_r/bin/Rscript \
  /home/user/zhangzhishuai/myhermes/2080/HapGAM/scripts/02_baselines/bglr_fit.R "$DIR" "$MODEL" "$NITER" "$BURN"
