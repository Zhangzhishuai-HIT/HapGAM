#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Redraw manuscript figures on a compute node.
# usage: sbatch -J figs jobs/slurm_figures.sh "fig_main.py fig2" "fig_main.py fig3" ...   (no argument: all figures)
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
if [ $# -eq 0 ]; then
  bash scripts/04_analysis/figures.sh
else
  for cmd in "$@"; do echo "== $cmd"; $PY scripts/04_analysis/$cmd 2>&1 | grep -v timestamp | tail -4; done
fi
echo "figures done $(date +%T)"
