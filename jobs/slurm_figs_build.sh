#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Redraw the listed figures (scripts relative to scripts/04_analysis), then rebuild and render the .docx files.
# usage: sbatch -J figbuild jobs/slurm_figs_build.sh "fig_blockmap.py --preset fig5" ...
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
for cmd in "$@"; do echo "== $cmd"; $PY scripts/04_analysis/$cmd 2>&1 | grep -v timestamp | tail -2; done
bash jobs/slurm_build_render.sh
