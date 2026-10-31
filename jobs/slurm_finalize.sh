#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Regenerate all derived results (scripts/04_analysis/finalize.sh) on a compute node instead of the login node.
# usage: sbatch -J finalize jobs/slurm_finalize.sh
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
bash scripts/04_analysis/finalize.sh
