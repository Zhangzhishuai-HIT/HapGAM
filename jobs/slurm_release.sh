#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Build the two release archives and their SHA-256 sums (scripts/05_release/make_release.sh) on a compute node.
# usage: sbatch -J release jobs/slurm_release.sh
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
bash scripts/05_release/make_release.sh
