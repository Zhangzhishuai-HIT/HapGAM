#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/HapGAM/logs/slurm/%x_%j.log
# Rebuild the submission folder (manuscript/build/make_submission.sh) and render the manuscript, the supplementary
# information, the cover letter and the Chinese reading copy to PDF + page PNGs for checking
# (work/render/{main,si,cover,zh}).
# usage: sbatch -J render jobs/slurm_build_render.sh   (or: srun -p q05 ... bash jobs/slurm_build_render.sh)
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
bash manuscript/build/make_submission.sh
R=/home/user/zhangzhishuai/soft/libreoffice_25.8.7/tools/render_docx.sh
mkdir -p work/render/zh && cp submission/中文版/HapGAM_manuscript_zh.docx work/render/zh/   # ASCII path for LibreOffice
for pair in "submission/HapGAM_manuscript.docx main" "manuscript/build/HapGAM_supplementary.docx si" \
            "submission/HapGAM_cover_letter.docx cover" "work/render/zh/HapGAM_manuscript_zh.docx zh"; do
  set -- $pair
  rm -rf "work/render/$2/pages" "work/render/$2"/sheet_*.png
  bash $R "$1" "work/render/$2" 2>&1 | grep -E "pages|rror" || true
done
echo "build+render done $(date +%T)"
