#!/bin/bash
# Confirmatory GPU runs on the temporary GPUs 5 and 7 (hard stop 06:30 via launch_gpu_temp.sh).
# Run only after scripts/freeze.py.  usage: launch_confirmatory_gpu57.sh <primary> <secondary>
#   primary   = hapgam_e or hapgam (pre-registered selection rule)
#   secondary = the other HapGAM variant
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/HapGAM
[ -f docs/FREEZE.json ] || { echo "not frozen: run scripts/freeze.py first"; exit 1; }
PM=$1; SV=$2
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
R="$PY scripts/run_model.py --gpu-cap 12000"
L=jobs/launch_gpu_temp.sh
$L 5 12000 c5a -- jobs/run_seq.sh "$R --model $PM --scenario C2" "$R --model $PM --scenario C1"; sleep 20
$L 5 12000 c5b -- jobs/run_seq.sh "$R --model $PM --scenario grin2soynam" "$R --model $PM --scenario grin2nust"; sleep 20
$L 5 9000 c5c -- jobs/run_seq.sh \
  "$R --model gblup --scenario C2" "$R --model egblup --scenario C2" "$R --model gblup --scenario C1" "$R --model egblup --scenario C1" \
  "$R --model gblup --scenario grin2soynam" "$R --model egblup --scenario grin2soynam" \
  "$R --model gblup --scenario grin2nust" "$R --model egblup --scenario grin2nust" \
  "$R --model gblup --scenario soynam_lofo" "$R --model egblup --scenario soynam_lofo" \
  "$R --model gblup --scenario soynam_random" "$R --model egblup --scenario soynam_random" \
  "$R --model rkhs --scenario C2" "$R --model hapgblup --scenario C2" "$R --model rkhs --scenario grin2soynam" \
  "$R --model hapgblup --scenario grin2soynam"; sleep 20
$L 5 9000 c5d -- jobs/run_seq.sh "$R --model $PM --scenario soynam_lofo" "$R --model $PM --scenario soynam_random"; sleep 20
$L 7 12000 c7a -- jobs/run_seq.sh "$R --model $SV --scenario C2" "$R --model $SV --scenario grin2soynam"; sleep 20
$L 7 12000 c7b -- jobs/run_seq.sh "$R --model hapgam_e_linear --scenario C2" "$R --model hapgam_e_linear --scenario grin2soynam" \
  "$R --model hapgam_e_linear --scenario C1"; sleep 20
$L 7 10000 c7c -- jobs/run_seq.sh "$R --model mlp --scenario C2" "$R --model dnngp --scenario C2" "$R --model mlp --scenario C1" \
  "$R --model dnngp --scenario C1" "$R --model dnngp --scenario grin2soynam" "$R --model mlp --scenario grin2soynam"; sleep 20
$L 7 12000 c7d -- jobs/run_seq.sh "$R --model $PM --scenario C3"; sleep 20
$L 7 12000 c7e -- jobs/run_seq.sh "$R --model $PM --scenario C4"
