#!/bin/bash
# TEMPORARY (user permission on 2026-10-20 ~23:37): GPUs 5 and 7 on gpu10 may be used until
# 2026-10-21 06:30 only.  Every job is wrapped in `timeout` so it terminates by 06:30; launches
# after 06:20 are refused.  After 06:30 only GPU 2 may be used (jobs/launch_gpu.sh).
# usage: launch_gpu_temp.sh GPU NEED_MB LOGNAME -- command args...
set -euo pipefail
GPU=$1; NEED=$2; NAME=$3; shift 3
[ "$1" == "--" ] && shift
case "$GPU" in 5|7) ;; *) echo "only GPU 5 or 7 allowed here"; exit 2;; esac
DEADLINE=$(date -d "2026-10-21 06:30:00" +%s)
NOW=$(date +%s)
LEFT=$(( DEADLINE - NOW ))
if [ "$LEFT" -lt 600 ]; then echo "less than 10 min before the 06:30 deadline; not launching"; exit 3; fi
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
mkdir -p "$ROOT/logs/gpu"
CMD=$(printf '%q ' "$@")
ssh -o BatchMode=yes gpu10 "
  cd $ROOT
  FREE=\$(nvidia-smi -i $GPU --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print \$1-\$2}')
  if [ \"\$FREE\" -lt \$(( $NEED + 4096 )) ]; then echo \"GPU $GPU has only \${FREE} MiB free; not launching $NAME\"; exit 3; fi
  echo \"launch $NAME on GPU $GPU (free \${FREE} MiB), hard stop in $LEFT s (06:30)\"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    nohup timeout --signal=TERM --kill-after=60 $LEFT nice -n 5 $CMD > logs/gpu/$NAME.log 2>&1 < /dev/null &
  echo \$! > logs/gpu/$NAME.pid
"
