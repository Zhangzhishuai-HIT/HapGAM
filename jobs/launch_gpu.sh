#!/bin/bash
# Launch a command on gpu10, GPU 2 ONLY (user rule, 2026-10-20), in the background.
# The job is launched only if GPU 2 has at least NEED_MB + 4096 MiB free; it never touches
# other processes.  Each process caps its own allocation (hapgam.gpu.setup_torch_device).
# usage: launch_gpu.sh NEED_MB LOGNAME -- command args...
set -euo pipefail
NEED=$1; NAME=$2; shift 2
[ "$1" == "--" ] && shift
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/hapgam/bin/python
mkdir -p "$ROOT/logs/gpu"
CMD=$(printf '%q ' "$@")
CMD=${CMD//env\/venv\/bin\/python/$PY}
ssh -o BatchMode=yes gpu10 "
  cd $ROOT
  FREE=\$(nvidia-smi -i 2 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print \$1-\$2}')
  if [ \"\$FREE\" -lt \$(( $NEED + 4096 )) ]; then echo \"GPU 2 has only \${FREE} MiB free; not launching $NAME\"; exit 3; fi
  echo \"launch $NAME on GPU 2 (free \${FREE} MiB)\"
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    nohup nice -n 5 $CMD > logs/gpu/$NAME.log 2>&1 < /dev/null &
  echo \$! > logs/gpu/$NAME.pid
"
