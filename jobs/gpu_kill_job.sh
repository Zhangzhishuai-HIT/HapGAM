#!/bin/bash
# usage: gpu_kill_job.sh NAME...   (run on gpu10) - terminates the whole process tree started by
# launch_gpu*.sh for logs/gpu/NAME.pid (timeout wrapper, run_seq shell and python children).
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
tree() { local p=$1; echo $p; for c in $(ps -o pid= --ppid $p); do tree $c; done; }
for n in "$@"; do
  P=$(cat $ROOT/logs/gpu/$n.pid 2>/dev/null) || continue
  PIDS=$(tree $P)
  kill -CONT $PIDS 2>/dev/null; kill -TERM $PIDS 2>/dev/null
  echo "$n: terminated $(echo $PIDS | wc -w) processes"
done
