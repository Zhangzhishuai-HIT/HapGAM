#!/bin/bash
# Runs on gpu10.  At 2026-10-21 06:30 it terminates every process of this user that runs from the
# HapGAM directory on GPUs 5 or 7 (the temporary permission ends then).  Logs to logs/gpu/watchdog57.log.
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
DEADLINE=$(date -d "2026-10-21 06:30:00" +%s)
sleep $(( DEADLINE - $(date +%s) ))
UUIDS=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==5 || $1==7 {print $2}')
for pass in 1 2; do
  for U in $UUIDS; do
    for P in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$U" '$1==u {print $2}'); do
      OWNER=$(ps -o user= -p "$P" 2>/dev/null | tr -d ' ')
      CWD=$(readlink "/proc/$P/cwd" 2>/dev/null)
      if [ "$OWNER" == "zhangzhishuai" ] && [[ "$CWD" == $ROOT* ]]; then
        echo "$(date '+%F %T') pass $pass: stopping PID $P on $U" >> $ROOT/logs/gpu/watchdog57.log
        kill -TERM "$P" 2>/dev/null
      fi
    done
  done
  sleep 60
done
# also stop the timeout wrappers / run_seq parents started by launch_gpu_temp.sh
for f in $ROOT/logs/gpu/*.pid; do
  P=$(cat "$f"); C=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
  if [[ "$C" == timeout* ]]; then kill -TERM "$P" 2>/dev/null && echo "$(date '+%F %T') stopped wrapper $P ($f)" >> $ROOT/logs/gpu/watchdog57.log; fi
done
echo "$(date '+%F %T') watchdog done" >> $ROOT/logs/gpu/watchdog57.log
