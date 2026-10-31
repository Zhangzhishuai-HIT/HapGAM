#!/bin/bash
# Runs on gpu10.  At 2026-10-21 06:29 it resumes (SIGCONT) any stopped HapGAM python process so that
# the 06:30 watchdog's SIGTERM takes effect.
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
sleep $(( $(date -d "2026-10-21 06:29:00" +%s) - $(date +%s) ))
for p in $(pgrep -u zhangzhishuai python); do
  st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); cwd=$(readlink /proc/$p/cwd 2>/dev/null)
  if [ "$st" == "T" ] && [[ "$cwd" == $ROOT* ]]; then kill -CONT $p; echo "$(date '+%F %T') resumed $p" >> $ROOT/logs/gpu/watchdog57.log; fi
done
