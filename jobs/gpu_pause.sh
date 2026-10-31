#!/bin/bash
# usage: gpu_pause.sh STOP|CONT <regex on the python command line>...
# Sends SIGSTOP/SIGCONT only to this user's python processes running scripts/run_model.py from the
# HapGAM directory whose command line matches one of the regexes.
SIG=$1; shift
ROOT=/home/user/zhangzhishuai/myhermes/2080/HapGAM
for p in $(pgrep -u zhangzhishuai python); do
  exe=$(readlink /proc/$p/exe 2>/dev/null); cwd=$(readlink /proc/$p/cwd 2>/dev/null)
  [[ "$exe" == */python* && "$cwd" == $ROOT* ]] || continue
  c=$(tr '\0' ' ' < /proc/$p/cmdline)
  [[ "$c" == *scripts/run_model.py* ]] || continue
  for re in "$@"; do
    if [[ "$c" =~ $re ]]; then kill -$SIG $p && echo "$SIG $p: ${c##*run_model.py}"; break; fi
  done
done
