#!/bin/bash
# Run several commands sequentially (used to keep one process per GPU slot).
# usage: run_seq.sh "cmd1" "cmd2" ...
for c in "$@"; do echo ">>> $c"; eval "$c" || echo "FAILED: $c"; done
