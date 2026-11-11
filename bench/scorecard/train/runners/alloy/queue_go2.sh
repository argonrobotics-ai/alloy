#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ON the VM: the Alloy go2_flat jobs in order (smoke, matched pilot, native). Re-runnable after a preemption (cron
# @reboot): a finished job leaves ~/out/<name>.done and is skipped; an unfinished one restarts from scratch (its
# partial .jsonl is moved aside). Alloy tree ~/alloy, venv ~/alloy-venv (install.sh).
cd ~/alloy || exit 1
export PYTHONPATH=$HOME/alloy
PY=$HOME/alloy-venv/bin/python
RUN=bench/scorecard/train/runners/alloy/run.py
OUT=$HOME/out
mkdir -p "$OUT"
run() {  # run NAME args...
  local name=$1; shift
  [ -f "$OUT/$name.done" ] && return 0
  [ -f "$OUT/$name.jsonl" ] && mv "$OUT/$name.jsonl" "$OUT/$name.partial.$(date +%s).jsonl"
  if $PY $RUN --task go2_flat "$@" --out "$OUT/$name.jsonl" > "$OUT/$name.log" 2>&1; then
    touch "$OUT/$name.done"
  else
    touch "$OUT/$name.failed"; exit 1   # a deterministic failure stops the queue; the log says why
  fi
}
run go2_smoke --seed 0 --num-envs 64 --max-iterations 5
run go2_flat_s0 --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 3000
run go2_flat_native_s0 --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 3000 --variant native
touch "$OUT/ALL_DONE"
