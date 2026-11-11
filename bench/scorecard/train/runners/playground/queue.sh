#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ON the VM: all row-4 playground pilots, seed 0, in order. Re-runnable after a preemption: a finished job leaves
# ~/out/<name>.done and is skipped; an unfinished one restarts from scratch (its partial .jsonl is moved aside).
# Warp C-level prints "solver iterations limit reached" every step (23 GB of log): filtered out.
cd ~/playground
PY=~/pg/.venv/bin/python
OUT=~/out
run() {  # run NAME script args...
  local name=$1; shift
  [ -f $OUT/$name.done ] && return 0
  [ -f $OUT/$name.jsonl ] && mv $OUT/$name.jsonl $OUT/$name.partial.$(date +%s).jsonl
  $PY "$@" --out $OUT/$name.jsonl 2>&1 | grep --line-buffered -v -e "solver iterations" -e "To disable" > $OUT/$name.log
  touch $OUT/$name.done
}
S="--seed 0"
run go2_flat_mjx_native_s0 native.py --task go2_flat $S --num-evals 20
run go2_flat_mjx_s0_std0.3_evidence run.py --task go2_flat $S --num-envs 4096 --max-iterations 500 --init-std 0.3 --tag diagnostic_init_std_0.3_with_cmd_vel
run go2_flat_warp_s0 run.py --task go2_flat $S --num-envs 4096 --max-iterations 1000 --impl warp
run franka_lift_mjx_s0 run.py --task franka_lift $S --num-envs 4096 --max-iterations 3000 --max-wall-s 4500
run cube_reorient_mjx_s0 run.py --task cube_reorient $S --num-envs 4096 --max-iterations 5000 --max-wall-s 4500
run franka_lift_mjx_native_s0 native.py --task franka_lift $S --num-evals 20 --max-wall-s 4500
run cube_reorient_mjx_native_s0 native.py --task cube_reorient $S --num-evals 40 --max-wall-s 4500
run franka_lift_warp_s0 run.py --task franka_lift $S --num-envs 4096 --max-iterations 3000 --max-wall-s 4500 --impl warp
run cube_reorient_warp_s0 run.py --task cube_reorient $S --num-envs 4096 --max-iterations 5000 --max-wall-s 4500 --impl warp
touch $OUT/ALL_DONE
