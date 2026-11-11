#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ON the VM: sequential go2_flat pilots on the Warp backend (protocol std 1.0, then diagnostic std 0.3).
cd ~/playground
PY=~/pg/.venv/bin/python
$PY run.py --task go2_flat --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 4500 --impl warp --out ~/out/go2_flat_warp_s0.jsonl > ~/out/go2_flat_warp_s0.log 2>&1
$PY run.py --task go2_flat --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 4500 --impl warp --init-std 0.3 --tag diagnostic_init_std_0.3 --out ~/out/go2_flat_warp_s0_std0.3.jsonl > ~/out/go2_flat_warp_s0_std0.3.log 2>&1
$PY run.py --task go2_flat --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 4500 --init-std 0.3 --tag diagnostic_init_std_0.3_1000it --out ~/out/go2_flat_mjx_s0_std0.3_1000.jsonl > ~/out/go2_flat_mjx_s0_std0.3_1000.log 2>&1
echo done > ~/out/queue_go2_done
