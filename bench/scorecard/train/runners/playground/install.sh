#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ON a fresh DLVM Ubuntu 22.04 + driver 580 VM: MuJoCo Playground 0.2.0 (JAX MJX + MuJoCo Warp) + rsl_rl + torch.
# Venv: ~/pg/.venv. Re-runnable. Versions pinned; see NOTES.md.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
mkdir -p ~/pg && cd ~/pg
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python \
  "mujoco==3.17.0" "mujoco-mjx==3.17.0" "playground==0.2.0" "jax[cuda12]==0.12.0.1" \
  warp-lang==1.19.0 brax==0.14.2 numpy==2.5.4 nvidia-cuda-runtime-cu12==12.9.79 nvidia-cudnn-cu12==9.28.0.42 \
  pynvml "rsl-rl-lib==${RSL_RL_VERSION:-5.5.0}" torch
uv pip freeze --python .venv/bin/python > freeze.txt
grep -iE '^(jax|jaxlib|mujoco|playground|warp|brax|rsl|torch|numpy)' freeze.txt
