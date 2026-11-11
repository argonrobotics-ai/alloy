#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Row 4 (time to train), Alloy cell: fresh DLVM Ubuntu 22.04 + NVIDIA driver 580 VM -> a venv that trains.
#   bash install.sh [ALLOY_TREE] [VENV]      defaults: ~/alloy  ~/alloy-venv
# ALLOY_TREE is an Alloy source tree (a `git archive` of the commit under test, with bench/scorecard/train in it).
# Installs: Python 3.12 (uv fetches it), Alloy editable with its pinned Warp / MuJoCo, PyTorch, rsl-rl-lib.
# Versions are pinned by constraints.txt beside this script (the freeze of the pilot VM); re-runnable.
set -euo pipefail
TREE=${1:-$HOME/alloy}
VENV=${2:-$HOME/alloy-venv}
HERE=$(cd "$(dirname "$0")" && pwd)
UV_VERSION=0.9.7
RSL_RL=5.5.2

[[ -f "$TREE/pyproject.toml" ]] || { echo "no Alloy tree at $TREE" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.12 "$VENV"
CONSTRAINTS=()
[[ -s "$HERE/constraints.txt" ]] && CONSTRAINTS=(-c "$HERE/constraints.txt")
uv pip install --python "$VENV/bin/python" "${CONSTRAINTS[@]}" -e "$TREE[envs]" "rsl-rl-lib==$RSL_RL" tensordict pip
"$VENV/bin/python" - <<'PY'
import torch, warp, mujoco, rsl_rl, importlib.metadata as md
import alloy
print("alloy", alloy.__version__, "| warp", warp.__version__, "| mujoco", mujoco.__version__,
      "| torch", torch.__version__, "cuda", torch.cuda.is_available(), "| rsl-rl-lib", md.version("rsl-rl-lib"))
assert torch.cuda.is_available(), "PyTorch does not see the GPU (driver 580+ needed)"
PY
echo "ok: run with  PYTHONPATH=$TREE $VENV/bin/python $TREE/bench/scorecard/train/runners/alloy/run.py --help"
