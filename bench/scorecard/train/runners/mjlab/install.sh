#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Row 4, mjlab: fresh DLVM Ubuntu 22.04 + driver 580 -> mjlab 1.6.0 in ~/mjlab-env (uv, Python 3.12).
# Re-runnable. Also fetches MuJoCo Menagerie (BSD-3) at a pinned commit for the Go2 variant (not copied into the repo).
set -euo pipefail
MJLAB_VERSION=1.6.0
MENAGERIE_REV=${MENAGERIE_REV:-98a70c569c4c1b6da467a20012f766ee1f8dc521}   # pinned (2026-11-09 main)
sudo apt-get update -y >/dev/null
sudo apt-get install -y libegl1 git >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 "$HOME/mjlab-env"
uv pip install --python "$HOME/mjlab-env/bin/python" "mjlab==$MJLAB_VERSION" tensorboard
# mjlab 1.6.0 pins rsl-rl-lib==5.4.2; the protocol says rsl_rl 5.5.x, and 5.5.2 runs unchanged under mjlab 1.6.0
uv pip install --python "$HOME/mjlab-env/bin/python" "rsl-rl-lib==5.5.2"
if [ ! -d "$HOME/menagerie" ]; then git clone https://github.com/google-deepmind/mujoco_menagerie "$HOME/menagerie"; fi
[ -z "$MENAGERIE_REV" ] || git -C "$HOME/menagerie" checkout "$MENAGERIE_REV"
git -C "$HOME/menagerie" rev-parse HEAD
"$HOME/mjlab-env/bin/python" -c "import os; os.environ['MUJOCO_GL']='disable'; import mjlab, mujoco, warp, rsl_rl; print('mjlab ok')"
uv pip list --python "$HOME/mjlab-env/bin/python" | grep -iE 'mjlab|mujoco|warp|rsl|torch'
