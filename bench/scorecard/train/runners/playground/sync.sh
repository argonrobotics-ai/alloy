#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# From the worktree: copy the runner, schema.py and score.py to the VM (~/playground/). No rm anywhere.
set -euo pipefail
VM=${1:-alloy-st-playground-1}
TRAIN=$(cd "$(dirname "$0")/../.." && pwd)
cd "$TRAIN"
bash vm/tvm.sh push "$VM" runners/playground/ /home/$USER/playground/
bash vm/tvm.sh push "$VM" schema.py /home/$USER/playground/schema.py
