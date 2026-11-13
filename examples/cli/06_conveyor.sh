#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Act on a running simulation without Python: a kinematic belt carries a tote (drive), stops, the tote is put back
# above it (pose) and falls. Also lists the library's robots, whose names a recipe takes as a robot's path:
# {"add": "robot", "name": "arm", "path": "emb_e2_panda"}.
# Run: bash examples/cli/06_conveyor.sh
set -euo pipefail
alloy() { if type -P alloy >/dev/null; then command alloy "$@"; else python -m alloy.cli "$@"; fi; }
alloy list robots --match panda
alloy batch "$(dirname "$0")/06_conveyor.json" --pretty
