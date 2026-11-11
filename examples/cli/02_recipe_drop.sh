#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Describe a scene as JSON (a recipe), check it, then drop a box onto a table and save the trajectory.
# Run: bash examples/cli/02_recipe_drop.sh [out_dir]      (GPU if present, else CPU; the first build compiles kernels)
set -euo pipefail
alloy() { if type -P alloy >/dev/null; then command alloy "$@"; else python -m alloy.cli "$@"; fi; }
out=${1:-out_cli_drop}
mkdir -p "$out"

cat > "$out/scene.json" <<'JSON'
{"dt": 0.005,
 "objects": [{"add": "ground"},
             {"add": "box", "name": "table", "size": [0.8, 0.8, 0.70], "pos": [0, 0, 0.35], "static": true},
             {"add": "box", "name": "box", "size": [0.08, 0.08, 0.08], "pos": [0, 0, 0.90], "mass": 0.5}]}
JSON

alloy scene "$out/scene.json"                              # names and kinds; errors name the entry and the fix
alloy check "$out/scene.json" --quick                      # cards and overlaps at the start
alloy run "$out/scene.json" --seconds 1.5 --names box --out "$out/traj.npz"
# reply: result.final["box.pos"][2] is about 0.74 (table top 0.70 + half the box); result.saved is the .npz
