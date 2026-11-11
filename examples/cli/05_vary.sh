#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Scene variants: four copies of a table scene with the free objects moved (+-2 cm, +-15 degrees) and each look given
# another texture of its family, a contact sheet of them, then one variant run like any scene.
# Run: bash examples/cli/05_vary.sh [out_dir]      (CPU or GPU; the first contact sheet compiles the camera, ~1 min)
# Textures come from the CC0 pool; ones not fetched yet (python -m alloy.looks --fetch-textures) are left out of the
# sheet only, and result.sheet_left_out lists them.
set -euo pipefail
alloy() { if type -P alloy >/dev/null; then command alloy "$@"; else python -m alloy.cli "$@"; fi; }
out=${1:-out_cli_vary}
mkdir -p "$out"

cat > "$out/scene.json" <<'JSON'
{"dt": 0.01,
 "objects": [{"add": "ground"},
             {"add": "box", "name": "table", "size": [0.3, 0.3, 0.4], "pos": [0, 0, 0.2], "static": true,
              "material": "wood_pine"},
             {"add": "box", "name": "a", "size": [0.04, 0.04, 0.04], "pos": [-0.06, 0, 0.42], "material": "wood_beech"},
             {"add": "box", "name": "b", "size": [0.04, 0.04, 0.04], "pos": [0.06, 0, 0.42], "material": "abs"},
             {"add": "sphere", "name": "ball", "radius": 0.02, "pos": [0, 0.07, 0.42], "material": "steel_carbon"}]}
JSON

alloy vary "$out/scene.json" --n 4 --pos 0.02 --yaw 15 --seed 3 --out "$out/variants" --sheet
# reply: result.files (variant_000.json ...), result.variants[k].moves {"a": {"dx", "dy", "yaw_deg"}} and .textures,
# result.sheet (contact_sheet.png); the same seed gives the same variants
alloy run "$out/variants/variant_002.json" --seconds 0.5 --names a,b,ball --out "$out/variant_002.npz"
