#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Look around the library and check a scene without simulating it (seconds, no GPU needed).
# Run: bash examples/cli/01_explore.sh        (uses `alloy`, or `python -m alloy.cli` in a checkout)
set -euo pipefail
alloy() { if type -P alloy >/dev/null; then command alloy "$@"; else python -m alloy.cli "$@"; fi; }

alloy list scenes                                  # library scenes
alloy list materials --match steel --limit 3       # name, kind, density, friction, restitution
alloy list adds                                    # recipe "add" kinds and their arguments
alloy ops step                                     # the JSON schema of one operation, with an example
alloy check demo_peg_d12 --quick                   # cards and overlaps; every finding says what to change
