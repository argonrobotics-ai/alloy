#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Many calls in one process: build once, then step, ramp a joint, read state and check, all against the same
# simulation (its id is "lift"). The reply has one entry per request; it stops at the first failed call.
# Run: bash examples/cli/03_session.sh
set -euo pipefail
alloy() { if type -P alloy >/dev/null; then command alloy "$@"; else python -m alloy.cli "$@"; fi; }
alloy batch "$(dirname "$0")/03_session.json" --pretty
