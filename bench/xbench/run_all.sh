#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# run_all.sh: re-run the Newton/Alloy side of every xbench protocol; see run_all.py and CROSS_ENGINE.md "Re-running".
#   bash bench/xbench/run_all.sh --engine-under-test newton            # the Newton baseline, headline tier
#   bash bench/xbench/run_all.sh --engine-under-test alloy --tier full # Alloy (needs the alloy.xbench adapter)
# PYTHON defaults to the repo's .venv (Alloy venv); on a GPU VM point it at that VM's Alloy venv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" "$HERE/run_all.py" --python "$PY" "$@"
