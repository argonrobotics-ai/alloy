#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU CI (SPEC §6.2, WP0). Warms the Warp kernel cache (not timed), then runs the CPU suite.
# Also runs, once their owners have landed them: the analytic quick suite against the baseline (D23).
# Usage: scripts/ci_cpu.sh [extra pytest args]
# Env: PYTHON (default: the main clone's venv, else python3), ALLOY_CI_SKIP_WARM=1 to skip warming,
#      ALLOY_CI_WORKERS=N (N > 1; needs pytest-xdist) to run pytest as N workers (--dist loadfile: a file's tests stay on
#      one worker, each worker has its own Warp cache, tests/conftest.py) with the analytic suite alongside (SCR-GATE-1).
#      SCR-GATE-2 (WP1's fast lane and post-merge runner call this script):
#      ALLOY_CI_PATHS: space-separated test paths. Unset means `tests`. Set but empty means no default path, and the
#        caller passes its selection as extra args.
#      ALLOY_CI_DIST: the xdist mode with ALLOY_CI_WORKERS > 1, default loadfile.
#      ALLOY_CI_SKIP_ANALYTIC=1: no analytic quick suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMMON="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
DEFAULT_PY="${COMMON:+$(dirname "$COMMON")/.venv/bin/python}"
[[ -x "${DEFAULT_PY:-}" ]] || DEFAULT_PY="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PY" ]] || DEFAULT_PY="$(command -v python3)"
PY="${PYTHON:-$DEFAULT_PY}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# SCR-WP8-8: users get the MuJoCo C route on CPU by default, but CI (pytest and the analytic suite) stays on the Warp
# kernels, which stand in for the GPU ones (coordinator ruling 2026-11-06 14:18Z). Forced, like tests/conftest.py.
export ALLOY_CPU_ROUTE=warp
WORKERS="${ALLOY_CI_WORKERS:-1}"
DIST="${ALLOY_CI_DIST:-loadfile}"
case "$DIST" in
  load|loadscope|loadfile|loadgroup|worksteal|no) ;;
  *) echo "ci_cpu: ALLOY_CI_DIST=$DIST is not an xdist mode (load loadscope loadfile loadgroup worksteal no)" >&2
     exit 2 ;;
esac
PATHS=()   # expanded as ${PATHS[@]+"${PATHS[@]}"}, like XDIST below
read -r -a PATHS <<< "${ALLOY_CI_PATHS-tests}" || true
if [[ ${#PATHS[@]} -eq 0 && $# -eq 0 ]]; then   # pytest would collect the whole worktree
  echo "ci_cpu: ALLOY_CI_PATHS is empty and no test args were given" >&2
  exit 2
fi

if [[ "${ALLOY_CI_SKIP_WARM:-0}" != "1" ]]; then
  echo "ci_cpu: warming the Warp kernel cache (not timed)"
  "$PY" -m alloy.core.warm --device cpu
fi

HAVE_ANALYTIC=0
[[ -f alloy/bench/__main__.py && -f bench/goldens/analytic_baseline.json ]] && HAVE_ANALYTIC=1
if [[ "${ALLOY_CI_SKIP_ANALYTIC:-0}" == "1" && $HAVE_ANALYTIC -eq 1 ]]; then
  echo "ci_cpu: analytic suite skipped (ALLOY_CI_SKIP_ANALYTIC=1)"   # no "exit" line: gate_report reads it as not run
  HAVE_ANALYTIC=0
fi
# Reports go to a scratch dir (ALLOY_CI_OUT to keep them), never into the worktree: CI must not leave files that a
# builder's `git add -A` would commit.
OUT="${ALLOY_CI_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/alloy-ci-XXXXXX")}"
analytic() {
  "$PY" -m alloy.bench run --suite analytic --device cpu --quick --baseline bench/goldens/analytic_baseline.json \
    --out "$OUT"
}

XDIST=()   # expanded as ${XDIST[@]+"${XDIST[@]}"}: an empty array is "unbound" under set -u in bash 3.2 (macOS)
APID=""
trap '[[ -z "$APID" ]] || kill "$APID" 2>/dev/null || true' EXIT   # no orphaned analytic run if we are stopped
if [[ "$WORKERS" -gt 1 ]]; then
  XDIST=(-n "$WORKERS" --dist "$DIST")
  if [[ $HAVE_ANALYTIC -eq 1 ]]; then   # alongside pytest; its output is printed after pytest's
    analytic > "$OUT/analytic.log" 2>&1 &
    APID=$!
  fi
fi

echo "ci_cpu: pytest (not gpu and not slow)$([[ $WORKERS -le 1 ]] || echo " with -n $WORKERS --dist $DIST")"
start=$(date +%s)
rc=0
"$PY" -m pytest ${PATHS[@]+"${PATHS[@]}"} -q -m "not gpu and not slow" ${XDIST[@]+"${XDIST[@]}"} "$@" || rc=$?
# The merge gate reads these exit lines (pre-existing-red rule, scripts/gate_report.py): a run that failed only on
# tests still runs the analytic suite, so a waived failure does not skip it.
echo "ci_cpu: pytest exit $rc, wall $(( $(date +%s) - start )) s (budget 300 s on the Mac, SPEC §6.2)"

# The analytic suite counts after a pass or test failures (pytest exit 0 or 1), not after a crash or interrupt (run
# serially, it does not start then).
brc=""
if [[ -n "$APID" ]]; then
  brc=0
  wait "$APID" || brc=$?
  APID=""
  echo "ci_cpu: analytic quick suite vs baseline (D23, ran alongside pytest), reports in $OUT"
  cat "$OUT/analytic.log"
elif [[ $rc -le 1 && $HAVE_ANALYTIC -eq 1 ]]; then
  echo "ci_cpu: analytic quick suite vs baseline (D23), reports in $OUT"
  brc=0
  analytic || brc=$?
fi
if [[ -n "$brc" && $rc -le 1 ]]; then
  echo "ci_cpu: analytic suite exit $brc"
  [[ $rc -ne 0 ]] || rc=$brc
fi
# A passing run's scratch reports are removed (a failing one keeps them); ALLOY_CI_OUT keeps them.
[[ ( -n "$brc" && "$brc" -ne 0 ) || -n "${ALLOY_CI_OUT:-}" ]] || rm -rf "$OUT"
exit "$rc"
