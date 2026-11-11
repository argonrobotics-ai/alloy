#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the fresh-install check in a clean container and print its JSON line.
#   tools/deps/container.sh [--gpu] [--image IMAGE] [--cpus N] [--memory SIZE] [--check SCRIPT.py] <requirement> [<wheelhouse dir>]
# Example: tools/deps/container.sh "alloy-engine==0.0.1+ga7829db5" ~/wheelhouse
# The container gets only this directory (read-only, as /deps) and the wheelhouse (read-only, as /wheels); it runs
# as an unprivileged user with network access to PyPI. --gpu passes all GPUs (needs the NVIDIA container toolkit).
set -euo pipefail
IMAGE=python:3.12-slim-bookworm; GPU=(); CPUS=4; MEM=16g; CHECK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=(--gpus all); shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --memory) MEM="$2"; shift 2 ;;
    --check) CHECK="$2"; shift 2 ;;   # e.g. import_extra_check.py, run after the smoke example
    -*) echo "container.sh: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
REQ="${1:?usage: container.sh [--gpu] [--image IMAGE] <requirement> [<wheelhouse dir>]}"
HOUSE="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER=(docker); docker info >/dev/null 2>&1 || DOCKER=(sudo -n docker)
MOUNTS=(-v "$HERE:/deps:ro"); LINKS=""
if [ -n "$HOUSE" ]; then MOUNTS+=(-v "$(cd "$HOUSE" && pwd):/wheels:ro"); LINKS=/wheels; fi
NAME="alloy-install-check-$$"
"${DOCKER[@]}" pull -q "$IMAGE" >/dev/null
exec "${DOCKER[@]}" run --rm --name "$NAME" "${GPU[@]}" --cpus "$CPUS" --memory "$MEM" "${MOUNTS[@]}" \
  --user 1000:1000 -e HOME=/tmp/home -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e EXTRA_CHECK="$CHECK" "$IMAGE" \
  bash -c 'mkdir -p "$HOME" && bash /deps/install_check.sh "$0" "$1"' "$REQ" "$LINKS"
