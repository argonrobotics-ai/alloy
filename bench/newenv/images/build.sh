#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# build.sh ARM: build the sandbox image newenv-ARM on a Docker host with the nvidia runtime as the DEFAULT runtime
# (the GPU is then visible to the build's warm-up step; legacy builder only, no BuildKit), time the install, run the
# smoke test in the finished image under the eval run's limits, and append the record to $OUT (a JSON-lines file
# outside the repository; measured values are never written into the tree).
#
#   BASE=<kit sandbox image> OUT=<file> bash build.sh genesis    # BASE: built from bench/agents/sandbox/Dockerfile
#   ALLOY_STAGE=<dir with release/ and wheels/> BASE=... OUT=... bash build.sh alloy
#
# An arm missing here is looked up in $NEWENV_EXTRA_CONFIG/images/ARM (arms kept outside the public package).
# Record: arm, install_s (the pip / uv / install.sh step), warm_s (kernel-cache warm-up), build_s, image_gb,
# versions (pip freeze), smoke (the smoke's last lines).
set -euo pipefail
ARM=${1:?usage: build.sh ARM}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); D=${DOCKER:-sudo docker}
BASE=${BASE:?set BASE to the cold-agent kit sandbox image}; OUT=${OUT:?set OUT to a records file outside the repository}
TAG=newenv-$ARM; SRC=$HERE/$ARM
[ -f "$SRC/Dockerfile" ] || SRC=${NEWENV_EXTRA_CONFIG:-/nonexistent}/images/$ARM
[ -f "$SRC/Dockerfile" ] || { echo "no Dockerfile for $ARM"; exit 1; }
$D info 2>/dev/null | grep -q "Default Runtime: nvidia" || { echo "docker default runtime is not nvidia (nvidia-ctk runtime configure --runtime=docker --set-as-default)"; exit 1; }
CTX=$(mktemp -d); cp -r "$SRC"/. "$CTX"/
if [ "$ARM" = alloy ]; then   # the staged release tree + wheelhouse go in the context
  : "${ALLOY_STAGE:?set ALLOY_STAGE to the stage dir (release/, wheels/)}"
  cp -r "$ALLOY_STAGE/release" "$CTX/release"; cp -r "$ALLOY_STAGE/wheels" "$CTX/wheels"
fi
AGENT_HOME=$($D run --rm "$BASE" sh -c 'echo $HOME')     # the sandbox user's home, from the kit's image
T0=$(date +%s)
$D build --build-arg BASE="$BASE" --build-arg AGENT_HOME="$AGENT_HOME" -t "$TAG" "$CTX" 2>&1 \
  | tee "$OUT.$ARM.build.log" | tail -n 40
BUILD_S=$(( $(date +%s) - T0 )); rm -rf "$CTX"
SIZE=$($D image inspect --format '{{.Size}}' "$TAG")
NE=$($D run --rm --gpus all "$TAG" bash -c 'd=${NEWENV_DIR:-$VIRTUAL_ENV/.newenv}; cat $d/install_s $d/warm_s; echo ---; cat $d/freeze.txt')
SMOKE=$($D run --rm --gpus all --memory 13g --memory-swap 13g --cpus 4 --shm-size 2g -v "$SRC/smoke.py:/tmp/smoke.py:ro" \
        "$TAG" python /tmp/smoke.py 2>&1 | tail -n 3)
python3 - "$ARM" "$BUILD_S" "$SIZE" "$SMOKE" "$OUT" "$NE" <<'PY'
import json, sys
arm, build_s, size, smoke, out, ne = sys.argv[1:7]
head, _, freeze = ne.partition("---")
install_s, warm_s = (int(x) for x in head.split())
rec = {"arm": arm, "install_s": install_s, "warm_s": warm_s, "build_s": int(build_s),
       "image_gb": round(int(size) / 1e9, 2), "versions": [l for l in freeze.strip().splitlines() if l], "smoke": smoke}
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(json.dumps({k: v for k, v in rec.items() if k != "versions"}))
PY
