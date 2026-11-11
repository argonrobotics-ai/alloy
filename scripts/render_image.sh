#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the render VM image (family alloy-render-blender45): DLVM + OptiX libs + Blender 4.5 LTS + a warm Cycles
# kernel cache, so a farm VM renders ~2.5 min sooner than one booted from the stock DLVM image.
#   scripts/render_image.sh [ZONE ...]      (tries zones in order; deletes its build VM at the end)
# Builds on a CPU VM by default (spot L4s are often all in use by the farm); RENDER_IMAGE_GPU=1 builds on an L4 and
# also bakes a warm OptiX kernel cache. The DLVM image already carries the NVIDIA driver, so no GPU is needed.
# The farm uses this family when it exists (alloy.render.vm.image_args), else the DLVM image + startup script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
FAMILY="alloy-render-blender45"
NAME="alloy-render-imgbuild-$(date +%s)"
ZONES=("$@"); [[ ${#ZONES[@]} -gt 0 ]] || ZONES=(us-west1-b us-central1-a us-east1-b us-east4-a us-west4-a europe-west4-a)
GPU="${RENDER_IMAGE_GPU:-0}"
MACHINE=n2-standard-8; [[ "$GPU" == 1 ]] && MACHINE=g2-standard-8
SSH=(ssh -i "$HOME/.ssh/google_compute_engine" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ConnectTimeout=15)
log() { echo "render_image: $*" >&2; }

ZONE=""
for z in "${ZONES[@]}"; do
  if gcloud compute instances create "$NAME" --zone "$z" --machine-type "$MACHINE" --provisioning-model SPOT \
      --instance-termination-action DELETE --max-run-duration 2h --maintenance-policy TERMINATE \
      --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
      --boot-disk-size 50GB --metadata install-nvidia-driver=True \
      --metadata-from-file "startup-script=$ROOT/scripts/render_vm_startup.sh" \
      --labels purpose=alloy,role=wp20-imgbuild >/dev/null 2>&1; then ZONE="$z"; break; fi
  log "no capacity in $z"
done
[[ -n "$ZONE" ]] || { log "no zone could create the build VM"; exit 3; }
trap 'gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet >/dev/null 2>&1 || true' EXIT
IP="$(gcloud compute instances describe "$NAME" --zone "$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
log "build VM $NAME in $ZONE ($IP)"

for _ in $(seq 90); do
  "${SSH[@]}" "$USER@$IP" 'test -x /opt/blender/blender && test -e /usr/lib/x86_64-linux-gnu/libnvoptix.so.1' \
    2>/dev/null && break
  sleep 10
done
"${SSH[@]}" "$USER@$IP" 'test -x /opt/blender/blender' || { log "Blender never appeared"; exit 4; }

if [[ "$GPU" == 1 ]]; then
# Warm the OptiX/CUDA kernel cache where the farm points Blender (farm_remote.py CACHE_ENV) by rendering the
# synthetic bundle, which uses every feature the clips need (motion blur, curves, point clouds, volumes, SSS).
tar cf - -C "$ROOT" alloy/render/blender tests/render/synth.py | "${SSH[@]}" "$USER@$IP" 'mkdir -p /tmp/wp20 && tar xf - -C /tmp/wp20'
"${SSH[@]}" "$USER@$IP" 'bash -s' <<'REMOTE'
set -e
export OPTIX_CACHE_PATH=$HOME/.cache/cycles/optix CUDA_CACHE_PATH=$HOME/.cache/cycles/nv CUDA_CACHE_MAXSIZE=2147483647
mkdir -p "$OPTIX_CACHE_PATH" "$CUDA_CACHE_PATH"
cd /tmp/wp20 && "$(ls /opt/blender/*/python/bin/python3.* | head -1)" tests/render/synth.py /tmp/syn > /dev/null
/opt/blender/blender -b --python-exit-code 1 -P alloy/render/blender/build.py -- --bundle /tmp/syn/bundle \
  --look /tmp/syn/look.json --camera vertical --out /tmp/syn_out --still 12 --samples 4 > /tmp/warm.log 2>&1
du -sh "$HOME/.cache/cycles" | cat
rm -rf /tmp/wp20 /tmp/syn /tmp/syn_out; sudo apt-get clean
REMOTE
fi
"${SSH[@]}" "$USER@$IP" 'sudo apt-get clean; sudo rm -rf /tmp/* 2>/dev/null; true'

gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet >/dev/null
IMAGE="$FAMILY-$(date +%Y%m%d-%H%M)"
gcloud compute images create "$IMAGE" --source-disk "$NAME" --source-disk-zone "$ZONE" --family "$FAMILY" \
  --storage-location us --labels purpose=alloy,role=wp20-render >/dev/null
log "image $IMAGE (family $FAMILY) in project $PROJECT"
echo "$IMAGE"
