#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPU test runner (SPEC §6.3, WP0).
#   scripts/gpu_run.sh <WP> "<pytest args>" [--vm NAME] [--zone ZONE] [--exclusive] [--python REMOTE_PY]
# Rsyncs this worktree to <vm>:~/wt/<WP>/ (no .venv/.git), runs
#   cd ~/wt/<WP> && PYTHONPATH=. <REMOTE_PY> -m pytest <args>
# under `flock ~/alloy-gpu.lock` with a 20-minute cap per lock hold, then copies bench/results/ back.
# Under the lock it first installs the pinned Newton (scripts/newton_env.sh) if the VM has another one.
# Starts the VM if it is TERMINATED. --exclusive also waits (holding the lock) until no other process uses the GPU
# (timing metrics, PRF-08). Exit code = pytest's (124 = hit the 20-minute cap).
# Env: ALLOY_GPU_USER (ssh user, default $USER), ALLOY_DEVICE (default cuda:0), ALLOY_GPU_LOCK_WAIT_S (default 5400).
set -euo pipefail

usage() { sed -n '2,10p' "$0"; exit 2; }
[[ $# -ge 2 ]] || usage
WP="$1"; PYTEST_ARGS="$2"; shift 2
VM="a dev GPU VM"; ZONE=""; EXCLUSIVE=0; REMOTE_PY='$HOME/alloy/.venv/bin/python'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --exclusive) EXCLUSIVE=1; shift ;;
    --python) REMOTE_PY="$2"; shift 2 ;;
    *) echo "gpu_run: unknown argument $1" >&2; usage ;;
  esac
done
[[ "$WP" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "gpu_run: bad WP name '$WP'" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUSER="${ALLOY_GPU_USER:-${USER:-$(whoami)}}"
LOCK_CAP_S=1200
LOCK_WAIT_S="${ALLOY_GPU_LOCK_WAIT_S:-5400}"
DEVICE="${ALLOY_DEVICE:-cuda:0}"
PIN="$(sed -n 's/^NEWTON_PIN: Final\[str\] = "\([0-9a-f]\{40\}\)"$/\1/p' "$ROOT/alloy/core/newton_compat.py")"
[[ -n "$PIN" ]] || { echo "gpu_run: NEWTON_PIN not found" >&2; exit 2; }
KEY="$HOME/.ssh/google_compute_engine"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10
     -o ServerAliveInterval=30)

log() { echo "gpu_run[$WP@$VM]: $*" >&2; }

[[ -n "$ZONE" ]] || ZONE="$(gcloud compute instances list --filter="name=($VM)" --format='value(zone.basename())' | head -1)"
[[ -n "$ZONE" ]] || { log "no VM named $VM in the project"; exit 3; }

status() { gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)'; }
st="$(status)"
if [[ "$st" == "TERMINATED" || "$st" == "STOPPED" || "$st" == "SUSPENDED" ]]; then
  log "VM is $st; starting it"
  gcloud compute instances start "$VM" --zone "$ZONE" --quiet >&2
fi
for _ in $(seq 60); do st="$(status)"; [[ "$st" == "RUNNING" ]] && break; sleep 5; done
[[ "$st" == "RUNNING" ]] || { log "VM did not reach RUNNING (status $st)"; exit 3; }

ips="$(gcloud compute instances describe "$VM" --zone "$ZONE" \
        --format='value(networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)')"
HOST=""
for _ in $(seq 24); do
  for ip in $ips; do
    if "${SSH[@]}" "$RUSER@$ip" true 2>/dev/null; then HOST="$RUSER@$ip"; break 2; fi
  done
  sleep 5
done
[[ -n "$HOST" ]] || { log "cannot ssh to $VM ($ips)"; exit 3; }
log "host $HOST, zone $ZONE"

log "rsync worktree -> ~/wt/$WP/"
"${SSH[@]}" "$HOST" "mkdir -p wt/$WP"
rsync -az --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude 'bench/results/' \
  -e "${SSH[*]}" "$ROOT/" "$HOST:wt/$WP/"

set +e
"${SSH[@]}" "$HOST" bash -s <<REMOTE
set -uo pipefail
cd "\$HOME/wt/$WP" || exit 3
mkdir -p bench/results
exec 9>"\$HOME/alloy-gpu.lock"
if ! flock -w $LOCK_WAIT_S 9; then echo "gpu_run: lock wait exceeded ${LOCK_WAIT_S}s" >&2; exit 3; fi
if [[ $EXCLUSIVE == 1 ]]; then
  for i in \$(seq 40); do
    busy=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [[ "\$busy" == 0 ]] && break
    echo "gpu_run: --exclusive: \$busy GPU process(es) running, waiting" >&2; sleep 15
  done
  [[ "\$busy" == 0 ]] || { echo "gpu_run: GPU never idle; not an exclusive run" >&2; exit 4; }
fi
got='import importlib.metadata as m, json; d = m.distribution("newton").read_text("direct_url.json")'
have=\$($REMOTE_PY -c "\$got; print(json.loads(d or '{}').get('vcs_info', {}).get('commit_id', ''))" 2>/dev/null)
if [[ "\$have" != "$PIN" ]]; then  # the private pin (docs/notes/wp0-newton-pin.md), under the lock
  echo "gpu_run: VM newton \${have:-unknown} is not the pin $PIN; running scripts/newton_env.sh" >&2
  PATH="\$HOME/.local/bin:\$PATH" bash scripts/newton_env.sh $REMOTE_PY >&2 || exit 3
fi
echo "gpu_run: lock held on \$(hostname) at \$(date -u +%FT%TZ); pytest $PYTEST_ARGS" >&2
PYTHONPATH=. ALLOY_DEVICE=$DEVICE OMP_NUM_THREADS=4 timeout $LOCK_CAP_S $REMOTE_PY -m pytest $PYTEST_ARGS
REMOTE
rc=$?
set -e

if "${SSH[@]}" "$HOST" "test -d wt/$WP/bench/results && [ -n \"\$(ls -A wt/$WP/bench/results)\" ]"; then
  mkdir -p "$ROOT/bench/results"
  rsync -az -e "${SSH[*]}" "$HOST:wt/$WP/bench/results/" "$ROOT/bench/results/"
  log "copied bench/results/ back"
fi
[[ $rc -eq 124 ]] && log "pytest hit the ${LOCK_CAP_S}s lock cap"
log "pytest exit code $rc"
exit $rc
