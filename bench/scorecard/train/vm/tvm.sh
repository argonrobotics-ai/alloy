#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Row 4 (time to train) VM helper: one spot GPU VM per framework run.
#   tvm.sh up NAME [l4|g4]        create NAME (first zone that lands) and wait for ssh; never adopts an existing VM
#   tvm.sh ssh NAME [CMD...]      run CMD on NAME (exit 4 when NAME is not RUNNING; never starts or recreates it)
#   tvm.sh push NAME LOCAL REMOTE rsync LOCAL (file or dir) to REMOTE on the VM
#   tvm.sh pull NAME REMOTE LOCAL rsync REMOTE on the VM into LOCAL
#   tvm.sh status NAME            RUNNING / TERMINATED / ... / ABSENT
#   tvm.sh start NAME             restart a preempted (TERMINATED) NAME in place and wait for ssh; exit 5 on stock-out
#   tvm.sh delete NAME            delete NAME
#   tvm.sh list                   every VM this helper owns
# Rules: names alloy-st-*, label role=score-train (a VM is ours only when BOTH match), spot, STOP on preemption,
# 12 h max run, delete after use. l4 = g2-standard-8 + L4; g4 = g4-standard-48 + RTX PRO 6000 (the scorecard GPU).
# ZONES="a b c" overrides the zone list (default: EU and Asia first, the US L4s are reserved from the release cut).
set -uo pipefail
PROJECT=${PROJECT:?set PROJECT to your GCP project id}
ROLE=score-train
LABELS=${VM_LABELS:-purpose=alloy-bench},role=$ROLE
KEY=$HOME/.ssh/google_compute_engine
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
     -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
L4_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-northeast1-a asia-northeast1-c europe-west3-b europe-west2-a us-south1-a us-central1-a us-east4-a us-west1-a"
G4_ZONES="europe-west4-a europe-west4-b asia-southeast1-b asia-northeast1-b us-central1-b us-central1-f us-east4-c us-west1-b us-east1-b us-south1-a"
G() { gcloud --project "$PROJECT" "$@" < /dev/null; }
log() { echo "$(date -u +%H:%M:%SZ) tvm[${NAME:-}]: $*" >&2; }
die() { log "$*"; exit 1; }

check_name() { [[ "$NAME" == alloy-st-* ]] || die "refusing: names must start with alloy-st-"; }
zone_of() {  # ours only: name AND role label
  G compute instances list --filter="name=($NAME) AND labels.role=$ROLE" --format='value(zone.basename())' 2>/dev/null | head -1
}
status_of() {
  local z; z=$(zone_of); [[ -n "$z" ]] || { echo ABSENT; return; }
  G compute instances describe "$NAME" --zone "$z" --format='value(status)' 2>/dev/null
}
host_of() {
  local z ip; z=$(zone_of); [[ -n "$z" ]] || return 1
  for ip in $(G compute instances describe "$NAME" --zone "$z" \
      --format='value(networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null); do
    "${SSH[@]}" -n "$USER@$ip" true 2>/dev/null && { echo "$USER@$ip"; return 0; }
  done
  return 1
}
need_running() { local s; s=$(status_of); [[ "$s" == RUNNING ]] || { log "not RUNNING ($s)"; exit 4; }; }

create() {  # $1 = l4 | g4
  local z shape=() zones
  case "$1" in
    l4) shape=(--machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 --boot-disk-type=pd-balanced); zones=${ZONES:-$L4_ZONES} ;;
    g4) shape=(--machine-type=g4-standard-48 --boot-disk-type=hyperdisk-balanced); zones=${ZONES:-$G4_ZONES} ;;
    *) die "GPU class must be l4 or g4" ;;
  esac
  local errf; errf=$(mktemp)
  for z in $zones; do
    log "creating ($1) in $z"
    if G compute instances create "$NAME" --zone="$z" "${shape[@]}" \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
        --boot-disk-size=250GB --maintenance-policy=TERMINATE --provisioning-model=SPOT \
        --instance-termination-action=STOP --max-run-duration=12h \
        --metadata=install-nvidia-driver=True --labels="$LABELS" > "$errf" 2>&1; then
      rm -f "$errf"; log "created in $z"; return 0
    fi
    grep -o "ERROR.*\|code: .*\|message: .*" "$errf" | head -2 | cut -c1-160 >&2
  done
  rm -f "$errf"; return 1
}
wait_ssh() {
  local i h
  for i in $(seq 1 40); do h=$(host_of) && { echo "$h"; return 0; }; sleep 10; done
  return 1
}

cmd=${1:-}; NAME=${2:-}
case "$cmd" in
  list) G compute instances list --filter="labels.role=$ROLE" --format='table(name,zone.basename(),machineType.basename(),status,creationTimestamp)' ;;
  up)
    check_name
    exists=$(G compute instances list --filter="name=($NAME)" --format='value(name)' 2>/dev/null | head -1)
    [[ -z "$exists" ]] || die "a VM named $NAME already exists (status $(status_of)); pick another name or delete it"
    create "${3:-l4}" || die "no zone had capacity or quota"
    h=$(wait_ssh) || die "created but ssh never answered"
    "${SSH[@]}" "$h" "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" || log "nvidia-smi not ready yet"
    ;;
  ssh) check_name; need_running; h=$(host_of) || die "no ssh"; shift 2; "${SSH[@]}" "$h" "$@" ;;
  push) check_name; need_running; h=$(host_of) || die "no ssh"; rsync -az --exclude __pycache__ -e "${SSH[*]}" "$3" "${h}:$4" ;;
  pull) check_name; need_running; h=$(host_of) || die "no ssh"; mkdir -p "$4"; rsync -az -e "${SSH[*]}" "${h}:$3" "$4" ;;
  status) check_name; status_of ;;
  start)
    check_name; z=$(zone_of); [[ -n "$z" ]] || die "ABSENT (or not ours)"
    G compute instances start "$NAME" --zone "$z" > /dev/null 2>&1 || { log "start refused (stock-out or quota)"; exit 5; }
    h=$(wait_ssh) || die "started but ssh never answered"; log "up at $h"
    ;;
  delete)
    check_name; z=$(zone_of); [[ -n "$z" ]] || { log "ABSENT (or not ours)"; exit 0; }
    G compute instances delete "$NAME" --zone "$z" --quiet && log "deleted"
    ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
