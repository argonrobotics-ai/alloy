#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Mixed-batch sweep on one GPU: one process per point, batch doubled until a point fails (out of memory) or the
# speed stops rising. Run from the Alloy tree, inside its venv:
#   bash bench/scorecard/mixed/sweep.sh OUT MIXNAME TASKS [PER_TYPE_START] [PER_TYPE_MAX]
# e.g. bash bench/scorecard/mixed/sweep.sh ~/out/mix5 mix5 franka_lift,ur10_reach,go2_flat,g1_flat,hand_cube 256 65536
# Writes OUT/<tag>.json per point and OUT/logs/<tag>.log; OUT/sweep.done when finished. Re-running skips points
# that already have a JSON file.
set -uo pipefail
OUT=$1; MIX=$2; TASKS=$3; START=${4:-256}; MAX=${5:-65536}
K=$(awk -F, '{print NF}' <<<"$TASKS")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
export PYTHONPATH=.

point() {  # tag, then mixed.py arguments; 0 when the point produced a speed or quality result
  local tag=$1; shift
  if [[ ! -f "$OUT/$tag.json" ]]; then
    echo "$(date -u +%H:%M:%SZ) start $tag"
    # repeated lines (MuJoCo Warp prints its overflow warning every step) are kept once, with a count at the end
    timeout 5400 python "$HERE/mixed.py" "$@" --tasks "$TASKS" --out "$OUT" --tag "$tag" 2>&1 \
      | awk '{n[$0]++; if (n[$0] == 1) print} END {for (l in n) if (n[l] > 1) print "REPEATED " n[l] "x: " l}' > "$OUT/logs/$tag.log"
    echo "$(date -u +%H:%M:%SZ) exit $? $tag"
  fi
  [[ -f "$OUT/$tag.json" ]] && ! grep -q '"error"' "$OUT/$tag.json"
}

sweep() {  # label, then engine arguments: doubles the batch until a point fails
  local label=$1; shift
  local per=$START
  while (( per <= MAX )); do
    point "${MIX}_${label}_speed_$((per * K))" "$@" --mode speed --envs $((per * K)) || break
    per=$((per * 2))
  done
}

QW=${QUALITY_PER_TYPE:-1024}
for spec in "alloy_multi --engine alloy --form multi" "alloy_one --engine alloy --form one_scene" \
            "alloy5ms_multi --engine alloy_5ms --form multi" \
            "mjw_sub4 --engine newton_mjw --substeps 4" "mjw_sub2 --engine newton_mjw --substeps 2" \
            "mjw_sub1 --engine newton_mjw --substeps 1"; do
  read -r label args <<<"$spec"
  # MJW_BUFFERS=<json of mjw_buffers.py>: the MuJoCo Warp rows run with measured buffer sizes, labelled mjws_*
  [[ -n "${MJW_BUFFERS:-}" && $label == mjw_* ]] && label=mjws_${label#mjw_}
  [[ " ${ONLY:-$label} " == *" $label "* ]] || continue
  # shellcheck disable=SC2086
  point "${MIX}_${label}_quality_$((QW * K))" $args --mode quality --envs $((QW * K))
  # shellcheck disable=SC2086
  sweep "$label" $args
done
date -u +%FT%TZ > "$OUT/sweep.done"
