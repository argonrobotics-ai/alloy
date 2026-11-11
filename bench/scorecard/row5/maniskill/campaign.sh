#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Row 5 ManiSkill campaign: 2 scenes x 3 outputs x 3 sizes x 3 world counts, one process per cell, alone on the GPU.
# Run after setup.sh:  bash campaign.sh [OUT_DIR]   (default ./out; rows -> OUT_DIR/maniskill_g4.jsonl, logs/, png/)
# Resumable: a cell with a row is skipped. SYNC=host (default) reads a value back after the last step of each repeat.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$(realpath -m "${1:-out}"); mkdir -p "$OUT/logs" "$OUT/png"
cd "$HERE"  # rows record the command as "bench.py ...", which is what the resume check greps for
ROWS=$OUT/maniskill_g4.jsonl
for scene in cartpole franka_cube; do for out in rgb rgb+depth rgb+depth+seg; do for size in 64 128 256; do for n in 256 1024 4096; do
  tag="$scene-${out//+/_}-$size-n$n"
  grep -q "\"command\": \"bench.py --scene $scene --outputs $out --size $size --worlds $n " "$ROWS" 2>/dev/null && continue
  timeout 900 ~/ms/bin/python bench.py --scene $scene --outputs $out --size $size --worlds $n \
    --sync "${SYNC:-host}" --out "$ROWS" --pngdir "$OUT/png" > "$OUT/logs/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    printf '{"engine":"maniskill","scene":"%s","outputs":"%s","width":%s,"height":%s,"worlds":%s,"status":"failed","reason":"process rc=%s (timeout 900 s = 124); see logs/%s.log","command":"bench.py --scene %s --outputs %s --size %s --worlds %s (killed)","utc":"%s"}\n' \
      $scene $out $size $size $n $rc $tag $scene $out $size $n "$(date -u +%FT%TZ)" >> "$ROWS"
  fi
  echo "$(date -u +%FT%TZ) rc=$rc $tag"
done; done; done; done
echo "$(date -u +%FT%TZ) CAMPAIGN_DONE"
