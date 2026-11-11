#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fresh-install check, run inside a clean container (tools/deps/container.sh starts it):
#   install_check.sh <requirement> [<find-links dir>]
# Makes a venv, times one `pip install`, measures the venv, times `import alloy.sim` cold and warm, runs the smoke
# example (and $EXTRA_CHECK, a script in this directory, when set), and prints one JSON line. Needs only python3 with
# venv and network access to PyPI.
set -uo pipefail
REQ="${1:?usage: install_check.sh <requirement> [find-links dir]}"
LINKS="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HOME}/alloy-install-check"
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK" || exit 1
now() { python3 -c 'import time; print(time.time())'; }
secs() { python3 -c "print(round($2 - $1, 1))"; }

python3 -m venv venv && . venv/bin/activate
pip install -q --upgrade pip > pip-upgrade.log 2>&1
ARGS=(--no-cache-dir); [ -n "$LINKS" ] && ARGS+=(--find-links "$LINKS")
t0=$(now); pip install "${ARGS[@]}" "$REQ" > pip.log 2>&1; rc=$?; t1=$(now)
venv_mb=$(du -sm venv | cut -f1)
site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
top=$(du -sm "$site"/* 2>/dev/null | sort -rn | head -6 | awk '{printf "%s\"%s\": %s", (NR>1?", ":""), $2, $1}' \
      | sed "s#$site/##g")
imp() { python -c 'import time; t=time.time(); import alloy.sim; print(round(time.time()-t, 2))' 2>>import.log | tail -1; }
i1=$(imp); i2=$(imp)
gpu=$(python -c 'import warp as wp; wp.config.quiet=True; wp.init(); print(wp.get_cuda_device_count())' 2>/dev/null | tail -1)
t2=$(now); python "$HERE/smoke_example.py" > smoke.json 2> smoke.log; src=$?; t3=$(now)
smoke=$(grep '^{' smoke.json 2>/dev/null | tail -1)
extra=null; xrc=0
if [ -n "${EXTRA_CHECK:-}" ]; then
  python "$HERE/$EXTRA_CHECK" > extra.json 2> extra.log; xrc=$?
  extra=$(grep '^{' extra.json 2>/dev/null | tail -1); extra=${extra:-null}
fi
python - "$rc" "$(secs "$t0" "$t1")" "$venv_mb" "{${top}}" "${i1:-null}" "${i2:-null}" "${gpu:-0}" "$src" \
         "$(secs "$t2" "$t3")" "${smoke:-null}" "$REQ" "$extra" "$xrc" <<'EOF'
import json, platform, sys
a = sys.argv[1:]
print(json.dumps({"requirement": a[10], "pip_rc": int(a[0]), "install_s": float(a[1]), "venv_mb": int(a[2]),
                  "largest_mb": json.loads(a[3]), "import_s": [None if x == "null" else float(x) for x in a[4:6]],
                  "cuda_devices": int(a[6] or 0), "smoke_rc": int(a[7]), "smoke_s": float(a[8]),
                  "smoke": json.loads(a[9]), "extra_check": json.loads(a[11]), "extra_rc": int(a[12]),
                  "python": platform.python_version(), "machine": platform.machine()}))
EOF
[ "$rc" -ne 0 ] && tail -5 pip.log >&2
[ "$src" -ne 0 ] && tail -15 smoke.log >&2
[ "$xrc" -ne 0 ] && tail -15 extra.log >&2
exit $(( rc != 0 || src != 0 || xrc != 0 ))
