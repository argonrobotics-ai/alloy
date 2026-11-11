#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install Alloy into a fresh virtual environment, then run the README quick start as a check.
#
#   bash install.sh                  # Alloy's wheel from the GitHub Release (checksum checked), the rest from PyPI
#   bash install.sh --wheels DIR     # Alloy's wheel from a local folder (the Release's files: *.whl, SHA256SUMS)
#   bash install.sh --extras import,mesh   # also the optional extras (pyproject.toml), Alloy kept at this version:
#                                          # never `pip install -e .` in this environment
#   On a machine where Warp finds an NVIDIA GPU, the envs extra (PyTorch + huggingface_hub) is added unless
#   --extras is given or --no-torch.
#   bash install.sh --help
#
# Linux x86_64 (CPU or NVIDIA GPU) or macOS on Apple silicon (CPU); Linux aarch64 is not tested yet. In a clone the
# environment is the clone's .venv; run on its own (downloaded from the Release) it is ./alloy-venv. Python 3.12 is
# used; if the machine has none, uv (https://docs.astral.sh/uv/) fetches it. Nothing else is changed except uv's
# cache and, when uv is not installed, a uv binary in ~/.cache/alloy/uv (your PATH and shell files are left alone).
set -euo pipefail

VERSION="0.1.0"     # the Alloy release this script installs (make_release checks it against alloy/__init__.py)
REPO="argonrobotics-ai/alloy"
RELEASE_BASE="${ALLOY_RELEASE_BASE:-https://github.com/$REPO/releases/download}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE=0
if [ -f "$ROOT/alloy/__init__.py" ] && [ -f "$ROOT/README.md" ]; then CLONE=1; fi
if [ "$CLONE" = 1 ]; then VENV="$ROOT/.venv"; else VENV="$PWD/alloy-venv"; fi
WHEELS=""
WHEEL=""
PYTHON=""
CPU=0
GPU=0               # 1 once Warp in the new environment finds a CUDA device (report_device)
NO_TORCH=0
QUICKSTART=1
EXTRAS=""
PCI_DEVICES=/sys/bus/pci/devices
CONTAINER_MARKERS="/.dockerenv /run/.containerenv"   # files Docker and Podman put in a container
EXTRAS_RE='^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$'
PY_WANT="3.12"
UV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/alloy/uv"

usage() {
  cat <<EOF
Usage: bash install.sh [options]

  --wheels DIR      take Alloy's wheel from DIR (alloy_engine-$VERSION-py3-none-any.whl, with SHA256SUMS to check
                    it against) instead of downloading it; the other packages come from PyPI, or from DIR when it
                    has them
  --python PATH     use this Python 3.12 interpreter
  --venv DIR        create the environment in DIR (default: .venv in a clone, else ./alloy-venv)
  --cpu             run the check on the CPU even when an NVIDIA GPU is present
  --no-quickstart   install only; skip the quick start check
  --extras LIST     also install these optional extras, comma-separated, e.g. import,mesh (or barcode, envs);
                    Alloy stays at this wheel, Warp and MuJoCo at its pins
  --no-torch        on a machine with an NVIDIA GPU, do not add the envs extra (PyTorch + huggingface_hub,
                    about 3 GB), which is added there by default when --extras is not given
  -h, --help        show this help

Without --wheels the wheel and SHA256SUMS come from the GitHub Release v$VERSION of $REPO
($RELEASE_BASE/v$VERSION/), over plain https: no login or token.
EOF
}

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'install.sh: error: %s\n' "$*" >&2; exit 1; }

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --wheels) [ $# -ge 2 ] || die "--wheels needs a directory"; WHEELS="$(cd "$2" 2>/dev/null && pwd)" \
                  || die "--wheels: no such directory: $2"; shift 2 ;;
      --python) [ $# -ge 2 ] || die "--python needs a path"; PYTHON="$2"; shift 2 ;;
      --venv) [ $# -ge 2 ] || die "--venv needs a directory"
              VENV="$(cd "$(dirname "$2")" 2>/dev/null && pwd)/$(basename "$2")" \
                || die "--venv: the parent folder of $2 does not exist"; shift 2 ;;
      --cpu) CPU=1; shift ;;
      --no-quickstart) QUICKSTART=0; shift ;;
      --no-torch) NO_TORCH=1; shift ;;
      --extras) [ $# -ge 2 ] || die "--extras needs a comma-separated list, e.g. import,mesh"
                [[ "$2" =~ $EXTRAS_RE ]] || die "--extras: not a comma-separated list of names: $2"
                EXTRAS="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown option: $1" ;;
    esac
  done
}

check_platform() {
  local os arch have
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os/$arch" in
    Linux/x86_64|Darwin/arm64) ;;
    Linux/aarch64) say "note: Linux aarch64 is not tested yet; installing anyway" ;;
    *) die "Alloy supports Linux x86_64 and macOS on Apple silicon (arm64); this is $os $arch" ;;
  esac
  [ "$CLONE" = 1 ] || return 0
  have="$(sed -n 's/^__version__ = "\([^"]*\)".*/\1/p' "$ROOT/alloy/__init__.py" | head -n 1)"
  [ "$have" = "$VERSION" ] || die "this clone is Alloy $have but this script installs $VERSION: use the install.sh \
of the same release, or a developer install (pip install -e .)"
}

# A Python 3.12 that can create a venv with pip (Debian and Ubuntu split venv/ensurepip into python3.12-venv).
usable_python() {
  local py="$1"
  command -v "$py" >/dev/null 2>&1 || return 1
  "$py" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" 2>/dev/null || return 1
  "$py" -c "import venv, ensurepip" 2>/dev/null
}

find_python() {
  if [ -n "$PYTHON" ]; then
    usable_python "$PYTHON" && return 0
    die "--python $PYTHON is not a Python $PY_WANT with the venv module"
  fi
  local py
  for py in python3.12 python3 python; do
    if usable_python "$py"; then PYTHON="$(command -v "$py")"; return 0; fi
  done
  return 1
}

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
  if [ -x "$UV_DIR/uv" ]; then echo "$UV_DIR/uv"; return 0; fi
  say "no Python $PY_WANT with venv found: fetching uv into $UV_DIR (PATH and shell files untouched)" >&2
  mkdir -p "$UV_DIR"
  local installer="https://astral.sh/uv/install.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$installer" | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh >&2
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$installer" | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh >&2
  else
    die "no Python $PY_WANT and no curl/wget to fetch it: install curl (e.g. apt-get install curl) and rerun"
  fi
  [ -x "$UV_DIR/uv" ] || die "uv did not install into $UV_DIR"
  echo "$UV_DIR/uv"
}

make_venv() {
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import sys, pip; sys.exit(sys.version_info[:2] != (3, 12))" \
       2>/dev/null; then
    say "reusing $VENV"
    return 0
  fi
  rm -rf "$VENV"
  if find_python; then
    say "creating $VENV with $PYTHON"
    "$PYTHON" -m venv "$VENV"
  else
    local uv
    uv="$(find_uv)"
    say "creating $VENV with uv's Python $PY_WANT"
    "$uv" venv --quiet --seed --python "$PY_WANT" "$VENV"
  fi
  "$VENV/bin/python" -m pip --version >/dev/null || die "the environment in $VENV has no pip"
}

fetch() {   # fetch URL FILE over plain https (the Release is public)
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    "$VENV/bin/python" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"
  fi
}

download_wheels() {
  local dest="$VENV/wheels" base="$RELEASE_BASE/v$VERSION" name
  rm -rf "$dest" && mkdir -p "$dest"
  say "downloading Alloy $VERSION from $base/"
  for name in "alloy_engine-$VERSION-py3-none-any.whl" SHA256SUMS; do
    fetch "$base/$name" "$dest/$name" || die "could not download $base/$name (offline? pass --wheels DIR)"
  done
  WHEELS="$dest"
}

verify_wheels() {
  WHEEL="$WHEELS/alloy_engine-$VERSION-py3-none-any.whl"
  [ -f "$WHEEL" ] || die "no alloy_engine-$VERSION-py3-none-any.whl in $WHEELS"
  if [ ! -f "$WHEELS/SHA256SUMS" ]; then
    say "no SHA256SUMS in $WHEELS: skipping the checksum check"
    return 0
  fi
  say "checking the wheel against SHA256SUMS"
  (cd "$WHEELS" && "$VENV/bin/python" - <<'PY'
import hashlib, pathlib, sys
bad, wheels = [], 0
for line in pathlib.Path("SHA256SUMS").read_text().splitlines():
    if not line.strip():
        continue
    digest, name = line.split(maxsplit=1)
    p = pathlib.Path(name.lstrip("*"))
    if not p.is_file():
        continue                      # the Release's other files (the sdist, install.sh) need not be here
    wheels += p.suffix == ".whl"
    if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        bad.append(p.name)
if bad:
    sys.exit(f"install.sh: error: checksum mismatch: {', '.join(bad)}")
if not wheels:
    sys.exit("install.sh: error: SHA256SUMS lists no wheel that is here")
PY
  ) || die "the files in $WHEELS do not match SHA256SUMS"
}

install_wheels() {
  say "installing Alloy $VERSION (Warp, MuJoCo and the other dependencies from PyPI)"
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q --find-links "$WHEELS" "$WHEEL" \
    || die "pip could not install Alloy (output above)"
  (cd "${TMPDIR:-/tmp}" && "$VENV/bin/python" -c "import alloy, warp; print('  alloy', alloy.__version__, \
'| warp', warp.config.version)")
}

# The extras named by --extras (pyproject.toml's optional dependencies), with Alloy held at this wheel. `pip install
# -e ".[x]"` would reinstall Alloy from the checkout and re-resolve its dependencies instead.
install_extras() {
  local py="$VENV/bin/python" known name
  known="$("$py" -c 'import importlib.metadata as m
print(",".join(sorted(set(m.metadata("alloy-engine").get_all("Provides-Extra") or []))))')"
  if [ -z "$EXTRAS" ] && [ "$GPU" = 1 ] && [ "$NO_TORCH" = 0 ] && [[ ",$known," == *",envs,"* ]]; then
    EXTRAS="envs"
    say "adding the envs extra (PyTorch + huggingface_hub, about 3 GB): Warp found an NVIDIA GPU, and the Gym envs \
and camera images hand back torch tensors on it; --no-torch skips this"
  fi
  [ -n "$EXTRAS" ] || return 0
  for name in ${EXTRAS//,/ }; do
    [[ ",$known," == *",$name,"* ]] || die "--extras: Alloy $VERSION has no extra '$name' (it has: ${known//,/, })"
  done
  say "installing the extras $EXTRAS"
  "$py" -m pip install --disable-pip-version-check -q --find-links "$WHEELS" "alloy-engine[$EXTRAS] @ file://$WHEEL" \
    || die "could not install the extras $EXTRAS (pip output above); Alloy itself is installed in $VENV"
  "$py" -m pip check --disable-pip-version-check >/dev/null 2>&1 || say "warning: pip check reports a conflict: \
$("$py" -m pip check --disable-pip-version-check 2>&1 | head -n 3)"
}

# CPU robot scenes compile a small C step on first use (alloy/route/_mjc, which looks up $CC, else cc); without a
# compiler they still run, on a slower fallback. Only a note: nothing fails.
check_compiler() {
  if [ "$(uname -s)" = Darwin ] && [ -z "${CC:-}" ]; then
    xcode-select -p >/dev/null 2>&1 && return 0
    say "note: no Apple command line tools: CPU robot scenes will run on a slower fallback instead of MuJoCo's C \
step. Run 'xcode-select --install' (or set CC): the step then compiles on first use, no reinstall needed"
    return 0
  fi
  local cc="${CC:-cc}"
  command -v "$cc" >/dev/null 2>&1 && return 0
  say "note: no C compiler ('$cc' not found): CPU robot scenes will run on a slower fallback instead of MuJoCo's C \
step. Install one (e.g. apt-get install gcc) or set CC: the step then compiles on first use, no reinstall needed"
}

# The device Alloy will use, as Warp in the new environment sees it (alloy.build picks the GPU when Warp has one);
# sets GPU=1 for a CUDA device. Warns when the driver lists a GPU that Warp cannot start (a container without GPU
# access, a driver fault), and says how to get a driver when the machine has an NVIDIA GPU but no driver.
report_device() {
  local warp smi=""
  warp="$("$VENV/bin/python" - 2>/dev/null <<'PY'
import warp as wp
wp.config.quiet = True
if hasattr(wp, "LOG_WARNING"):
    wp.config.log_level = wp.LOG_WARNING
print("device=" + (f"GPU 0: {wp.get_device('cuda:0').name} (cuda:0)" if wp.get_cuda_device_count() else "CPU"))
PY
)" || warp=""
  warp="$(printf '%s\n' "$warp" | sed -n 's/^device=//p' | tail -n 1)"
  case "$warp" in GPU*) GPU=1 ;; esac
  if [ "$CPU" = 1 ]; then
    say "device: CPU (--cpu)"
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 && smi="$(nvidia-smi -L 2>/dev/null | head -n 1 | sed 's/ (UUID.*//')"
  case "$warp" in
    GPU*) say "device: $warp" ;;
    CPU) if [ -n "$smi" ]; then
           say "warning: the driver lists $smi, but Warp cannot use it, so Alloy runs on the CPU (much slower). \
The reason: $VENV/bin/python -c 'import warp; warp.init()'"
         elif nvidia_hardware && in_container; then
           say "note: this container sees an NVIDIA GPU but not its driver (no nvidia-smi inside). Start it with GPU \
access (docker run --gpus all, with the NVIDIA Container Toolkit on the host), or from a CUDA image. Until then \
Alloy runs on the CPU"
         elif nvidia_hardware; then
           say "note: this machine has an NVIDIA GPU but no NVIDIA driver (no working nvidia-smi). On Ubuntu: \
sudo ubuntu-drivers install, then reboot (or start from a CUDA image); check with nvidia-smi, then run install.sh \
again. Until then Alloy runs on the CPU"
         else
           say "device: CPU (no NVIDIA GPU found)"
         fi ;;
    *) say "device: not known (Warp did not load; the quick start below shows why)" ;;
  esac
}

# True when the PCI bus shows an NVIDIA device (vendor 0x10de), driver or not (Linux; false elsewhere).
nvidia_hardware() {
  grep -qsx 0x10de "$PCI_DEVICES"/*/vendor
}

# True inside a Docker or Podman container (they leave a marker file at the root).
in_container() {
  local f
  for f in $CONTAINER_MARKERS; do [ -e "$f" ] && return 0; done
  return 1
}
# The README quick start (a box dropped onto a table, its rest height checked); on its own, the same scene inline.
run_quickstart() {
  [ "$QUICKSTART" = 1 ] || return 0
  say "running the quick start (the first run compiles kernels: about 2 minutes on a GPU, 1-3 on a CPU)"
  local work start readme=""
  work="$(mktemp -d)"
  start=$(date +%s)
  [ "$CLONE" = 1 ] && readme="$ROOT/README.md"
  (cd "$work" && README="$readme" CPU="$CPU" "$VENV/bin/python" - <<'PY'
import os, pathlib, re
if os.environ["CPU"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
INLINE = """
import alloy
scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_box("table", size=(0.8, 0.8, 0.70), pos=(0, 0, 0.35), static=True)
scene.add_box("box", size=(0.08, 0.08, 0.08), pos=(0, 0, 0.90), mass=0.5)
sim = alloy.build(scene)
traj = sim.run(1.5, record=["box"])
z = float(traj.world(0)["box.pos"][-1, 2])
print(f"box rests at z = {z:.3f} m, health ok: {sim.check()['ok']}")
assert abs(z - 0.74) < 0.005 and sim.check()["ok"]
"""
code, where = INLINE, "install.sh quick start"
if os.environ["README"]:
    m = re.search(r"## Quick start.*?```python\n(.*?)```", pathlib.Path(os.environ["README"]).read_text(), re.S)
    if not m:
        raise SystemExit("install.sh: error: no python block under '## Quick start' in README.md")
    code, where = m.group(1), "README.md quick start"
exec(compile(code, where, "exec"), {"__name__": "__main__"})
PY
  ) || die "the quick start failed (output above); the environment is installed in $VENV"
  rm -rf "$work"
  say "quick start passed in $(( $(date +%s) - start )) s"
}

main() {
  parse_args "$@"
  check_platform
  local start
  start=$(date +%s)
  make_venv
  [ -n "$WHEELS" ] || download_wheels
  verify_wheels
  install_wheels
  report_device
  install_extras
  check_compiler
  run_quickstart
  say "done in $(( $(date +%s) - start )) s. Use it with:  source $VENV/bin/activate"
  echo "    Next: AGENTS.md (the API in one page), examples/, docs/guide/ (https://github.com/$REPO)."
}

main "$@"
