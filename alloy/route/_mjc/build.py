# SPDX-License-Identifier: Apache-2.0
"""Build ``liballoy_mjc`` from this package's C sources against the installed ``mujoco`` wheel (SCR-WP8-6).

Until WP0's pyproject config ships a prebuilt library, :func:`library_path` compiles on first use into
``$ALLOY_CACHE_DIR/mjc/<key>/`` (default ``~/.cache/alloy``), keyed by the sources, the MuJoCo version, the compiler
and the machine; a file lock keeps concurrent processes to one compile (~2 s). A prebuilt
``liballoy_mjc.<so|dylib>`` next to the sources wins when present.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("alloy.route")

HERE = Path(__file__).resolve().parent
SOURCES = ("step.c", "noslip.c", "cone.c", "boxbox_line.c")
HEADERS = ("alloy_mjc.h", "mjc_internal.h")
LIB_NAME = "liballoy_mjc.dylib" if sys.platform == "darwin" else "liballoy_mjc.so"


class BuildError(RuntimeError):
    """The extension cannot be built or loaded here (the message says why)."""


def mujoco_paths() -> tuple[str, Path, Path]:
    """(version, include dir, libmujoco) of the installed wheel."""
    try:
        import mujoco
    except ImportError as e:
        raise BuildError(f"mujoco is not installed ({e})") from e
    root = Path(mujoco.__file__).resolve().parent
    libs = sorted(root.glob("libmujoco*.so*")) + sorted(root.glob("libmujoco*.dylib"))
    inc = root / "include"
    if not libs or not (inc / "mujoco" / "mujoco.h").is_file():
        raise BuildError(f"the mujoco wheel at {root} has no headers or libmujoco to build against")
    return mujoco.__version__, inc, libs[0]


def _darwin_tools(cc: str) -> None:
    """macOS: ``/usr/bin/cc`` without the Command Line Tools is a shim, and running it can open the installer dialog
    (in every new process, as a failure is not cached). ``xcode-select -p`` asks without one (WP2's review)."""
    if sys.platform != "darwin" or Path(cc).parent != Path("/usr/bin"):
        return
    try:
        res = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BuildError(f"no C compiler: cannot ask xcode-select for the developer tools ({e})") from e
    if res.returncode != 0:
        raise BuildError("no C compiler: the Xcode Command Line Tools are not installed (xcode-select -p failed)")


def compiler() -> tuple[str, str]:
    """(compiler path, first line of its --version). It must run: on macOS ``/usr/bin/cc`` exists without the Xcode
    Command Line Tools and only prints an ``xcrun: error``."""
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        raise BuildError("no C compiler (set CC or install cc/gcc/clang)")
    _darwin_tools(cc)
    try:
        res = subprocess.run([cc, "--version"], capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BuildError(f"no C compiler: {cc} does not run ({e})") from e
    first = (res.stdout.splitlines() or [""])[0]
    if res.returncode != 0 or not first:
        why = (res.stderr.strip().splitlines() or ["no version"])[-1][:200]
        raise BuildError(f"no C compiler: {cc} --version failed ({why})")
    return cc, first


def build_key(version: str, cc_version: str) -> str:
    h = hashlib.sha256()
    for name in SOURCES + HEADERS:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes())
    h.update(f"{version}|{cc_version}|{platform.machine()}|{sys.platform}".encode())
    if sys.platform == "darwin":                  # a darwin link change rebuilds (an old one linked to the framework)
        h.update(" ".join(link_flags(Path("libmujoco"), "darwin")).encode())
    return h.hexdigest()[:16]


def cache_root() -> Path:
    return Path(os.environ.get("ALLOY_CACHE_DIR", Path.home() / ".cache" / "alloy")) / "mjc"


def link_flags(libmujoco: Path, platform_name: str = sys.platform) -> list[str]:
    """How the library finds MuJoCo. Linux: against the wheel's libmujoco, with an rpath to it. macOS: no link to it;
    its symbols resolve at load from the libmujoco ``import mujoco`` loaded (``-undefined dynamic_lookup``, as Python
    extensions resolve libpython). The macOS wheel's ``libmujoco.<v>.dylib`` names itself
    ``@rpath/mujoco.framework/Versions/A/...``, a path the wheel does not have, so a library linked to it would not
    load (mujoco 3.16.0, macosx_11_0_arm64)."""
    if platform_name == "darwin":
        return ["-dynamiclib", "-undefined", "dynamic_lookup", "-lm"]
    return ["-shared", str(libmujoco), f"-Wl,-rpath,{libmujoco.parent}", "-lm", "-lpthread"]


def compile_to(out: Path, cc: str, inc: Path, libmujoco: Path) -> None:
    cmd = [cc, "-std=c11", "-O2", "-fPIC", "-D_POSIX_C_SOURCE=200809L", f"-I{inc}", f"-I{HERE}",
           *(str(HERE / s) for s in SOURCES), *link_flags(libmujoco), "-o", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise BuildError(f"compiling liballoy_mjc failed ({cc}): {res.stderr.strip()[-800:]}")
    said = (res.stderr or "").strip()
    if said:                          # built, with the compiler's or linker's warnings (e.g. ld on dynamic_lookup)
        log.warning("liballoy_mjc built with messages from %s: %s", cc, said[-800:])


def library_path() -> Path:
    """The library to load: prebuilt next to the sources, else the cached build (compiled now if missing)."""
    prebuilt = HERE / LIB_NAME
    if prebuilt.is_file():
        return prebuilt
    version, inc, libmujoco = mujoco_paths()
    cc, cc_version = compiler()
    target = cache_root() / build_key(version, cc_version) / LIB_NAME
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.parent / "build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            t0 = time.perf_counter()
            tmp = target.with_suffix(f".tmp{os.getpid()}")
            compile_to(tmp, cc, inc, libmujoco)
            os.replace(tmp, target)
            log.info("built %s against MuJoCo %s in %.1f s", target, version, time.perf_counter() - t0)
    return target


__all__ = ["BuildError", "LIB_NAME", "SOURCES", "build_key", "cache_root", "library_path", "link_flags",
           "mujoco_paths"]
