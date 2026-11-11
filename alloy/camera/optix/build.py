# SPDX-License-Identifier: Apache-2.0
"""Build the OptiX helper (``libowf.so`` + ``owf_programs.optixir``) from ``csrc/`` into the cache.

The NVIDIA OptiX headers are under NVIDIA's SDK licence (not Apache-2.0), so Alloy never ships them and never fetches
them silently: the build refuses until the licence is accepted (``--accept-nvidia-optix-license`` or the environment
variable ``ALLOY_ACCEPT_NVIDIA_OPTIX_LICENSE=1``). The headers are then cloned (tag v9.0.0 of
github.com/NVIDIA/optix-dev) into ``$ALLOY_CACHE/camera/optix-dev-9.0.0``, unless ``OPTIX_SDK`` points at existing
headers (the directory that holds ``include/optix.h``). Nothing is installed system-wide; the compilers are ``g++`` (or
``g++-12``; ``CXX`` overrides) and ``nvcc`` (``CUDA_HOME``/``CUDA_PATH`` or ``PATH``). The result lands in
``$ALLOY_CACHE/camera/optix/<hash of the csrc files>/`` (``ALLOY_CACHE`` defaults to ``~/.cache/alloy``), so changing
the sources builds a new directory. The runtime library ``libnvoptix.so.1`` comes from the NVIDIA driver.

Example::

    python -m alloy.camera.optix build --accept-nvidia-optix-license

    from alloy.camera.optix.build import build_dir, build
    build(accept_licence=True)          # -> Path of the build directory
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..errors import CameraError

log = logging.getLogger(__name__)
OPTIX_TAG = "v9.0.0"
OPTIX_REPO = "https://github.com/NVIDIA/optix-dev"
LICENCE_ENV = "ALLOY_ACCEPT_NVIDIA_OPTIX_LICENSE"
LIB_NAME, PROGRAMS_NAME = "libowf.so", "owf_programs.optixir"
CSRC = Path(__file__).resolve().parent / "csrc"
LICENCE_NOTICE = (
    "The NVIDIA OptiX headers (github.com/NVIDIA/optix-dev, tag v9.0.0) are under NVIDIA's SDK licence, not "
    "Apache-2.0, so Alloy does not ship or fetch them on its own. To build the OptiX backend, read that licence and "
    "run: python -m alloy.camera.optix build --accept-nvidia-optix-license (or set "
    f"{LICENCE_ENV}=1), or download the headers yourself and point OPTIX_SDK at them.")


def cache_root() -> Path:
    """``$ALLOY_CACHE/camera`` (``ALLOY_CACHE`` defaults to ``~/.cache/alloy``)."""
    return Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")) / "camera"


def csrc_hash() -> str:
    """First 12 hex digits of the SHA-256 over the names and bytes of the files in ``csrc/``."""
    h = hashlib.sha256()
    for f in sorted(CSRC.iterdir()):
        if f.is_file():
            h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def build_dir() -> Path:
    """Where the built library and device programs for the current ``csrc/`` live."""
    return cache_root() / "optix" / csrc_hash()


def built_files() -> tuple[Path, Path]:
    """(libowf.so, owf_programs.optixir) of the current ``csrc/``; they exist after :func:`build`."""
    d = build_dir()
    return d / LIB_NAME, d / PROGRAMS_NAME


def is_built() -> bool:
    """True when both build outputs exist."""
    return all(f.is_file() for f in built_files())


def licence_accepted(flag: bool = False) -> bool:
    """True when the caller passed the flag or ``ALLOY_ACCEPT_NVIDIA_OPTIX_LICENSE=1`` is set."""
    return bool(flag) or os.environ.get(LICENCE_ENV, "") == "1"


def sdk_dir() -> Path:
    """Directory that holds ``include/optix.h``: ``OPTIX_SDK`` if set, else the cache clone."""
    env = os.environ.get("OPTIX_SDK")
    return Path(env).expanduser() if env else cache_root() / f"optix-dev-{OPTIX_TAG.lstrip('v')}"


def _fetch_headers(sdk: Path) -> None:
    """Clone the optix-dev headers into ``sdk`` (only after the licence was accepted)."""
    if (sdk / "include" / "optix.h").is_file():
        return
    if shutil.which("git") is None:
        raise CameraError("git is not installed, so the OptiX headers cannot be cloned",
                          f"install git, or download {OPTIX_REPO} (tag {OPTIX_TAG}) and set OPTIX_SDK to it")
    sdk.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s (%s) into %s", OPTIX_REPO, OPTIX_TAG, sdk)
    cmd = ["git", "clone", "-q", "--depth", "1", "-b", OPTIX_TAG, OPTIX_REPO, str(sdk)]
    if subprocess.run(cmd, capture_output=True, text=True).returncode != 0 or not (sdk / "include" / "optix.h").is_file():
        raise CameraError(f"cloning {OPTIX_REPO} failed",
                          f"check the network, or download tag {OPTIX_TAG} yourself and set OPTIX_SDK to its folder")


def find_compilers() -> tuple[str, Path]:
    """(host C++ compiler, nvcc path); raises CameraError naming the missing tool and what to set."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("g++-12")
    if not cxx:
        raise CameraError("no C++ compiler found (g++ or g++-12)", "install one with your package manager, or set CXX")
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = Path(home) / "bin" / "nvcc" if home else None
    nvcc = nvcc if nvcc and nvcc.is_file() else (Path(p) if (p := shutil.which("nvcc")) else None)
    if nvcc is None:
        raise CameraError("nvcc (CUDA toolkit) not found", "install the CUDA toolkit and set CUDA_HOME, or add nvcc to PATH")
    return cxx, nvcc


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=CSRC)
    if r.returncode != 0:
        raise CameraError(f"compiling {what} failed:\n{r.stderr[-2000:]}", "check the compiler output above")


def compile_all(out: Path, sdk: Path, cxx: str, nvcc: Path) -> None:
    """Compile libowf.so and owf_programs.optixir into the existing directory ``out``."""
    inc_sdk, inc_cuda = f"-I{sdk / 'include'}", f"-I{nvcc.parent.parent / 'include'}"
    _run([cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", inc_sdk, inc_cuda, "owf.cpp", "-o",
          str(out / LIB_NAME), "-lcuda", "-ldl", "-lpthread"], "libowf.so")
    _run([str(nvcc), "-ccbin", cxx, "-optix-ir", "-arch=compute_75", "-std=c++17", "--use_fast_math", "-O3", inc_sdk,
          "owf_programs.cu", "-o", str(out / PROGRAMS_NAME)], "owf_programs.optixir")


def build(accept_licence: bool = False, force: bool = False) -> Path:
    """Build (or find) the OptiX helper for the current ``csrc/`` and return its directory.

    Example::

        d = build(accept_licence=True)
        assert (d / "libowf.so").is_file()
    """
    if is_built() and not force:
        return build_dir()
    sdk = sdk_dir()
    if not (sdk / "include" / "optix.h").is_file() and not licence_accepted(accept_licence):
        raise CameraError("the NVIDIA OptiX headers are needed and are not under the Apache-2.0 licence\n" + LICENCE_NOTICE,
                          "pass --accept-nvidia-optix-license or set " + LICENCE_ENV + "=1, or set OPTIX_SDK to headers you fetched")
    _fetch_headers(sdk)
    cxx, nvcc = find_compilers()
    final = build_dir()
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="build-", dir=final.parent))
    try:
        compile_all(tmp, sdk, cxx, nvcc)
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("built the OptiX helper in %s", final)
    return final
