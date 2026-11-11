# SPDX-License-Identifier: Apache-2.0
"""Provenance of a bench result: code versions, device and run settings, all as strings (`LawResult.provenance`)."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections.abc import Mapping
from functools import lru_cache
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def _newton_commit() -> str:
    """Newton is vendored (alloy/_vendor): its commit is the pin."""
    from alloy.core.newton_compat import NEWTON_PIN  # noqa: PLC0415

    return NEWTON_PIN


def _vendored_version(name: str) -> str:
    from alloy._vendor import VERSIONS  # noqa: PLC0415

    return VERSIONS[name]


@lru_cache(maxsize=1)
def code_versions() -> dict[str, str]:
    """Alloy commit (-dirty if tracked files under alloy/ or bench/ changed; `ALLOY_COMMIT` env where there is no git
    checkout, e.g. the rsynced GPU worktree), the vendored Newton (version + commit) and MuJoCo Warp, MuJoCo, Warp,
    NumPy, SciPy, Python, host."""
    sha = _git("rev-parse", "--short=10", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "alloy", "bench"))
    alloy = f"{sha}{'-dirty' if dirty else ''}" if sha else os.environ.get("ALLOY_COMMIT", "")  # no .git on GPU VMs
    return {
        "alloy": alloy,
        "newton": _vendored_version("newton"),
        "newton_commit": _newton_commit(),
        "mujoco_warp": _vendored_version("mujoco_warp"),
        "mujoco": _version("mujoco"),
        "warp": _version("warp-lang"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def device_name(device: str) -> str:
    try:
        import warp as wp

        return str(wp.get_device(device).name)
    except Exception:  # noqa: BLE001 - provenance is best effort; the device string is kept regardless
        return ""


def commit_short() -> str:
    """Seven-character Alloy commit for result paths (``ALLOY_COMMIT`` outside git, else ``unknown``)."""
    return _git("rev-parse", "--short=7", "HEAD") or os.environ.get("ALLOY_COMMIT", "")[:7] or "unknown"


def provenance(device: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    out = dict(code_versions())
    out.update({"device": device, "device_name": device_name(device),
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if extra:
        out.update({k: str(v) for k, v in extra.items()})
    return out


__all__ = ["provenance", "code_versions", "commit_short", "device_name", "REPO"]
