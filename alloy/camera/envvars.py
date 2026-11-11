# SPDX-License-Identifier: Apache-2.0
"""Environment knobs of the camera package (``ALLOY_CAMERA_*``): typed readers that fall back to the default on an
unset or empty variable and say which variable is malformed.

Example::

    from alloy.camera.envvars import env_int
    group = env_int("ALLOY_CAMERA_IAS_GROUP", 8)
"""
from __future__ import annotations

import os

from .errors import CameraError


def env_str(name: str, default: str) -> str:
    """The variable's value, or ``default`` when it is unset or empty."""
    return os.environ.get(name) or default


def env_int(name: str, default: int) -> int:
    """The variable as an int, or ``default`` when unset; CameraError on a value that is not an integer."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise CameraError(f"{name}={raw!r} is not an integer", f"set {name} to a whole number or unset it") from exc


def env_flag(name: str, default: bool) -> bool:
    """The variable as a switch ("1"/"true"/"yes"/"on" = True, "0"/"false"/"no"/"off" = False)."""
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise CameraError(f"{name}={raw!r} is not a switch", f"set {name} to 1 or 0, or unset it")
