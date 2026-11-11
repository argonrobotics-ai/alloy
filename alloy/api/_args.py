# SPDX-License-Identifier: Apache-2.0
"""Argument checking and conversion for ``alloy.Scene`` (no engine imports). Every error says what to pass instead."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.api.errors import SceneError

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def check_name(name: Any, taken: Sequence[str]) -> str:
    """A new object name: a letter, then letters, digits, '_' or '-', not used yet. Example: ``"cube_1"``."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SceneError(f"bad name {name!r}", "use a name that starts with a letter and has only letters, digits, "
                         "'_' or '-', for example 'cube_1'", field="name")
    if name in taken:
        raise SceneError(f"name {name!r} is already in the scene", f"pick another name (taken: {', '.join(taken)})",
                         field="name")
    return name


def positive(value: Any, field: str, unit: str) -> float:
    """A finite number > 0. Example: ``positive(0.1, "radius", "m")``."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise SceneError(f"{field} must be a number, got {value!r}", f"pass {field}=<number in {unit}>",
                         field=field) from None
    if not math.isfinite(x) or x <= 0.0:
        raise SceneError(f"{field} must be > 0, got {value!r}", f"pass a positive {field} in {unit}", field=field)
    return x


def vector(value: Any, n: int, field: str, unit: str = "m") -> list[float]:
    """``n`` finite numbers. Example: ``vector((0, 0, 1), 3, "pos")``."""
    arr = np.asarray(value, dtype=float) if value is not None else None
    if arr is None or arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise SceneError(f"{field} needs {n} finite numbers, got {value!r}",
                         f"pass {field}=({', '.join(['...'] * n)}) in {unit}", field=field)
    return [float(x) for x in arr]


def sizes(value: Any, n: int, field: str) -> list[float]:
    """``n`` positive lengths in metres. Example: ``sizes((0.1, 0.1, 0.1), 3, "size")``."""
    out = vector(value, n, field)
    if min(out) <= 0.0:
        raise SceneError(f"{field} must be > 0 in every axis, got {value!r}", f"pass positive {field} in metres",
                         field=field)
    return out


def rpy_to_xyzw(rpy: Sequence[float]) -> list[float]:
    """Quaternion (x, y, z, w) of roll-pitch-yaw in radians (extrinsic x, then y, then z). Example: ``(0, 0, 1.57)``."""
    r, p, y = (0.5 * float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy]


def orientation(quat_xyzw: Any = None, quat_wxyz: Any = None, rpy: Any = None) -> list[float]:
    """One orientation as a unit quaternion (x, y, z, w) from exactly one of the three forms (or identity).

    Example: ``orientation(quat_wxyz=(1, 0, 0, 0)) == [0, 0, 0, 1]``."""
    given = [k for k, v in (("quat_xyzw", quat_xyzw), ("quat_wxyz", quat_wxyz), ("rpy", rpy)) if v is not None]
    if len(given) > 1:
        raise SceneError(f"orientation given twice ({', '.join(given)})", "pass only one of quat_xyzw, quat_wxyz, rpy",
                         field=given[1])
    if quat_xyzw is not None:
        q = vector(quat_xyzw, 4, "quat_xyzw", "(x, y, z, w)")
    elif quat_wxyz is not None:
        w, x, y, z = vector(quat_wxyz, 4, "quat_wxyz", "(w, x, y, z)")
        q = [x, y, z, w]
    elif rpy is not None:
        q = rpy_to_xyzw(vector(rpy, 3, "rpy", "radians"))
    else:
        return list(IDENTITY_XYZW)
    norm = math.sqrt(sum(c * c for c in q))
    if norm < 1e-9:
        raise SceneError("quaternion has zero length", "pass a unit quaternion, e.g. quat_wxyz=(1, 0, 0, 0)",
                         field=given[0])
    return [c / norm for c in q]


def positions(pos: Any, worlds: int) -> tuple[list[float], list[list[float]] | None]:
    """(position for world 0, per-world positions or None) from a (3,) or (worlds, 3) array-like.

    Example: ``positions([[0, 0, 1], [0.1, 0, 1]], 2)`` -> ``([0, 0, 1], [[0, 0, 1], [0.1, 0, 1]])``."""
    arr = np.asarray(pos, dtype=float)
    if arr.shape == (3,):
        return vector(arr, 3, "pos"), None
    if arr.ndim == 2 and arr.shape[1] == 3 and np.all(np.isfinite(arr)):
        if arr.shape[0] != worlds:
            raise SceneError(f"pos has {arr.shape[0]} rows but the scene has worlds={worlds}",
                             f"create the scene with alloy.Scene(worlds={arr.shape[0]}) or pass {worlds} rows",
                             field="pos")
        return [float(x) for x in arr[0]], arr.tolist()
    raise SceneError(f"pos must be (3,) or (worlds, 3), got shape {arr.shape}",
                     "pass pos=(x, y, z) for every world, or one row per world", field="pos")


def one_of(value: Any, allowed: Sequence[str], field: str) -> str:
    """``value`` if it is in ``allowed``. Example: ``one_of("sdf", ("convex", "sdf"), "collision")``."""
    if value not in allowed:
        raise SceneError(f"{field} must be one of {list(allowed)}, got {value!r}", f"pass {field}={allowed[0]!r}",
                         field=field)
    return str(value)


def user_q(value: float, note: str) -> dict[str, Any]:
    """A card quantity set by the user (source "model"). Example: ``user_q(1000.0, "density from mass")``."""
    return {"value": float(value), "source": "model", "note": note}


def slug(text: str) -> str:
    """A card-file-safe lower-case name. Example: ``slug("Cube-1") == "cube_1"``."""
    return re.sub(r"[^a-z0-9_]", "_", text.lower())
