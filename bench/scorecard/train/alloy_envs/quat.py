# SPDX-License-Identifier: Apache-2.0
"""Quaternion helpers on the env backend (NumPy or torch), (W, 4) arrays in w, x, y, z order."""

from __future__ import annotations

import math
from typing import Any


def _lib(xp: Any) -> Any:
    """The array module behind a backend: torch for the torch backend, else NumPy."""
    if xp.name == "torch":
        return xp.torch
    import numpy as np  # noqa: PLC0415

    return np


def random_unit_quats(xp: Any, n: int) -> Any:
    """``n`` rotations drawn uniformly on SO(3) (Shoemake), (n, 4) wxyz. Example: ``random_unit_quats(xp, 8)``."""
    lib = _lib(xp)
    u = xp.uniform((n, 3), 0.0, 1.0)
    a, b = lib.sqrt(1.0 - u[:, 0]), lib.sqrt(u[:, 0])
    t1, t2 = 2.0 * math.pi * u[:, 1], 2.0 * math.pi * u[:, 2]
    return xp.stack([b * lib.cos(t2), a * lib.sin(t1), a * lib.cos(t1), b * lib.sin(t2)], -1)


def angle_between(xp: Any, q1: Any, q2: Any) -> Any:
    """Rotation angle (rad, 0..pi) taking ``q1`` to ``q2``, both (W, 4) unit quaternions.

    ``4 atan2(|q1 - q2|, |q1 + q2|)`` with ``q2`` sign-matched to ``q1``: exact at 0 in float32, where
    ``2 acos(dot)`` reads up to ~1e-3 rad for a dot one ulp under 1. Example: ``angle_between(xp, q, q) == 0``."""
    lib = _lib(xp)
    q2s = lib.where(((q1 * q2).sum(-1) < 0)[..., None], -q2, q2)
    diff = lib.sqrt(((q1 - q2s) ** 2).sum(-1))
    summ = lib.sqrt(((q1 + q2s) ** 2).sum(-1))
    return 4.0 * lib.arctan2(diff, summ)
