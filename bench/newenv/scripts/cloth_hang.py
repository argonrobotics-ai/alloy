# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E11 (cloth held by two adjacent corners): slide both corners sideways, then lift them."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_move", "seconds": 1.0, "reset": True},
          {"name": "act_lift", "seconds": 1.0, "reset": False}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Both held corners at full speed: +x during "act_move", then +z during "act_lift"; the 6-D action is the
    two corners' velocities (x, y, z each)."""
    out = np.zeros((1, act_dim))
    axis = {"act_move": 0, "act_lift": 2}.get(phase)
    if axis is not None:
        out[:, axis] = 1.0
        out[:, 3 + axis] = 1.0
    return out
