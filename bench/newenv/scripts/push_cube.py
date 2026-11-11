# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E01 (push cube): zero actions at rest, then a constant +x push for 0.2 s."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_push", "seconds": 0.2, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Full +x force on the cube during "act_push"; nothing otherwise."""
    out = np.zeros((1, act_dim))
    if phase == "act_push":
        out[:, 0] = 1.0
    return out
