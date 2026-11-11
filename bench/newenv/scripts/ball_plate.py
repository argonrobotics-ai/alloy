# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E02 (ball on a tilting plate).

"act_tilt": pitch rate +1 until the plate's pitch reaches 0.1 rad, then zero (hold). "act_clamp": roll rate +1 throughout.
"""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_tilt", "seconds": 0.55, "reset": True},
          {"name": "act_clamp", "seconds": 1.4, "reset": True}]
TILT_TARGET = 0.1


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """act = (roll rate, pitch rate); the tilt law reads the plate's pitch from the probe."""
    n = probe["plate_rpy"].shape[0]
    out = np.zeros((n, act_dim))
    if phase == "act_tilt":
        out[:, 1] = np.where(probe["plate_rpy"][:, 1] < TILT_TARGET, 1.0, 0.0)
    elif phase == "act_clamp":
        out[:, 0] = 1.0
    return out
