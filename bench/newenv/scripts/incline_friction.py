# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E04 (puck on an incline): after the zero-action rest, a constant +1 force for 0.5 s."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_up", "seconds": 0.5, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 (the full 1 N along the slope) during "act_up"."""
    out = np.zeros((1, act_dim))
    if phase == "act_up":
        out[:] = 1.0
    return out
