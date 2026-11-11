# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E03 (bin and objects): three more seconds at rest, then the bin driven +x, then -y."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "rest_more", "seconds": 1.0, "reset": False},
          {"name": "act_x", "seconds": 0.7, "reset": False},
          {"name": "act_y", "seconds": 0.7, "reset": False}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 on the first action axis in "act_x", -1 on the second in "act_y", zeros otherwise."""
    out = np.zeros((1, act_dim))
    if phase == "act_x":
        out[:, 0] = 1.0
    elif phase == "act_y":
        out[:, 1] = -1.0
    return out
