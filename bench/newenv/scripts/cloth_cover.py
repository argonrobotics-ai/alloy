# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E12 (cloth over a cube): one more second at rest, then slide the cube sideways."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "settle", "seconds": 1.0, "reset": False},
          {"name": "act_slide", "seconds": 1.0, "reset": False}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Nothing during "settle" (the cloth has then had 3 s with the rest phase); full +x during "act_slide"."""
    out = np.zeros((1, act_dim))
    if phase == "act_slide":
        out[:, 0] = 1.0
    return out
