# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E10 (wheeled cart): zero torque at rest, equal +1 torques for 1 s, then opposite torques.

Both phases start from a fresh reset. "act_forward": left and right both +1 (action columns 0 and 1).
"act_spin": left +1, right -1 (the cart turns in place).
"""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_forward", "seconds": 1.0, "reset": True},
          {"name": "act_spin", "seconds": 1.0, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    out = np.zeros((1, act_dim))
    if phase == "act_forward":
        out[:] = 1.0
    elif phase == "act_spin":
        out[0, 0], out[0, 1:] = 1.0, -1.0
    return out
