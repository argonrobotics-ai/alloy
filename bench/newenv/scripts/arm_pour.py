# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E23 (arm pour): +1 on joint 1 for 10 steps, which turns the held cup with the arm."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_joint", "seconds": 1.0, "reset": True}]
JOINT_STEPS = 10
_calls = {"n": 0}


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Joint 1 pushed for the first 10 calls of "act_joint" (the phase's own call count); zero otherwise."""
    out = np.zeros((1, act_dim))
    if phase != "act_joint":
        return out
    _calls["n"] = 0 if t == 0.0 else _calls["n"] + 1
    out[:, 0] = 1.0 if _calls["n"] < JOINT_STEPS else 0.0
    return out
