# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E21 (arm drawer): +1 on joint 1 for 10 steps, then gripper close and open."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_joint", "seconds": 1.0, "reset": True},
          {"name": "act_gripper", "seconds": 3.0, "reset": True}]
JOINT_STEPS = 10
CLOSE_S = 1.5
_calls = {"n": 0}


def _joint_action(t: float, act_dim: int) -> np.ndarray:
    """+1 on joint 1 for the first 10 calls of the phase (the step count is the phase's own call count)."""
    _calls["n"] = 0 if t == 0.0 else _calls["n"] + 1
    out = np.zeros((1, act_dim))
    out[:, 0] = 1.0 if _calls["n"] < JOINT_STEPS else 0.0
    return out


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Joint 1 pushed for 10 steps in "act_joint"; the last column (gripper) -1 then +1 in "act_gripper"."""
    if phase == "act_joint":
        return _joint_action(t, act_dim)
    out = np.zeros((1, act_dim))
    if phase == "act_gripper":
        out[:, -1] = -1.0 if t < CLOSE_S else 1.0
    return out
