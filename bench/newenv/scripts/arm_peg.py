# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E22 (arm peg): +1 on joint 1 for 10 steps (the peg is held, there is no gripper action)."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_joint", "seconds": 1.0, "reset": True}]
JOINT_STEPS = 10
_calls = {"n": 0}


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 on joint 1 for the first 10 calls of the phase (the call count restarts when t == 0), zeros after."""
    _calls["n"] = 0 if t == 0.0 else _calls["n"] + 1
    out = np.zeros((1, act_dim))
    out[:, 0] = 1.0 if _calls["n"] < JOINT_STEPS else 0.0
    return out
