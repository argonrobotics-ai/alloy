# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E20 (arm push): +1 on joint 1, then -1 on joint 4, each for 10 steps."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_joint", "seconds": 1.0, "reset": True},
          {"name": "act_joint4", "seconds": 1.0, "reset": True}]
JOINT_STEPS = 10
_calls = {"n": 0}


def _column(phase: str) -> tuple[int, float]:
    """(action column, sign) driven in a phase."""
    return (0, 1.0) if phase == "act_joint" else (3, -1.0)


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Full action on one joint for the phase's first 10 calls (the call count restarts when t == 0), zeros after."""
    _calls["n"] = 0 if t == 0.0 else _calls["n"] + 1
    col, sign = _column(phase)
    out = np.zeros((1, act_dim))
    out[:, col] = sign if _calls["n"] < JOINT_STEPS else 0.0
    return out
