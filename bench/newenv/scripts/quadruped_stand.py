# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E09 (Go2 stand): zero offsets for 3 s, then +1 on every joint for 1 s."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "stand", "seconds": 3.0, "reset": True},
          {"name": "act_plus", "seconds": 1.0, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 on every joint's target offset during "act_plus"; zero otherwise."""
    return np.full((1, act_dim), 1.0 if phase == "act_plus" else 0.0)
