# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E13 (soft cube under a plate): press down 1 s, lift 1 s, then hold still."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_down", "seconds": 1.0, "reset": False},
          {"name": "act_up", "seconds": 1.0, "reset": False},
          {"name": "act_settle", "seconds": 0.5, "reset": False}]

_SIGN = {"act_down": -1.0, "act_up": 1.0}


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Full speed down, full speed up, then zero."""
    return np.full((1, act_dim), _SIGN.get(phase, 0.0))
