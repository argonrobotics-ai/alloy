# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E17 (tap and glass): zero flow at rest, then full flow for 3 s, then zero for 3 s."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_fill", "seconds": 6.0, "reset": True}]

T_OPEN = 3.0                # seconds of full flow


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 (the full 50 ml/s) for the first 3 s, then 0."""
    if phase != "act_fill" or t >= T_OPEN - 1e-9:
        return np.zeros((1, act_dim))
    return np.ones((1, act_dim))
