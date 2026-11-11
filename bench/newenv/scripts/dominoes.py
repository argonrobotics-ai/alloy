# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E05 (dominoes): the full +1 push for 0.5 s, then zero, over 4.5 s from a fresh start."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_push", "seconds": 4.5, "reset": True}]
PUSH_S = 0.5


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 until PUSH_S seconds into "act_push", zero afterwards."""
    out = np.zeros((1, act_dim))
    if phase == "act_push" and t < PUSH_S - 1e-9:
        out[:] = 1.0
    return out
