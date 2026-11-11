# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E16 (tank slosh): zero action at rest, then a constant 2 m/s2 for 0.5 s, then zero."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_slosh", "seconds": 0.5, "reset": True}]

A_MAX = 3.0                 # the request's maximum acceleration (m/s2)
A_CMD = 2.0                 # commanded acceleration (m/s2)
T_ACCEL = 0.5               # seconds of acceleration


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+2/3 (= 2 m/s2 of the 3 m/s2 maximum) for the first 0.5 s, then zero."""
    if phase != "act_slosh" or t >= T_ACCEL - 1e-9:
        return np.zeros((1, act_dim))
    return np.full((1, act_dim), A_CMD / A_MAX)
