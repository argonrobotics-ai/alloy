# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E07 (double pendulum): zero torque at rest, then a constant full +1 elbow torque for 0.3 s (the elbow spins up to ~100 rad/s in 1 s)."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_torque", "seconds": 0.3, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """+1 on the elbow torque during "act_torque"; zero otherwise."""
    return np.full((1, act_dim), 1.0 if phase == "act_torque" else 0.0)
