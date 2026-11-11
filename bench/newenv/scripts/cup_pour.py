# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E15 (cup pour): zero tilt rate at rest, then tilt at 0.5 rad/s up to 2.2 rad, then hold."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_pour", "seconds": 8.0, "reset": True}]


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """Half the maximum tilt rate until the cup has tipped 2.2 rad, then zero."""
    if phase != "act_pour":
        return np.zeros((1, act_dim))
    tilt = np.asarray(probe["cup_tilt"], np.float64).reshape(-1)
    return np.where(tilt < 2.2, 0.5, 0.0).reshape(-1, 1)
