# SPDX-License-Identifier: Apache-2.0
"""Action laws for the fake cart-pole (harness test double; the real one is scripts/cartpole.py)."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act", "seconds": 5.0, "reset": True}]
GAINS = {"cart_x": 1.0, "pole_angle": 30.0, "cart_v": 2.0, "pole_rate": 6.0}


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    force = sum(gain * probe[key] for key, gain in GAINS.items())
    return np.clip(force / 10.0, -1.0, 1.0)[:, None]
