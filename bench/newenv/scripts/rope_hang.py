# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E14 (rope held by its free end): a straight lift, then a feedback move to a point."""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_speed", "seconds": 1.0, "reset": True},
          {"name": "act_raise", "seconds": 5.0, "reset": True}]

SPAN_M = 0.6      # horizontal distance from the anchor to the raised end
GAIN = 10.0       # action per metre of error: full speed beyond 0.1 m, then a smooth stop


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """"act_speed": full +z. "act_raise": steer the free end to the anchor's height, SPAN_M away along +x."""
    if phase == "act_speed":
        out = np.zeros((1, act_dim))
        out[:, 2] = 1.0
        return out
    if phase != "act_raise":
        return np.zeros((1, act_dim))
    pts = probe["rope_points"]
    d_first = np.linalg.norm(pts[:, 0] - probe["end_pos"], axis=-1)
    far_is_first = d_first > np.linalg.norm(pts[:, -1] - probe["end_pos"], axis=-1)
    anchor = np.where(far_is_first[:, None], pts[:, 0], pts[:, -1])      # the rope end that is not the held one
    goal = anchor + np.array([SPAN_M, 0.0, 0.0])
    return np.clip(GAIN * (goal - probe["end_pos"]), -1.0, 1.0)
