# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E18 (sand scoop): zero action at rest, then lower the scoop 3 cm below the bed's top, push it
15 cm along +x (or -x, toward the bed's middle), and hold still for 2 s.

Feedback on the probe values only: the bed's top is the 99th percentile height of ``sand_points``, the bed's middle the
mean x of the points, the scoop's place ``scoop_pos``. The law keeps its stage in module state, reset at t = 0.
"""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_dig", "seconds": 8.0, "reset": True}]

DEPTH, TRAVEL = 0.03, 0.15                  # m below the bed's top, m along the push
_STATE: dict[str, float] = {}


def _bed(probe: dict[str, np.ndarray]) -> tuple[float, float]:
    pts = np.asarray(probe["sand_points"], np.float64)[0]
    pts = pts[np.isfinite(pts).all(axis=-1)]
    return float(np.percentile(pts[:, 2], 99)), float(pts[:, 0].mean())


def _first_call(t: float, probe: dict[str, np.ndarray]) -> None:
    top, mid = _bed(probe)
    scoop = np.asarray(probe["scoop_pos"], np.float64)[0]
    _STATE.clear()
    _STATE.update(stage=0.0, top=top, sign=1.0 if scoop[0] < mid else -1.0)


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """4-D action (vx, vy, vz, pitch rate) in [-1, 1]; same for every world."""
    out = np.zeros((1, act_dim))
    if phase != "act_dig":
        return out
    if t <= 1e-9 or not _STATE:
        _first_call(t, probe)
    scoop = np.asarray(probe["scoop_pos"], np.float64)[0]
    if _STATE["stage"] == 0.0:
        if scoop[2] <= _STATE["top"] - DEPTH:
            _STATE.update(stage=1.0, x0=float(scoop[0]))
        else:
            out[0, 2] = -1.0
            return out
    if _STATE["stage"] == 1.0:
        if abs(scoop[0] - _STATE["x0"]) >= TRAVEL:
            _STATE["stage"] = 2.0
        else:
            out[0, 0] = _STATE["sign"]
    return out
