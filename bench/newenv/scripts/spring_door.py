# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E08 (spring door).

Phases (each from a fresh reset unless it says otherwise):
  act_half  4 s  constant action 0.5
  act_full  4 s  constant action 1.0
  act_open  2.5 s  a feedback law on the probe values drives the door to 1.0 rad and holds it there
  release   2 s  (continues from act_open's last state) zero action: the door swings shut
The feedback law is a PD on door_angle / door_rate with the spring's torque (2 N m/rad x angle) added as feed-forward;
the action is that torque / 3 N m, clipped to [-1, 1].
"""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_half", "seconds": 4.0, "reset": True},
          {"name": "act_full", "seconds": 4.0, "reset": True},
          {"name": "act_open", "seconds": 2.5, "reset": True},
          {"name": "release", "seconds": 2.0, "reset": False}]

TARGET, SPRING, MAX_TORQUE, KP, KD = 1.0, 2.0, 3.0, 20.0, 8.0


def _open_law(probe: dict[str, np.ndarray]) -> np.ndarray:
    angle, rate = probe["door_angle"], probe["door_rate"]
    torque = SPRING * angle + KP * (TARGET - angle) - KD * rate
    return np.clip(torque / MAX_TORQUE, -1.0, 1.0).reshape(-1, 1)


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    if phase == "act_half":
        return np.full((1, act_dim), 0.5)
    if phase == "act_full":
        return np.full((1, act_dim), 1.0)
    if phase == "act_open":
        return _open_law(probe)
    return np.zeros((1, act_dim))
