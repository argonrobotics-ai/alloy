# SPDX-License-Identifier: Apache-2.0
"""Shared helper of the liquid references (E16-E18): move one kinematic object with a velocity per world and period.

``sim.drive`` takes one velocity (and one angular velocity) for every world (alloy/api/_drive.py make_drive), so worlds
that command the same twist use one drive (walls move at that speed inside the period, which the liquid and the sand
see), and worlds that differ fall back to ``set_pose`` (a teleport of v * dt per period: no wall speed in the period).
"""
from __future__ import annotations

import numpy as np


def pitch_quat_wxyz(pitch: np.ndarray) -> np.ndarray:
    """(W, 4) wxyz quaternions of a rotation by ``pitch`` (W,) about +y."""
    return np.stack([np.cos(pitch / 2), np.zeros_like(pitch), np.sin(pitch / 2), np.zeros_like(pitch)], axis=1)


class KinematicMover:
    """Keeps a per-world position (and pitch about y) of kinematic body ``name``; moves it each control period."""

    def __init__(self, sim, name: str, start: np.ndarray) -> None:
        self.sim, self.name = sim, name
        self.start = np.asarray(start, np.float64).reshape(1, 3).repeat(sim.worlds, axis=0)
        self.pos = self.start.copy()
        self.pitch = np.zeros(sim.worlds)

    def reset(self) -> None:
        self.pos = self.start.copy()
        self.pitch = np.zeros(self.sim.worlds)

    def advance(self, velocity: np.ndarray, dt: float, pitch_rate: np.ndarray | float = 0.0) -> None:
        """Move every world by ``velocity`` (W, 3) m/s and turn it by ``pitch_rate`` (W,) rad/s over one period."""
        v = np.broadcast_to(np.asarray(velocity, np.float64).reshape(-1, 3), self.pos.shape)
        w = np.broadcast_to(np.asarray(pitch_rate, np.float64).reshape(-1), self.pitch.shape)
        before = self.pos.copy()
        self.pos = self.pos + v * dt
        self.pitch = self.pitch + w * dt
        if np.allclose(v, v[0], atol=1e-12) and np.allclose(w, w[0], atol=1e-12):
            self.sim.drive(self.name, velocity=tuple(v[0]), angular_velocity=(0.0, float(w[0]), 0.0),
                           pivot=tuple(before[0]), duration=dt)
            return
        self.sim.set_pose(self.name, pos=self.pos.astype(np.float32), quat_wxyz=pitch_quat_wxyz(self.pitch).astype(np.float32))
