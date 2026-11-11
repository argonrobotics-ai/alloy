# SPDX-License-Identifier: Apache-2.0
"""E04 reference (Alloy): a 200 g puck (6 x 6 x 2 cm) on a 15 degree, 1.5 m incline; friction 0.1-0.6 per world.

Action (N, 1) in [-1, 1] = force along the slope, +1 = 1 N up-slope. Up-slope is +x (and +z).
Alloy combines two objects' friction by their harmonic mean, so the incline gets a very large coefficient and the
puck's draw is half the wanted pair value: pair = 2 p M / (p + M), about 2 p. The probe reports that pair value from
the engine's per-world material table (there is no public read-back of ``scene.randomize`` draws).
"""
from __future__ import annotations

import math

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
ANGLE = math.radians(15.0)
LENGTH, WIDTH, SLAB = 1.5, 0.4, 0.02
PUCK = (0.06, 0.06, 0.02)
MASS = 0.2
FORCE_N = 1.0
MU_LO, MU_HI = 0.1, 0.6
MU_INCLINE = 100.0
CENTRE = np.array([0.0, 0.0, 0.5])
UP = np.array([math.cos(ANGLE), 0.0, math.sin(ANGLE)])          # unit vector up the slope
NORMAL = np.array([-math.sin(ANGLE), 0.0, math.cos(ANGLE)])
START = CENTRE + NORMAL * (SLAB / 2 + PUCK[2] / 2)               # the middle of the incline


def _pair_mu(sim: alloy.Sim, worlds: int) -> np.ndarray:
    """The puck-incline coefficient per world: harmonic mean of the puck's draw and the incline's value."""
    table = sim.engine.model.shape_material_mu.numpy().reshape(worlds, -1)
    varying = np.flatnonzero(np.ptp(table, axis=0) > 0)
    if worlds > 1 and len(varying) != 1:
        raise RuntimeError(f"expected one randomized friction column, found {len(varying)}")
    p = table[:, varying[0]] if len(varying) else np.full(worlds, 0.5 * (MU_LO + MU_HI) / 2)
    return (2.0 * p * MU_INCLINE / (p + MU_INCLINE)).astype(np.float32)


class InclineEnv:
    obs_dim, act_dim, dt = 3, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_box("incline", size=(LENGTH, WIDTH, SLAB), pos=tuple(CENTRE), rpy=(0.0, -ANGLE, 0.0), static=True,
                      friction=MU_INCLINE)
        scene.add_box("puck", size=PUCK, pos=tuple(START), rpy=(0.0, -ANGLE, 0.0), mass=MASS, friction=0.15)
        scene.randomize("puck", "friction", MU_LO / 2, MU_HI / 2)
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._mu = _pair_mu(self._sim, num_envs)
        self._k = 0
        self._fresh = True

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()
        self._fresh = False
        self._k = 0
        return self._obs()

    def _along(self) -> tuple[np.ndarray, np.ndarray]:
        s = self._sim.state(["puck"])
        return (s["puck.pos"] - START) @ UP, s["puck.vel"][:, :3] @ UP

    def _obs(self) -> np.ndarray:
        pos, vel = self._along()
        return np.stack([pos, vel, self._mu], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, 1), -1.0, 1.0)
        self._sim.apply_wrench("puck", force=(a * FORCE_N * UP[None, :]).astype(np.float32))
        self._sim.step(1)
        self._k += 1
        reward = (-np.abs(self._along()[0])).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"puck_pos": self._sim.state(["puck"])["puck.pos"].astype(np.float32), "friction": self._mu.copy()}

    def success(self) -> np.ndarray:
        pos, vel = self._along()
        return (np.abs(pos) < 0.05) & (np.abs(vel) < 0.05)


def make_env(num_envs: int, seed: int) -> InclineEnv:
    return InclineEnv(num_envs, seed)
