# SPDX-License-Identifier: Apache-2.0
"""E13 reference (Alloy): a 10 cm silicone cube (E 100 kPa, 1,100 kg/m3) on the ground under a 1.2 kg rigid plate.

The plate is a one-body robot on a vertical slide joint with a stiff position servo; the 1-D action sets its speed
(1 = 2 cm/s down is -1). There is no contact-force read-out in the public API, so ``plate_force`` is the force the cube
exerts on the plate, from the servo's own law (kp x error - kv x velocity, a joint-effort reading) and the plate's weight.
``soft_points`` are the cube's nodes moved out by the contact radius (the surface the contact sees).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
SIDE = 0.1
RADIUS = 0.0025              # the soft body's contact radius: nodes sit this far inside the surface
SPEED = 0.02                 # m/s at action 1
PLATE_M = 1.2
PLATE_T = 0.02
GAP = 0.0005                 # plate bottom above the cube top at the start
KP, KV, FMAX = 4e5, 3000.0, 800.0
TARGET_H = 0.08
TOL_H = 0.003
Z_MIN, Z_MAX = 0.07, 0.14    # plate bottom travel: at most 30 % strain below, 4 cm above
ROOT = Path(__file__).parent / "cards"

PLATE = f"""<mujoco><compiler angle="radian"/><worldbody><body name="plate" pos="0 0 {SIDE + GAP + PLATE_T / 2}">
 <joint name="slide" type="slide" axis="0 0 1" range="-0.2 0.2" damping="0"/>
 <geom type="box" size="{SIDE} {SIDE} {PLATE_T / 2}" mass="{PLATE_M}" friction="0.8"/></body></worldbody>
 <actuator><position joint="slide" kp="{KP}" kv="{KV}" forcerange="-{FMAX} {FMAX}"/></actuator></mujoco>"""


class SoftPressEnv:
    obs_dim, act_dim, dt = 5, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(dt=DT, worlds=num_envs, roots=[str(ROOT)])
        scene.add_ground()
        scene.add_box("cube", size=(SIDE,) * 3, pos=(0, 0, SIDE / 2), material="silicone_100kpa", rigid=False)
        scene.add_robot("press", mjcf=PLATE)
        self._sim = _build(scene, seed)
        self._z0 = SIDE + GAP + PLATE_T / 2                      # plate centre at the start
        self._rest = self._sim.state(["cube"])["cube.points"].copy()
        self._outward = self._outward_shift(self._rest[0])
        self._target = np.zeros(num_envs)
        self._force = np.zeros(num_envs)
        self._k = 0
        self._fresh = True

    @staticmethod
    def _outward_shift(rest: np.ndarray) -> np.ndarray:
        """(V, 3) per-node shift: +/- RADIUS on every axis where the node lies on the cube's outer face."""
        lo, hi = rest.min(axis=0), rest.max(axis=0)
        return RADIUS * ((rest >= hi - 1e-6).astype(np.float32) - (rest <= lo + 1e-6).astype(np.float32))

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()
        self._fresh = False
        self._k = 0
        self._target[:] = 0.0
        self._sim.set_targets({"press/slide": self._target})
        self._force[:] = 0.0
        return self._obs()

    def _points(self) -> np.ndarray:
        return self._sim.state(["cube"])["cube.points"] + self._outward[None]

    def _plate_z(self) -> np.ndarray:
        return self._z0 + self._sim.state(["press/slide"])["press/slide.q"].reshape(self.num_envs)

    def _height(self) -> np.ndarray:
        p = self._points()
        return p[..., 2].max(axis=1) - p[..., 2].min(axis=1)

    def _obs(self) -> np.ndarray:
        z, h = self._plate_z(), self._height()
        return np.stack([z, self._force, h, h - TARGET_H, self._target], axis=1).astype(np.float32)

    def _servo_force(self, z: np.ndarray) -> np.ndarray:
        """Force of the cube on the plate (N, up): plate weight - servo effort (the plate is quasi-static)."""
        vel = self._sim.state(["press/plate"])["press/plate.vel"][:, 2]
        effort = np.clip(KP * (self._target - (z - self._z0)) - KV * vel, -FMAX, FMAX)
        return PLATE_M * 9.81 - effort

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs), -1.0, 1.0)
        lo, hi = Z_MIN - (self._z0 - PLATE_T / 2), Z_MAX - (self._z0 - PLATE_T / 2)
        self._target = np.clip(self._target + a * SPEED * DT, lo, hi)
        self._sim.set_targets({"press/slide": self._target})
        self._sim.step(1)
        self._k += 1
        self._force = np.abs(self._servo_force(self._plate_z()))
        h = self._height()
        reward = (-np.abs(h - TARGET_H) / SIDE + (np.abs(h - TARGET_H) < TOL_H)).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"soft_points": self._points().astype(np.float32), "plate_z": self._plate_z().astype(np.float32),
                "plate_force": self._force.astype(np.float32)}

    def success(self) -> np.ndarray:
        return np.abs(self._height() - TARGET_H) < TOL_H


def _build(scene: alloy.Scene, seed: int) -> alloy.Sim:
    """One L4 when there is one; else the CPU, one process per core for many worlds (AGENTS.md section 1)."""
    import warp as wp  # noqa: PLC0415

    if wp.get_cuda_device_count() > 0:
        return alloy.build(scene, seed=seed)
    return alloy.build(scene, seed=seed, device="cpu", processes="auto" if scene.worlds > 1 else None)


def make_env(num_envs: int, seed: int) -> SoftPressEnv:
    return SoftPressEnv(num_envs, seed)
