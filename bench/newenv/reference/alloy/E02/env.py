# SPDX-License-Identifier: Apache-2.0
"""E02 reference (Alloy): a 3 cm, 100 g solid ball on a 40 cm square plate that tilts about its centre.

The plate is a two-hinge MJCF robot (pitch about y outside, roll about x inside, so ``plate_rpy = (roll, pitch, 0)`` in
the roll-pitch-yaw convention), both joints limited to +-0.3 rad. The action is a rate: the env integrates it into a
joint target (0.5 rad/s at +-1, clamped) and a stiff position servo follows. Done at ``episode_steps`` for all worlds.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
PLATE_Z = 0.3
PLATE_HALF = 0.2
THICK_HALF = 0.01
BALL_R = 0.03
BALL_MASS = 0.1
RATE = 0.5
TILT_MAX = 0.3
SPAWN_HALF = 0.15
FRICTION = 0.6
SUCCESS_M = 0.05

PLATE_MJCF = f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="gimbal" pos="0 0 {PLATE_Z}"><inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
  <joint name="pitch" type="hinge" axis="0 1 0" range="-{TILT_MAX} {TILT_MAX}"/>
  <body name="top"><inertial pos="0 0 0" mass="2" diaginertia="0.03 0.03 0.06"/>
    <joint name="roll" type="hinge" axis="1 0 0" range="-{TILT_MAX} {TILT_MAX}"/>
    <geom type="box" size="{PLATE_HALF} {PLATE_HALF} {THICK_HALF}" friction="{FRICTION}"/>
  </body></body></worldbody>
<actuator><position joint="pitch" kp="2000" kv="10"/><position joint="roll" kp="2000" kv="10"/></actuator></mujoco>"""


class BallPlateEnv:
    obs_dim, act_dim, dt = 8, 2, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._start = self._spawn()
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_robot("plate", mjcf=PLATE_MJCF, pos=(0.0, 0.0, 0.0))
        scene.add_sphere("ball", BALL_R, pos=self._start, mass=BALL_MASS, friction=FRICTION)
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._target = np.zeros((num_envs, 2))          # (roll, pitch) joint targets
        self._k = 0
        self._fresh = True

    def _spawn(self) -> np.ndarray:
        xy = self._rng.uniform(-SPAWN_HALF, SPAWN_HALF, (self.num_envs, 2))
        z = np.full((self.num_envs, 1), PLATE_Z + THICK_HALF + BALL_R + 0.002)
        return np.concatenate([xy, z], axis=1).astype(np.float32)

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._start = self._spawn()
            self._sim.reset()
            self._sim.set_pose("ball", pos=self._start, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        self._fresh = False
        self._target[:] = 0.0
        self._apply_targets()
        self._k = 0
        return self._obs()

    def _apply_targets(self) -> None:
        self._sim.set_targets({"roll": self._target[:, 0], "pitch": self._target[:, 1]})

    def _rpy(self) -> np.ndarray:
        s = self._sim.state(["plate/roll", "plate/pitch"])
        return np.stack([s["plate/roll.q"], s["plate/pitch.q"], np.zeros(self.num_envs)], axis=1).astype(np.float32)

    def _ball(self) -> dict[str, np.ndarray]:
        return self._sim.state(["ball"])

    def _obs(self) -> np.ndarray:
        b = self._ball()
        rel = b["ball.pos"][:, :2] - [0.0, 0.0]
        return np.concatenate([rel, b["ball.pos"][:, 2:], b["ball.vel"][:, :3], self._rpy()[:, :2]], axis=1).astype(
            np.float32)

    def _distance(self) -> np.ndarray:
        return np.linalg.norm(self._ball()["ball.pos"][:, :2], axis=1)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 2), -1.0, 1.0)
        self._target = np.clip(self._target + a * RATE * DT, -TILT_MAX, TILT_MAX)
        self._apply_targets()
        self._sim.step(1)
        self._k += 1
        reward = (-self._distance()).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"ball_pos": self._ball()["ball.pos"].astype(np.float32), "plate_rpy": self._rpy()}

    def success(self) -> np.ndarray:
        return self._distance() < SUCCESS_M


def make_env(num_envs: int, seed: int) -> BallPlateEnv:
    return BallPlateEnv(num_envs, seed)
