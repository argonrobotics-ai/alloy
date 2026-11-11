# SPDX-License-Identifier: Apache-2.0
"""E03 reference (Alloy): ten objects (4 cm boxes, 2.5 cm radius spheres, 5 cm long cylinders; 50-200 g) drop into an
open bin 30 cm square and 15 cm deep standing on the ground.

The bin is a MJCF robot (floor + four walls on two slide joints, x and y, velocity actuators): action (N, 2) in [-1, 1] =
bin velocity, 0.2 m/s at 1. The shape mix (which of the ten slots is a box, sphere or cylinder) is drawn once from the
seed (Alloy builds one shape list for all worlds); drop spots and masses (density) differ per world.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
HALF, DEPTH, WALL, FLOOR = 0.15, 0.15, 0.01, 0.01
SPEED = 0.2
COUNT = 10
BOX, SPHERE_R, CYL_R, CYL_LEN = 0.04, 0.025, 0.015, 0.05
MASS_LO, MASS_HI = 0.05, 0.2
DROP_HALF = 0.08
DROP_Z0, DROP_DZ = 0.18, 0.055
FRICTION = 0.5
ROLL_DRAG = 1.0         # N s/m: Alloy has no rolling resistance (a sphere or cylinder rolls for ever); drag on the COM stands in
VOLUME = {"box": BOX ** 3, "sphere": 4.0 / 3.0 * np.pi * SPHERE_R ** 3, "cylinder": np.pi * CYL_R ** 2 * CYL_LEN}


def _bin_mjcf() -> str:
    w, hh = HALF + WALL, DEPTH / 2
    walls = "".join(f'<geom type="box" size="{sx} {sy} {hh}" pos="{px} {py} {hh}" friction="{FRICTION}"/>'
                    for sx, sy, px, py in [(WALL / 2, w, HALF + WALL / 2, 0), (WALL / 2, w, -HALF - WALL / 2, 0),
                                           (w, WALL / 2, 0, HALF + WALL / 2), (w, WALL / 2, 0, -HALF - WALL / 2)])
    return f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="frame"><inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
 <joint name="bx" type="slide" axis="1 0 0" range="-20 20"/>
 <body name="shell"><inertial pos="0 0 0.05" mass="8" diaginertia="0.2 0.2 0.3"/>
  <joint name="by" type="slide" axis="0 1 0" range="-20 20"/>
  <geom type="box" size="{w} {w} {FLOOR / 2}" pos="0 0 {FLOOR / 2}" friction="{FRICTION}"/>{walls}
 </body></body></worldbody>
<actuator><velocity joint="bx" kv="1000"/><velocity joint="by" kv="1000"/></actuator></mujoco>"""


class BinClutterEnv:
    obs_dim, act_dim, dt = COUNT * 3 + 3, 2, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._kinds = self._rng.choice(["box", "sphere", "cylinder"], COUNT)
        self._names = [f"o{i}" for i in range(COUNT)]
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground(friction=FRICTION)
        scene.add_robot("bin", mjcf=_bin_mjcf(), pos=(0.0, 0.0, 0.0))
        spots = self._spots()
        for i, (name, kind) in enumerate(zip(self._names, self._kinds)):
            self._add(scene, name, kind, spots[:, i])
            scene.randomize(name, "density", MASS_LO / VOLUME[kind], MASS_HI / VOLUME[kind])
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._k = 0
        self._fresh = True

    @staticmethod
    def _add(scene: alloy.Scene, name: str, kind: str, pos: np.ndarray) -> None:
        if kind == "box":
            scene.add_box(name, size=(BOX,) * 3, pos=pos, density=1000.0, friction=FRICTION, restitution=0.0)
        elif kind == "sphere":
            scene.add_sphere(name, SPHERE_R, pos=pos, density=1000.0, friction=FRICTION, restitution=0.0)
        else:
            scene.add_cylinder(name, CYL_R, CYL_LEN, pos=pos, density=1000.0, friction=FRICTION, restitution=0.0)

    def _spots(self) -> np.ndarray:
        xy = self._rng.uniform(-DROP_HALF, DROP_HALF, (self.num_envs, COUNT, 2))
        z = np.broadcast_to(DROP_Z0 + DROP_DZ * np.arange(COUNT), (self.num_envs, COUNT))[..., None]
        return np.concatenate([xy, z], axis=-1).astype(np.float32)

    def reset(self) -> np.ndarray:
        if not self._fresh:
            spots = self._spots()
            self._sim.reset()
            for i, name in enumerate(self._names):
                self._sim.set_pose(name, pos=spots[:, i], quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        self._fresh = False
        self._sim.set_targets({"bx": np.zeros(self.num_envs), "by": np.zeros(self.num_envs)}, kind="velocity")
        self._k = 0
        return self._obs()

    def _bin(self) -> np.ndarray:
        s = self._sim.state(["bin/bx", "bin/by"])
        return np.stack([s["bin/bx.q"], s["bin/by.q"], np.zeros(self.num_envs)], axis=1).astype(np.float32)

    def _objects(self) -> np.ndarray:
        s = self._sim.state(self._names)
        return np.stack([s[f"{n}.pos"] for n in self._names], axis=1).astype(np.float32)

    def _inside(self) -> np.ndarray:
        rel = self._objects() - self._bin()[:, None, :]
        return (np.abs(rel[..., :2]).max(-1) < HALF) & (rel[..., 2] > 0.0) & (rel[..., 2] < DEPTH)

    def _obs(self) -> np.ndarray:
        rel = (self._objects() - self._bin()[:, None, :]).reshape(self.num_envs, -1)
        return np.concatenate([rel, self._bin()], axis=1).astype(np.float32)

    def _drag_spheres(self) -> None:
        """Rolling resistance as a drag force -c v held on each round object's (sphere, cylinder) COM for the control period."""
        spheres = [n for n, k in zip(self._names, self._kinds) if k != "box"]
        if not spheres:
            return
        vel = self._sim.state(spheres)
        for n in spheres:
            self._sim.apply_wrench(n, force=(-ROLL_DRAG * vel[f"{n}.vel"][:, :3]).astype(np.float32))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 2), -1.0, 1.0) * SPEED
        self._sim.set_targets({"bx": a[:, 0], "by": a[:, 1]}, kind="velocity")
        self._drag_spheres()
        self._sim.step(1)
        self._k += 1
        reward = self._inside().sum(axis=1).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"object_pos": self._objects(), "bin_pos": self._bin()}

    def success(self) -> np.ndarray:
        return self._inside().all(axis=1)


def make_env(num_envs: int, seed: int) -> BinClutterEnv:
    return BinClutterEnv(num_envs, seed)
