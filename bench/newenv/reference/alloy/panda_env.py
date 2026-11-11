# SPDX-License-Identifier: Apache-2.0
"""Shared base of the row 7 Franka Panda reference environments (E19-E22) on Alloy's public API.

Each environment is an ``alloy.envs.VectorEnv`` (per-world resets, start jitters, delta joint-target actions) behind
the row 7 contract: ``reset() -> obs``, ``step(a) -> (obs, reward, done)``, ``probe()``, ``success()``. Subclasses
add the task's objects, probes, reward and success test, and a scripted policy built on ``panda_kin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import alloy
import alloy.envs  # noqa: F401  (not loaded by ``import alloy``)
from alloy.cards.library import default_root
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs.mjcf import without_bodies
from alloy.envs.randomize import Jitter, jitter_joint

import panda_kin as kin

ARM_JOINTS = [f"panda/joint{k}" for k in range(1, 8)]
ARM_ACTUATORS = [f"panda/actuator{k}" for k in range(1, 8)]
TABLE_Z = 0.0                 # top of the table in the library Panda scene (robot base on it, at the origin)
FINGER_OPEN = 0.04            # each finger's travel (m); gripper width = sum of both


@dataclass
class PandaConfig:
    """What differs between the Panda tasks."""

    start_q: np.ndarray                         # (7,) arm joint angles at every reset
    gripper_action: bool = True                 # 8th action column drives the fingers
    finger_start: float = FINGER_OPEN           # each finger's start opening (m)
    finger_target: float | None = None          # servo target at reset (None: the start opening)
    joint_step: float = 0.05                    # rad per step at action +-1
    dt: float = 0.02
    episode_s: float = 6.0
    obs: list[Any] = field(default_factory=lambda: ["panda"])
    jitters: list[Jitter] = field(default_factory=list)
    goals: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    robot_file: Path | None = None              # default: the library Panda scene without its props


def panda_file() -> Path:
    """The library Panda scene (arm, hand, table, floor) without its loose box, cylinder and mug."""
    return without_bodies(default_root() / "robots" / "emb_e2_panda" / "scene.xml", ["box", "cylinder", "mug"])


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (3,) or (N, 3) by unit quaternions q (N, 4), w first."""
    w, xyz = q[:, :1], q[:, 1:]
    v = np.broadcast_to(v, xyz.shape)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


class PandaEnv:
    """Row 7 contract over a VectorEnv. Subclasses: ``add_objects``, ``probe``, ``reward``, ``success_now``."""

    def __init__(self, num_envs: int, seed: int, cfg: PandaConfig) -> None:
        self.cfg, self.num_envs, self.dt = cfg, int(num_envs), cfg.dt
        self.episode_steps = round(cfg.episode_s / cfg.dt)
        scene = alloy.Scene(worlds=self.num_envs, dt=cfg.dt)
        scene.add_robot("panda", cfg.robot_file or panda_file())
        self.add_objects(scene)
        self.venv = VectorEnv(scene, self._task(), worlds=self.num_envs, seed=seed, device="cpu", backend="numpy")
        self.act_dim = 8 if cfg.gripper_action else 7
        self.hold = np.zeros(self.num_envs, np.int64)        # consecutive steps the success condition held
        self.stage = np.zeros(self.num_envs, np.int64)       # scripted policy: stage per world
        self.clock = np.zeros(self.num_envs, np.int64)       # scripted policy: steps in the stage
        self.obs_dim = int(self.reset().shape[1])

    # --- to override ---------------------------------------------------------------------------------------------

    def add_objects(self, scene: Any) -> None:
        """Add the task's objects to the scene (before the build)."""

    def probe(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def reward(self) -> np.ndarray:
        raise NotImplementedError

    def success_now(self) -> np.ndarray:
        """(N,) bool: the success condition at this step (``success_hold_s`` turns it into success)."""
        raise NotImplementedError

    success_hold_s: float = 0.0

    # --- contract ------------------------------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs, _info = self.venv.reset()
        self._clear(np.ones(self.num_envs, bool))
        return np.asarray(obs, np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, self.act_dim), -1.0, 1.0)
        obs, _r, terminated, truncated, _info = self.venv.step(a)
        done = np.asarray(terminated | truncated, bool)
        self.hold = np.where(self.success_now(), self.hold + 1, 0)
        self._clear(done)
        return np.asarray(obs, np.float32), self.reward().astype(np.float32), done

    def success(self) -> np.ndarray:
        return self.hold > max(0, round(self.success_hold_s / self.dt) - 1)

    def close(self) -> None:
        self.venv.close()

    # --- state ---------------------------------------------------------------------------------------------------

    def state(self, key: str) -> np.ndarray:
        """One named state array, (N, ...) float64. Example: ``env.state("cube.pos")``."""
        return np.asarray(self.venv.state[key], np.float64)

    def joint_pos(self) -> np.ndarray:
        return np.stack([self.state(f"{j}.q") for j in ARM_JOINTS], axis=1)

    def ee_pos(self) -> np.ndarray:
        """The grasp point: centre between the finger pads, from the hand body's pose."""
        tool = np.array([0.0, 0.0, kin.GRASP_Z])
        return self.state("panda/hand.pos") + quat_rotate_wxyz(self.state("panda/hand.quat_wxyz"), tool)

    def gripper_width(self) -> np.ndarray:
        return self.state("panda/finger_joint1.q") + self.state("panda/finger_joint2.q")

    # --- helpers -------------------------------------------------------------------------------------------------

    def _task(self) -> Task:
        cfg = self.cfg
        names = ARM_ACTUATORS + (["panda/hand_finger"] if cfg.gripper_action else [])
        steps = [cfg.joint_step] * 7 + ([FINGER_OPEN] if cfg.gripper_action else [])
        home = [jitter_joint(j, at=float(v)) for j, v in zip(ARM_JOINTS, cfg.start_q)]
        fingers = [jitter_joint("panda/finger_joint1", at=cfg.finger_start, target=cfg.finger_target),
                   jitter_joint("panda/finger_joint2", at=cfg.finger_start)]
        return Task(obs=cfg.obs, actions=ActionSpec(names, mode="delta", delta=steps),
                    max_steps=self.episode_steps, end_on_success=False, randomize=home + fingers + cfg.jitters,
                    goals=cfg.goals, name=type(self).__name__)

    def _clear(self, mask: np.ndarray) -> None:
        self.hold[mask], self.stage[mask], self.clock[mask] = 0, 0, 0

    def advance(self, move_on: np.ndarray) -> None:
        """Scripted policies: worlds in ``move_on`` enter their next stage; every world's stage clock ticks."""
        self.stage = np.where(move_on, self.stage + 1, self.stage)
        self.clock = np.where(move_on, 0, self.clock + 1)

    def arm_action(self, p_des: np.ndarray, r_des: np.ndarray = kin.DOWN, *, speed: float = 0.4,
                   gain: float = 0.15) -> np.ndarray:
        """(N, 7) joint actions moving the grasp point toward p_des at up to ``speed`` m/s (resolved-rate IK on the
        measured joint angles; the servo targets integrate the steps, which also removes the gravity sag)."""
        q = self.joint_pos()
        err = np.asarray(p_des, np.float64) - kin.fk(q).grasp
        step = gain * err
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, speed * self.dt / np.maximum(norm, 1e-9))
        dq = kin.ik_step(q, kin.fk(q).grasp + step, r_des, max_lin=1.0, max_dq=self.cfg.joint_step,
                         rest=self.cfg.start_q, rot_gain=gain)
        return (dq / self.cfg.joint_step).astype(np.float32)
