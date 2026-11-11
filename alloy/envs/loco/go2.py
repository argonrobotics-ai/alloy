# SPDX-License-Identifier: Apache-2.0
"""Go2 flat-ground velocity tracking: the reference legged environment.

The Unitree Go2 from the library (MuJoCo Menagerie file, BSD-3-Clause) stands on a flat floor; every world gets a
velocity command (forward, sideways, yaw rate) redrawn every 10 s, and a push on the base every 8 s. The 12 actions
move the leg servos' targets around the stand (0.25 rad per unit). An episode ends when the robot falls (base below
15 cm or tilted over 60 degrees) or after 20 s. Example::

    from alloy.envs.loco import go2_velocity
    env = go2_velocity(num_envs=4096)
    obs, info = env.reset(seed=0)

Covered: flat ground, one robot (Go2), position-servo actions, the reward terms in ``DEFAULT_REWARD``. Not covered:
rough terrain, curricula, actuator models, observation noise or delay, and no policy trained here has been run on a
real robot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alloy.envs.actions import ActionSpec
from alloy.envs.errors import EnvError
from alloy.envs.loco import terms as T
from alloy.envs.loco.commands import VelocityCommands
from alloy.envs.loco.env import LocoConfig, LocoEnv
from alloy.envs.loco.pushes import Pushes
from alloy.envs.loco.robot import GO2
from alloy.envs.randomize import jitter_joint
from alloy.envs.task import Task

ROBOT = "go2"
STAND_HEIGHT = 0.28          # m: the base height the reward asks for
#: reward term -> weight per second of simulated time (each step adds weight * dt * value); "fell" is per event
DEFAULT_REWARD: dict[str, float] = {
    "track_lin_vel": 2.0, "track_yaw_rate": 1.0, "lin_vel_z": -2.0, "ang_vel_xy": -0.05, "flat_orientation": -5.0,
    "torques": -2e-4, "joint_acc": -2.5e-7, "action_rate": -0.01, "feet_air_time": 1.0, "feet_slide": -0.25,
    "body_contacts": -5.0, "stand_still": -0.5, "pose_deviation": -0.1, "hip_deviation": -1.0, "base_height": -50.0,
    "feet_too_high": -10.0,
    "fell": -2.0}
_TERMS = {"track_lin_vel": T.track_lin_vel, "track_yaw_rate": T.track_yaw_rate, "lin_vel_z": T.lin_vel_z,
          "ang_vel_xy": T.ang_vel_xy, "flat_orientation": T.flat_orientation, "torques": T.torques,
          "joint_acc": T.joint_acc, "action_rate": T.action_rate, "feet_air_time": T.feet_air_time,
          "feet_slide": T.feet_slide, "body_contacts": T.body_contacts, "pose_deviation": T.pose_deviation,
          "hip_deviation": T.hip_deviation, "base_height": lambda: T.base_height(STAND_HEIGHT),
          "feet_too_high": T.feet_too_high,
          "stand_still": T.stand_still, "fell": T.fell}
PER_EVENT = ("fell", "feet_air_time")        # not scaled by dt: they fire on single steps


@dataclass
class Go2VelocityConfig:
    """Options of the Go2 velocity environment. Example: ``Go2VelocityConfig(pushes=None, episode_s=10.0)``."""

    dt: float = 0.02                                       # control period, s (50 Hz)
    episode_s: float = 20.0
    commands: VelocityCommands = field(default_factory=VelocityCommands)
    pushes: Pushes | None = field(default_factory=Pushes)
    reward: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD))
    action_scale: float = 0.25
    joint_jitter: float = 0.0                              # rad: each leg joint starts within +- this of the stand
    policy_obs: Sequence[Any] = T.POLICY_OBS
    privileged_obs: Sequence[Any] = T.PRIVILEGED_OBS       # the critic group sees policy_obs + these
    contact: Any = "force"                                 # foot contact: "force" (solver foot forces) or "height"
    scene_randomize: Sequence[tuple[str, str, float, float]] = ()   # (name, field, low, high) -> scene.randomize


def go2_scene(worlds: int = 1, seed: int = 0, dt: float = 0.02) -> Any:
    """The Go2 on its flat floor (the library file without its box obstacle); parts are "go2/<body or joint>".

    Example: ``alloy.build(go2_scene(64), worlds=64)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415
    from alloy.envs.mjcf import without_bodies  # noqa: PLC0415

    path = default_root() / "robots" / "emb_e8_go2" / "scene.xml"
    if not path.is_file():
        raise EnvError(f"the Go2 robot file is missing ({path.name})", "install Alloy with its library (a source "
                       "checkout or the full wheel)", field="scene")
    scene = Scene(worlds=worlds, dt=dt)
    scene.add_robot(ROBOT, without_bodies(path, ["obstacle"]))
    return scene


def go2_velocity_task(config: Go2VelocityConfig | None = None) -> Task:
    """The velocity-tracking task (observations, actions, reward, ends). Example: ``go2_velocity_task().max_steps``."""
    cfg = config or Go2VelocityConfig()
    unknown = sorted(set(cfg.reward) - set(_TERMS))
    if unknown:
        raise EnvError(f"unknown reward terms {unknown}", f"use names from {sorted(_TERMS)}", field="reward")
    reward = {name: (w if name in PER_EVENT else w * cfg.dt, _TERMS[name]()) for name, w in cfg.reward.items() if w}
    joints = GO2.joint_names(ROBOT)
    jitters = [jitter_joint(j, by=cfg.joint_jitter) for j in joints] if cfg.joint_jitter > 0 else []
    obs = {"policy": T.entries(cfg.policy_obs), "critic": T.entries([*cfg.policy_obs, *cfg.privileged_obs])}
    return Task(name="Go2Velocity-v0", obs=obs,
                actions=ActionSpec(joints, kind="ctrl", mode="absolute"), reward=reward, failure=T.fell(),
                max_steps=max(1, int(round(cfg.episode_s / cfg.dt))), randomize=jitters)


def go2_velocity(num_envs: int = 1, device: str | None = None, seed: int = 0,
                 config: Go2VelocityConfig | None = None, **env_options: Any) -> LocoEnv:
    """The Go2 velocity-tracking environment with ``num_envs`` worlds. ``env_options`` go to the environment
    (``backend=``, ``autoreset=``, ``render_mode=``).

    Example: ``env = go2_velocity(num_envs=1024, config=Go2VelocityConfig(pushes=None))``."""
    cfg = config or Go2VelocityConfig()
    scene = go2_scene(num_envs, seed, cfg.dt)
    for name, fld, low, high in cfg.scene_randomize:
        scene.randomize(name, fld, low, high)
    loco = LocoConfig(robot=GO2, name=ROBOT, commands=cfg.commands, pushes=cfg.pushes,
                      action_scale=cfg.action_scale, contact=cfg.contact)
    return LocoEnv(scene, go2_velocity_task(cfg), loco, worlds=num_envs, device=device, seed=seed, **env_options)


__all__ = ["DEFAULT_REWARD", "Go2VelocityConfig", "ROBOT", "go2_scene", "go2_velocity", "go2_velocity_task"]
