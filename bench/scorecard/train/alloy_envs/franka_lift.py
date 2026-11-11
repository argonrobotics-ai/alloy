# SPDX-License-Identifier: Apache-2.0
"""FrankaLift: the shipped Franka Panda lifts a 5 cm cube from the table to a goal point in the air.

Modelled on Isaac Lab's Isaac-Lift-Franka: joint position actions for the 7 arm joints and the gripper, a cube
placed at random on the table, a goal point drawn in the air above it at every reset, a shaped reward (reach,
lift, goal tracking) and a 5 s episode. Example::

    env = make_franka_lift(num_envs=64)
    obs, _ = env.reset(seed=0); obs, reward, term, trunc, info = env.step(env.action_space.sample())
    env.last_metrics["goal_dist"]          # (64,) m, cube to goal, measured this step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alloy.envs import ActionSpec, Task
from alloy.envs import checks as C
from alloy.envs.randomize import jitter_body, jitter_joint
from alloy.envs.state import BatchState, rotate

from bench.scorecard.train.alloy_envs.metered import MeteredEnv

PANDA = "panda"
HOME = {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": -1.5708, "joint5": 0.0, "joint6": 1.5708,
        "joint7": -0.7854}
FINGER_OPEN = 0.04
TCP_OFFSET = (0.0, 0.0, 0.1)            # m, from the hand body to between the fingertips (hand z axis)


@dataclass(frozen=True)
class FrankaLiftConfig:
    """Scene, task and randomization settings of FrankaLift. Example: ``FrankaLiftConfig(max_steps=150)``."""

    dt: float = 0.02                    # s per control step (50 Hz)
    max_steps: int = 250                # 5 s episodes
    cube_size: float = 0.05             # m, full edge
    cube_mass: float = 0.1              # kg
    cube_friction: float = 1.0
    cube_start: tuple[float, float] = (0.5, 0.0)        # m, centre of the random start region (x, y)
    cube_jitter: tuple[float, float] = (0.1, 0.25)      # m, half-ranges around it
    goal_box: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.4, -0.25, 0.25), (0.6, 0.25, 0.5))
    home_jitter: float = 0.1            # rad, random offset of each arm joint at reset
    arm_delta: float = 0.05             # rad of target change per step at action 1
    grip_delta: float = 0.008           # m of finger target change per step at action 1
    lift_height: float = 0.04           # m the cube must rise to count as lifted


def franka_lift_scene(cfg: FrankaLiftConfig, worlds: int) -> Any:
    """Panda on its table (no props) plus the free cube. Example: ``franka_lift_scene(FrankaLiftConfig(), 8)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415
    from alloy.envs.mjcf import without_bodies  # noqa: PLC0415

    path = without_bodies(default_root() / "robots" / "emb_e2_panda" / "scene.xml", ["box", "cylinder", "mug"])
    scene = Scene(worlds=worlds, dt=cfg.dt)
    scene.add_robot(PANDA, path)
    x, y = cfg.cube_start
    scene.add_box("cube", size=(cfg.cube_size,) * 3, pos=(x, y, cfg.cube_size / 2), mass=cfg.cube_mass,
                  friction=cfg.cube_friction)
    return scene


def tcp_position(s: BatchState) -> Any:
    """(W, 3) point between the fingertips. Example: ``tcp_position(env.state)``."""
    hand = s[f"{PANDA}/hand.pos"]
    off = s.xp.asarray(TCP_OFFSET).reshape(1, 3) + hand * 0.0
    return hand + rotate(s.xp, s[f"{PANDA}/hand.quat_xyzw"], off)


def _goal_dist(s: BatchState) -> Any:
    return s.xp.norm(s["cube.pos"] - s["target.pos"])


def _lifted(cfg: FrankaLiftConfig) -> Any:
    return lambda s: s["cube.pos"][:, 2] - s.start["cube.pos"][:, 2] > cfg.lift_height


def franka_lift_task(cfg: FrankaLiftConfig) -> Task:
    """Observations, delta joint-target actions, Isaac-Lift-style reward terms and the cube-lost failure."""
    actuators = [f"{PANDA}/actuator{k}" for k in range(1, 8)] + [f"{PANDA}/hand_finger"]
    spec = ActionSpec(actuators, mode="delta", delta=[cfg.arm_delta] * 7 + [cfg.grip_delta])
    lifted = C.custom("lifted", _lifted(cfg))
    reach = C.custom("reach", lambda s: 1.0 - s.xp.tanh(s.xp.norm(tcp_position(s) - s["cube.pos"]) / 0.1))
    track = C.custom("goal_dist", _goal_dist)
    joint_vel = C.custom("joint_vel", lambda s: sum(s[f"{PANDA}/joint{k}.qd"] ** 2 for k in range(1, 8)))
    return Task(
        name="FrankaLift", obs=[PANDA, "cube", "target", "action"], actions=spec,
        reward={"reach": (1.0, reach), "lift": (15.0, lifted),
                "goal": (16.0, lifted * C.custom("g", lambda s: 1.0 - s.xp.tanh(_goal_dist(s) / 0.3))),
                "goal_fine": (5.0, lifted * C.custom("gf", lambda s: 1.0 - s.xp.tanh(_goal_dist(s) / 0.05))),
                "action": (-1e-4, C.action_size()), "joint_vel": (-1e-4, joint_vel)},
        failure=C.height("cube") < -0.05, max_steps=cfg.max_steps,
        randomize=_randomize(cfg), goals={"target": cfg.goal_box}, end_on_success=False)


def _randomize(cfg: FrankaLiftConfig) -> list[Any]:
    jx, jy = cfg.cube_jitter
    jitters = [jitter_body("cube", pos=(jx, jy, 0.0), yaw_deg=180.0)]
    jitters += [jitter_joint(f"{PANDA}/{j}", by=cfg.home_jitter, at=v) for j, v in HOME.items()]
    jitters.append(jitter_joint(f"{PANDA}/finger_joint1", by=0.0, at=FINGER_OPEN))
    return jitters


def franka_lift_metrics(cfg: FrankaLiftConfig) -> Any:
    """The protocol metrics per world and step: goal distance (m) and whether the cube is lifted."""
    def fn(s: BatchState) -> dict[str, Any]:
        d = _goal_dist(s)
        return {"goal_dist": d, "lifted": _lifted(cfg)(s), "success_2cm": d < 0.02, "success_5cm": d < 0.05}
    return fn


def make_franka_lift(num_envs: int, cfg: FrankaLiftConfig | None = None, device: str | None = None, seed: int = 0,
                     **env_kwargs: Any) -> MeteredEnv:
    """A FrankaLift vector env (torch backend if PyTorch is there). Example: ``make_franka_lift(4096)``."""
    cfg = cfg or FrankaLiftConfig()
    return MeteredEnv(franka_lift_scene(cfg, num_envs), franka_lift_task(cfg), metrics=franka_lift_metrics(cfg),
                      worlds=num_envs, device=device, seed=seed, **env_kwargs)
