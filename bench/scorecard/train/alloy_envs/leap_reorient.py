# SPDX-License-Identifier: Apache-2.0
"""LeapReorient: the shipped LEAP hand, palm up, turns a cube in hand to a goal orientation.

Modelled on Isaac Lab's Reorient-Cube-Allegro and Playground's LeapCubeReorient: joint position targets for the 16
hand joints, observations of joint state, cube pose, goal orientation and last action, a goal orientation drawn
uniformly at every reset and again each time it is reached, a rotation-error reward with a success bonus and a drop
penalty, and an episode that ends when the cube is dropped. Example::

    env = make_leap_reorient(num_envs=64)
    obs, _ = env.reset(seed=0); obs, reward, term, trunc, info = env.step(env.action_space.sample())
    env.last_metrics["rot_err"]            # (64,) rad, cube to goal orientation, measured this step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alloy.envs import ActionSpec, Task
from alloy.envs import checks as C
from alloy.envs.randomize import jitter_body
from alloy.envs.state import BatchState

from bench.scorecard.train.alloy_envs.metered import MeteredEnv
from bench.scorecard.train.alloy_envs.quat import angle_between, random_unit_quats

HAND = "leap"
CUBE = f"{HAND}/cube"


@dataclass(frozen=True)
class LeapReorientConfig:
    """Scene, task and reward settings of LeapReorient. Example: ``LeapReorientConfig(tolerance=0.1)``."""

    dt: float = 0.05                    # s per control step (20 Hz)
    max_steps: int = 200                # 10 s episodes
    tolerance: float = 0.2              # rad, a goal counts as reached below this rotation error
    drop_distance: float = 0.1          # m the cube may leave its start before it counts as dropped
    hand_delta: float = 0.1             # rad of target change per step at action 1
    cube_jitter: float = 0.01           # m, random start offset in x and y
    success_bonus: float = 250.0
    drop_penalty: float = -50.0


class OrientationGoal:
    """A per-world goal orientation (wxyz), read as an observation and redrawn for chosen worlds.

    It allocates on first use from the state's backend. Example: ``goal.resample(xp, mask)``."""

    def __init__(self) -> None:
        self.value: Any = None

    def __call__(self, s: BatchState) -> Any:
        """(W, 4) goal quaternions; identity until the first resample."""
        if self.value is None:
            self.value = s.xp.zeros((s.worlds, 4))
            self.value[:, 0] = 1.0
        return self.value

    def resample(self, xp: Any, mask: Any, worlds: int) -> None:
        """Draw new uniform orientations for the worlds in the bool (W,) ``mask``."""
        if self.value is None:
            self.value = xp.zeros((worlds, 4))
            self.value[:, 0] = 1.0
        self.value = xp.where(mask[:, None], random_unit_quats(xp, worlds), self.value)


class LeapEnv(MeteredEnv):
    """``MeteredEnv`` that redraws the goal orientation at reset and when a world reaches it."""

    def __init__(self, sim: Any, task: Task, *, goal: OrientationGoal, **kwargs: Any) -> None:
        self.goal = goal
        super().__init__(sim, task, **kwargs)

    def reset_worlds(self, mask: Any) -> None:
        super().reset_worlds(mask)
        self.goal.resample(self.xp, self._mask(mask), self.num_envs)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(actions)
        reached = info["success"] & ~(terminated | truncated)
        if self.xp.any(reached):
            self.goal.resample(self.xp, reached, self.num_envs)
            obs = self._observe(self.state)
        return obs, reward, terminated, truncated, info


def leap_reorient_scene(cfg: LeapReorientConfig, worlds: int) -> Any:
    """The shipped LEAP hand with its cube (the file has its own floor). Example: ``leap_reorient_scene(cfg, 8)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415

    scene = Scene(worlds=worlds, dt=cfg.dt)
    scene.add_robot(HAND, default_root() / "robots" / "emb_e9_leap" / "scene.xml")
    return scene


def _rot_err(goal: OrientationGoal) -> Any:
    return lambda s: angle_between(s.xp, s[f"{CUBE}.quat_wxyz"], goal(s))


def _dropped(cfg: LeapReorientConfig) -> Any:
    return lambda s: s.xp.norm(s[f"{CUBE}.pos"] - s.start[f"{CUBE}.pos"]) > cfg.drop_distance


def leap_reorient_task(cfg: LeapReorientConfig, goal: OrientationGoal) -> Task:
    """Delta joint-target actions on all 16 joints, Isaac-style rotation reward and drop termination."""
    rot_err = C.custom("rot_err", _rot_err(goal))
    reached = rot_err < cfg.tolerance
    dropped = C.custom("dropped", _dropped(cfg))
    return Task(
        name="LeapReorient", obs=[HAND, CUBE, ("goal_quat", goal), "action"],
        actions=ActionSpec(None, kind="ctrl", mode="delta", delta=cfg.hand_delta),
        reward={"rotation": (1.0, C.custom("align", lambda s: 1.0 / (_rot_err(goal)(s) + 0.1))),
                "success": (cfg.success_bonus, reached), "drop": (cfg.drop_penalty, dropped),
                "action": (-1e-4, C.action_size())},
        success=reached, failure=dropped, max_steps=cfg.max_steps, end_on_success=False,
        randomize=[jitter_body(CUBE, pos=(cfg.cube_jitter, cfg.cube_jitter, 0.0), yaw_deg=180.0)])


def leap_reorient_metrics(cfg: LeapReorientConfig, goal: OrientationGoal) -> Any:
    """Per world and step: rotation error (rad), goal reached this step, cube dropped."""
    err, dropped = _rot_err(goal), _dropped(cfg)

    def fn(s: BatchState) -> dict[str, Any]:
        e = err(s)
        return {"rot_err": e, "goal_reached": e < cfg.tolerance, "dropped": dropped(s)}
    return fn


def make_leap_reorient(num_envs: int, cfg: LeapReorientConfig | None = None, device: str | None = None,
                       seed: int = 0, **env_kwargs: Any) -> LeapEnv:
    """A LeapReorient vector env (torch backend if PyTorch is there). Example: ``make_leap_reorient(4096)``."""
    cfg = cfg or LeapReorientConfig()
    goal = OrientationGoal()
    return LeapEnv(leap_reorient_scene(cfg, num_envs), leap_reorient_task(cfg, goal), goal=goal,
                   metrics=leap_reorient_metrics(cfg, goal), worlds=num_envs, device=device, seed=seed,
                   **env_kwargs)
