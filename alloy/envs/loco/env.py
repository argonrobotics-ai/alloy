# SPDX-License-Identifier: Apache-2.0
"""``LocoEnv``: a ``VectorEnv`` for legged robots that follow velocity commands.

It adds to ``alloy.envs.VectorEnv`` (nothing there changes): the legged state keys (``alloy.envs.loco.state``),
velocity commands redrawn per world, pushes on a timer, foot-contact / air-time / accelerometer buffers, and actions
as offsets from the stand (``target = scale * action``). Observation groups come from the task
(``Task(obs={"policy": [...], "critic": [...]})``). Example::

    from alloy.envs.loco import go2_velocity
    env = go2_velocity(num_envs=1024)                       # GPU if there is one
    obs, info = env.reset(seed=0)                           # {"policy": (1024, 55), "critic": (1024, 79)}
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.envs import spaces
from alloy.envs.episode import LiveArrays
from alloy.envs.errors import EnvError
from alloy.envs.loco.commands import CommandSampler, VelocityCommands
from alloy.envs.loco.pushes import Pusher, Pushes
from alloy.envs.loco.robot import LeggedRobot, RobotRows
from alloy.envs.loco.sensors import LocoSensors
from alloy.envs.loco.state import LocoState
from alloy.envs.task import Task
from alloy.envs.vector import VectorEnv


@dataclass
class LocoConfig:
    """What makes a ``VectorEnv`` a legged one. Example: ``LocoConfig(robot=GO2, name="go2")``."""

    robot: LeggedRobot
    name: str = "robot"                                   # the robot's name in the scene
    commands: VelocityCommands = field(default_factory=VelocityCommands)
    pushes: Pushes | None = field(default_factory=Pushes)
    action_scale: float = 0.25                            # rad of joint target per unit of action
    action_limit: float = 4.0                             # actions are clipped to +-this
    contact: Any = "height"                               # "height", "force" (solver forces), or a reader
    contact_height: float = 0.005
    contact_force: float = 1.0


class LocoEnv(VectorEnv):
    """A batched legged-robot environment. Example: see the module docstring."""

    def __init__(self, sim: Any, task: Task, config: LocoConfig, **kwargs: Any) -> None:
        self.config = config
        self._tick, self._sensed = 0, -1
        super().__init__(sim, task, **kwargs)

    # --- set-up (called by VectorEnv.__init__ once the simulation and layout exist) --------------------------------

    def _make_spaces(self, seed: int) -> None:
        cfg, xp, w = self.config, self.xp, self.num_envs
        self.robot_rows = RobotRows(cfg.robot, cfg.name, self.layout, xp)
        self.robot_rows.ground = self._ground_height()
        dt, n = float(self.sim.dt), len(cfg.robot.joints)
        self.commands = CommandSampler(cfg.commands, w, dt, xp)
        self.pusher = Pusher(cfg.pushes, self.sim, f"{cfg.name}/{cfg.robot.base}", dt, xp)
        reader = self._contact_reader()
        self.sensors = LocoSensors(w, len(cfg.robot.feet), n, dt, xp, reader=reader,
                                   contact_height=cfg.contact_height, contact_force=cfg.contact_force)
        self._action, self._prev_action, self._targets = (xp.zeros((w, n)) for _ in range(3))
        self.sensors.reset(self._state(self._base_extra()), xp.zeros(w, "bool") == 0)
        self._sensed = self._tick
        super()._make_spaces(seed)
        self.single_action_space = spaces.box(-cfg.action_limit, cfg.action_limit, (n,))
        self.action_space = spaces.batch(self.single_action_space, w)
        self.action_space.seed(seed)

    def _ground_height(self) -> Any:
        """A function xy (..., 2) -> ground height (m) for terrain environments; None on flat ground at z = 0."""
        return None

    def _contact_reader(self) -> Any:
        cfg = self.config
        if cfg.contact == "height":
            return None
        if cfg.contact == "force":
            from alloy.envs.loco.contact import foot_forces  # noqa: PLC0415

            return foot_forces(self.sim, cfg.robot, cfg.name, self.xp)
        if isinstance(cfg.contact, str):
            raise EnvError(f"contact is {cfg.contact!r}", "use 'height', 'force' or a reader with read() -> (W, F) "
                           "newtons", field="contact")
        return cfg.contact

    # --- state -----------------------------------------------------------------------------------------------------

    def _base_extra(self) -> dict[str, Any]:
        st = self.stats
        return {"steps": st.steps, "t": self.xp.float(st.steps) * float(self.sim.dt), **self.goals.extra(),
                "joints.target": self._targets}

    def _state(self, extra: dict[str, Any]) -> LocoState:
        start = LocoState(self.start.view, self._rows, self.robot_rows, extra=extra)
        return LocoState(LiveArrays(self.sim.engine, self.xp), self._rows, self.robot_rows, extra=extra, start=start)

    @property
    def state(self) -> LocoState:
        """Every world's named state now, with the legged keys (``env.state["base.gravity_b"]``, ``["command"]``)."""
        base = self._base_extra()
        if self._sensed != self._tick:                    # once per control step, after the physics step
            self.sensors.advance(self._state(base))
            self.commands.tick(self.stats.steps)
            self._sensed = self._tick
        return self._state({**base, **self.sensors.extra(), "action": self._action,
                            "action.prev": self._prev_action, "command": self.commands.value,
                            "push.force": self.pusher.force})

    # --- stepping --------------------------------------------------------------------------------------------------

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """One control period: ``actions`` (W, J) are joint-target offsets from the stand, in units of
        ``action_scale`` rad. Returns ``(obs, reward, terminated, truncated, info)``; ``info["command"]`` is the
        command of every world."""
        self._require_open()
        xp, cfg = self.xp, self.config
        a = xp.clip(self.writer._check(actions), -cfg.action_limit, cfg.action_limit)
        self._prev_action, self._action = self._action, a
        self._targets = xp.clip(cfg.action_scale * a, self.writer.low, self.writer.high)
        self.pusher.before_step(self._tick)
        self._tick += 1
        obs, reward, terminated, truncated, info = super().step(self._targets)
        info["command"] = self.commands.value
        return obs, reward, terminated, truncated, info

    def reset_worlds(self, mask: Any) -> None:
        """Reset the worlds in the bool (W,) ``mask``: start state, a new command, cleared buffers."""
        mask = self._mask(mask)
        if not self.xp.any(mask):
            return
        super().reset_worlds(mask)
        xp, m1 = self.xp, mask[:, None]
        self._action = xp.where(m1, 0.0, self._action)
        self._prev_action = xp.where(m1, 0.0, self._prev_action)
        self._targets = xp.where(m1, 0.0, self._targets)
        self.commands.draw(mask)
        self.sensors.reset(self._state(self._base_extra()), mask)

    def _policy_action(self, actions: Any) -> Any:
        return self._action

    def action_names(self) -> list[str]:
        """A label per action column (joint names). Example: ``env.action_names()[0]``."""
        return self.config.robot.joint_names(self.config.name)


def stand_action(env: LocoEnv) -> np.ndarray:
    """The action that holds the stand: zeros (W, J). Example: ``env.step(stand_action(env))``."""
    return np.zeros((env.num_envs, len(env.config.robot.joints)), np.float32)


__all__ = ["LocoConfig", "LocoEnv", "stand_action"]
