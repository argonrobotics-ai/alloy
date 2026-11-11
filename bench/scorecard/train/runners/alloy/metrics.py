# SPDX-License-Identifier: Apache-2.0
"""Protocol metrics for the Alloy cells: accumulate the env's per-step metrics, flush once per PPO iteration."""

from __future__ import annotations

from typing import Any

import torch


class Meter:
    """Sums per-step metrics over an iteration's rollout. ``record`` each step, ``flush`` once per iteration."""

    def __init__(self, task: str, num_envs: int, dt: float) -> None:
        self.task, self.num_envs, self.dt = task, num_envs, dt
        self.last: dict[str, float] = {}
        self._reset()

    def _reset(self) -> None:
        self.steps, self.reward, self.ended = 0, 0.0, 0.0
        self.acc: dict[str, Any] = {}

    def _add(self, key: str, value: Any) -> None:
        v = value.float().sum()
        self.acc[key] = self.acc[key] + v if key in self.acc else v

    def record(self, m: dict[str, Any], reward: Any, done: Any, timeouts: Any = None) -> None:
        """One env step: ``m`` the env's ``last_metrics``, ``done`` the bool (W,) of episodes that ended,
        ``timeouts`` the bool (W,) of those that ended by the time limit (go2_flat: the rest are falls)."""
        done = done.bool()
        self.steps += 1
        self._add("reward", reward)
        self._add("ended", done)
        if self.task == "go2_flat":
            self._add("vel_err_xy", m["vel_err_xy"])
            self._add("vel_err_yaw", m["vel_err_yaw"])
            self._add("falls", done & ~timeouts.bool())
        elif self.task == "franka_lift":
            for k in ("goal_dist", "lifted", "success_2cm", "success_5cm"):
                self._add(k, m[k].float() * done)
        else:
            self._add("rot_err", m["rot_err"])
            self._add("goals", m["goal_reached"])
            self._add("drops", m["dropped"] & done)

    def flush(self) -> dict[str, float]:
        """The iteration's metrics (protocol names). Keeps the previous ended-episode values when none ended."""
        a = {k: float(v) for k, v in self.acc.items()}
        env_steps = self.steps * self.num_envs
        out = {"reward": a["reward"] / env_steps}
        ended = a["ended"]
        if self.task == "go2_flat":
            out.update(vel_err_xy=a["vel_err_xy"] / env_steps, vel_err_yaw=a["vel_err_yaw"] / env_steps)
            if ended > 0:
                self.last = {"fall_rate": a["falls"] / ended}
        elif self.task == "franka_lift":
            if ended > 0:
                self.last = {"pos_err_end": a["goal_dist"] / ended, "success_2cm": a["success_2cm"] / ended,
                             "success_5cm": a["success_5cm"] / ended, "lifted": a["lifted"] / ended}
        else:
            self.last = {**self.last, "rot_err": a["rot_err"] / env_steps,
                         "goals_per_min": a["goals"] / (env_steps * self.dt / 60.0)}
            if ended > 0:
                self.last["drop_rate"] = a["drops"] / ended
        out.update(self.last)
        out["episodes_ended"] = ended
        self._reset()
        return out


def go2_step_metrics(state: Any) -> dict[str, Any]:
    """go2_flat per-env tracking errors from simulator state (the isaaclab / mjlab definition): ``vel_err_xy`` =
    norm of (commanded - actual) planar base velocity in the base frame [m/s], ``vel_err_yaw`` = |commanded - actual|
    yaw rate [rad/s]. ``state`` is the env's ``LocoState`` (read before any autoreset)."""
    cmd = state["command"]
    lin, ang = state["base.lin_vel_b"], state["base.ang_vel_b"]
    return {"vel_err_xy": torch.linalg.norm(cmd[:, :2] - lin[:, :2], dim=1), "vel_err_yaw": (cmd[:, 2] - ang[:, 2]).abs()}


def hook_go2_metrics(env: Any) -> None:
    """Make ``env.last_metrics`` hold this step's go2 errors, read from the state the reward is computed on: that
    is before ``step`` resets the worlds that ended, so a fall's last step counts with the state it ended in."""
    inner = env.rewarder

    def rewarder(state: Any) -> Any:
        env.last_metrics = go2_step_metrics(state)
        return inner(state)

    env.rewarder = rewarder
