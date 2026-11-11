# SPDX-License-Identifier: Apache-2.0
"""Protocol metrics for the mjlab runner, computed from simulator state (not from the reward).

A zero-weight probe is added as a termination term: mjlab computes terminations before it resets finished
episodes, so the probe sees the last state of every episode that ends this step. ``MetricsRecorder`` accumulates the
probe's per-step readings on the GPU and turns them into one metrics dict per PPO iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from mjlab.managers.termination_manager import TerminationTermCfg

PROBE_NAME = "bench_probe"


@dataclass
class StepReading:
    """What a probe leaves for one env step. ``per_step`` is averaged over envs and steps; ``at_end`` is read only
    for envs whose episode ends this step."""

    per_step: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> (num_envs,)
    at_end: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> (num_envs,)


class VelocityProbe:
    """go2_flat: commanded vs actual planar base velocity in the body frame, and yaw rate."""

    def __init__(self, command_name: str = "twist", entity: str = "robot") -> None:
        self.command_name, self.entity = command_name, entity
        self.reading = StepReading()

    def __call__(self, env) -> torch.Tensor:
        data = env.scene[self.entity].data
        cmd = env.command_manager.get_command(self.command_name)
        lin = torch.norm(cmd[:, :2] - data.root_link_lin_vel_b[:, :2], dim=-1)
        yaw = torch.abs(cmd[:, 2] - data.root_link_ang_vel_b[:, 2])
        self.reading = StepReading(
            per_step={"vel_err_xy": lin, "vel_err_yaw": yaw},
            at_end={"ep_len_steps": env.episode_length_buf.float()},
        )
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class LiftProbe:
    """franka_lift: object-to-goal distance at episode end, and whether the object rose 4 cm above its start."""

    LIFT_M = 0.04

    def __init__(self, command_name: str = "lift_height", object_name: str = "cube") -> None:
        self.command_name, self.object_name = command_name, object_name
        self.reading = StepReading()
        self._start_z: torch.Tensor | None = None
        self._max_rise: torch.Tensor | None = None

    def __call__(self, env) -> torch.Tensor:
        pos = env.scene[self.object_name].data.root_link_pos_w
        goal = env.command_manager.get_command(self.command_name)[:, :3]
        if self._start_z is None:
            self._start_z, self._max_rise = pos[:, 2].clone(), torch.zeros_like(pos[:, 2])
        first = env.episode_length_buf <= 1  # first step of a new episode: re-anchor the start height
        self._start_z = torch.where(first, pos[:, 2], self._start_z)
        self._max_rise = torch.where(first, torch.zeros_like(self._max_rise), self._max_rise)
        self._max_rise = torch.maximum(self._max_rise, pos[:, 2] - self._start_z)
        err = torch.norm(goal - pos, dim=-1)
        self.reading = StepReading(at_end={
            "ep_len_steps": env.episode_length_buf.float(),
            "pos_err_end": err,
            "success_2cm": (err < 0.02).float(),
            "success_5cm": (err < 0.05).float(),
            "lifted": (self._max_rise > self.LIFT_M).float(),
        })
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def install_probe(env_cfg, probe: Callable) -> None:
    """Add the probe to the env cfg as a never-firing termination term (before the env is built).

    The termination manager deep-copies its cfg, which would copy a callable object (and its readings); a plain
    function is shared by reference, so the closure keeps the recorder's probe live.
    """
    def probe_term(env):
        return probe(env)

    env_cfg.terminations[PROBE_NAME] = TerminationTermCfg(func=probe_term, params={})


class MetricsRecorder:
    """Accumulates readings over one iteration's rollout; ``flush`` returns the protocol metrics and resets."""

    def __init__(self, probe) -> None:
        self.probe = probe
        self._sum: dict[str, torch.Tensor] = {}
        self._end_sum: dict[str, torch.Tensor] = {}
        self._steps = 0
        self._ends = torch.zeros((), device="cuda")
        self._falls = torch.zeros((), device="cuda")
        self._reward = torch.zeros((), device="cuda")

    def record(self, env, rewards: torch.Tensor) -> None:
        """Call right after ``env.step``: the env's reset flags and the probe's pre-reset reading are both valid."""
        r = self.probe.reading
        for k, v in r.per_step.items():
            self._sum[k] = self._sum.get(k, 0) + v.mean()
        done = (env.reset_terminated | env.reset_time_outs)
        donef = done.float()
        for k, v in r.at_end.items():
            self._end_sum[k] = self._end_sum.get(k, 0) + (v * donef).sum()
        self._ends += donef.sum()
        self._falls += (env.reset_terminated & done).float().sum()
        self._reward += rewards.mean()
        self._steps += 1

    def flush(self) -> dict[str, float]:
        n = max(self._steps, 1)
        ends = float(self._ends)
        out = {k: float(v) / n for k, v in self._sum.items()}
        for k, v in self._end_sum.items():
            out[k] = float(v) / ends if ends > 0 else float("nan")
        out["fall_rate"] = float(self._falls) / ends if ends > 0 else float("nan")
        out["episodes_ended"] = ends
        out["reward"] = float(self._reward) / n
        self.__init__(self.probe)
        return out
