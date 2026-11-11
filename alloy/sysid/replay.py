# SPDX-License-Identifier: Apache-2.0
"""Replay a recording's commands in many worlds, each with its own candidate parameters.

``Bench`` builds the robot on its stand (``alloy.sysid.robot``) in ``worlds`` parallel worlds. ``rollout`` gives
every world its parameters, then for each episode: puts the joints at the first measured position at rest, sends
the recorded commands (through each world's action delay) one control period at a time, and returns the simulated
joint positions at the recording's sample times.

Time: sample ``k`` of an episode is at ``k * dt``; the command of sample ``k`` is held from then to sample
``k + 1``, where the simulated position is compared with the measured one. A recording at another rate is put on
the simulation's periods first (``resample``: commands held, positions interpolated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.sysid import model as M
from alloy.sysid.delay import DelayLine
from alloy.sysid.errors import SysidError
from alloy.sysid.recording import Episode, Recording
from alloy.sysid.robot import NAME, bench_scene

MIN_DT, MAX_DT = 0.002, 0.02       # control periods the replay runs at; a recording outside is resampled


@dataclass
class Track:
    """One episode on the simulation's control periods: ``command`` and ``position`` (n + 1, J)."""

    command: np.ndarray
    position: np.ndarray


def control_dt(rec: Recording) -> float:
    """The replay's control period for ``rec``: its sample period, held within [2 ms, 20 ms]."""
    return float(np.clip(round(rec.dt, 6), MIN_DT, MAX_DT))


def resample(episode: Episode, dt: float) -> Track:
    """``episode`` at times ``k * dt``: the last command at or before each time, positions interpolated."""
    steps = int(np.floor(episode.t[-1] / dt + 1e-9))
    if steps < 5:
        raise SysidError(f"an episode lasts {episode.t[-1]:.3g} s, under 5 control periods of {dt} s",
                         "record a longer take", field="data")
    t = np.arange(steps + 1) * dt
    last = np.clip(np.searchsorted(episode.t, t + 1e-9, side="right") - 1, 0, len(episode.t) - 1)
    pos = np.stack([np.interp(t, episode.t, episode.position[:, j]) for j in range(episode.position.shape[1])], 1)
    return Track(episode.command[last], pos)


class Bench:
    """The robot on its stand in ``worlds`` worlds, and the joints of the recording in it."""

    def __init__(self, robot: Any, joints: list[str] | None, worlds: int, dt: float,
                 device: str | None = None) -> None:
        import alloy  # noqa: PLC0415

        scene, self.notes = bench_scene(robot, worlds, dt)
        self.sim = alloy.build(scene, device=device, seed=0)
        self.joints = M.RobotJoints.read(self.sim.built, NAME)
        self.joint_names = list(joints) if joints else list(self.joints.names)      # None: every servoed joint
        self.cols = self.joints.pick(self.joint_names)
        self.names = [self.joints.full(j) for j in self.joint_names]
        self.worlds, self.dt = int(worlds), float(self.sim.dt)
        self.nominal = {f: v[self.cols] for f, v in self.joints.nominal.items()}

    @property
    def lower(self) -> np.ndarray:
        return self.joints.lower[self.cols]

    @property
    def upper(self) -> np.ndarray:
        return self.joints.upper[self.cols]

    @property
    def hinge(self) -> np.ndarray:
        return self.joints.hinge[self.cols]

    def set_parameters(self, fields: dict[str, np.ndarray]) -> None:
        """Give each world its joint parameters: ``fields[name]`` is (worlds, J) or (J,) (``delay`` is ignored
        here; ``rollout`` applies it)."""
        values = {f: v for f, v in fields.items() if f in M.FIELDS}
        if values:
            M.write(self.sim.built.model, self.joints, values, self.cols)
            M.notify(self.sim.engine)

    def _start(self, position: np.ndarray) -> None:
        """Every world's recorded joints at ``position`` (J,), all at rest."""
        from alloy._vendor import newton  # noqa: PLC0415

        self.sim.reset()
        eng = self.sim.engine
        model, state = eng.model, eng.state
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        at = self.joints.q_index[self.cols][None, :] + np.arange(self.worlds)[:, None] * self.joints.stride["coord"]
        q[at] = position[None, :]
        qd[:] = 0.0
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def rollout(self, fields: dict[str, np.ndarray], tracks: list[Track]) -> list[np.ndarray]:
        """Simulated positions per episode, each (n + 1, worlds, J), for the parameters ``fields`` (see
        ``set_parameters``; ``fields["delay"]`` in seconds, (worlds, J) or (J,))."""
        self.set_parameters(fields)
        shape = (self.worlds, len(self.cols))
        delay = np.broadcast_to(np.asarray(fields.get("delay", 0.0), np.float64), shape)
        keys = [f"{n}.q" for n in self.names]
        out = []
        for track in tracks:
            self._start(track.position[0])
            line = DelayLine(delay / self.dt)
            sim_q = np.empty((len(track.command), *shape))
            sim_q[0] = track.position[0]
            for k in range(len(track.command) - 1):
                target = line.push(np.broadcast_to(track.command[k], shape))
                self.sim.set_targets({n: target[:, j] for j, n in enumerate(self.names)})
                self.sim.step(1)
                state = self.sim.state(self.names)
                sim_q[k + 1] = np.stack([state[key] for key in keys], axis=1)
            out.append(sim_q)
        return out


def squared_error(sim_q: list[np.ndarray], tracks: list[Track]) -> np.ndarray:
    """Mean squared tracking error (worlds, J) over all episodes; a world that went non-finite reads ``inf``."""
    total = sum(np.sum((q - t.position[:, None, :]) ** 2, axis=0) for q, t in zip(sim_q, tracks))
    count = sum(len(t.position) for t in tracks)
    mse = total / count
    return np.where(np.isfinite(mse), mse, np.inf)


__all__ = ["Bench", "Track", "control_dt", "resample", "squared_error"]
