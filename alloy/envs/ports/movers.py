# SPDX-License-Identifier: Apache-2.0
"""What a task port's policy moves, as one action vector per world: servo targets on named joints, a kinematic tool's
pose and velocity, and plug-in command inputs (a spray valve, a faucet).

Each part reads the command the port's script wrote for this control period (``read``: the expert's action), writes
the policy's (``write``) and reports what a robot would sense (``proprio``). A write that equals what is there
changes nothing, so the expert's own action steps the port exactly as its script does. Parts join with
:class:`Mover`. Example::

    mover = Mover(ServoJoints([f"fr3_joint{i}" for i in range(1, 8)]), PluginInput("valve", low=0.0, high=1.0))
    mover.bind(sim)
    expert = mover.read(sim)          # (worlds, 8)
    mover.write(sim, expert)          # steps as the script would
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError


def _match(labels: Sequence[str], worlds: np.ndarray, name: str, what: str) -> np.ndarray:
    """Indices (one per world, world order) of the items whose label is ``name`` or ends with ``/name``."""
    hits = [i for i, lbl in enumerate(labels) if lbl == name or lbl.endswith("/" + name)]
    if not hits:
        raise EnvError(f"no {what} named {name!r} in the port's model", f"use one of {sorted(set(labels))[:20]}",
                       field=what)
    order = sorted(hits, key=lambda i: (int(worlds[i]), i))
    return np.asarray(order, dtype=np.int64)


def around_start(start: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Position bounds ``half`` metres either side of where the items start, over every world: ``start`` is (W, ..., 3),
    the result two flat float32 arrays of its per-world size. The scenes sit metres from the origin (the glass station
    beyond x = 3 m), so a box about the origin would leave the expert's own targets outside the action space, and a
    wrapper that clips to the space would break it. Example: ``around_start(np.zeros((2, 1, 3)), 3.0)[0]``."""
    start = np.asarray(start, np.float64)
    return ((start.min(axis=0) - half).reshape(-1).astype(np.float32),
            (start.max(axis=0) + half).reshape(-1).astype(np.float32))


def _worlds_of(sim: Any) -> int:
    return int(getattr(sim.model, "world_count", 1) or 1)


class ServoJoints:
    """Servos on joints ``joints`` (label suffixes, action order): the action is each servo's command (a position
    servo's target, a velocity servo's rate), written where the simulation reads it (Newton's ``joint_target_q`` for
    joint drives, else MuJoCo's ``ctrl``); proprio is the joints' q and qd. ``low`` / ``high``
    default to the joints' limits (position servos); give them for velocity servos."""

    def __init__(self, joints: Sequence[str], low: Sequence[float] | None = None,
                 high: Sequence[float] | None = None, proprio: bool = True) -> None:
        self.joints, self._low, self._high, self._proprio = list(joints), low, high, proprio
        self.size = len(self.joints)

    def bind(self, sim: Any) -> None:
        m = sim.model
        labels, world = [str(lbl) for lbl in m.joint_label], m.joint_world.numpy()
        idx = np.stack([_match(labels, world, j, "joint") for j in self.joints], axis=1)       # (W, J)
        self.q = m.joint_q_start.numpy()[idx]
        self.dof = m.joint_qd_start.numpy()[idx]
        ke = getattr(m, "joint_target_ke", None)
        target = getattr(sim.control, "joint_target_q", None)
        self.newton = bool(target is not None and ke is not None and (ke.numpy()[self.dof] > 0.0).all())
        if self.newton:                 # Newton's joint drives (alloy.build's MJCF <position>): joint_target_q
            self.act = m.joint_target_q_start.numpy()[idx]
        else:                           # MuJoCo actuators written directly (ctrl)
            by_dof = {int(d): a for a, d in enumerate(m.mujoco.actuator_trnid.numpy()[:, 0])}
            missing = [self.joints[k] for k in range(self.size) if int(self.dof[0, k]) not in by_dof]
            if missing:
                raise EnvError(f"joints {missing} have no actuator", "pick joints with a servo", field="joints")
            self.act = np.vectorize(lambda d: by_dof[int(d)])(self.dof)
        lo, hi = m.joint_limit_lower.numpy()[self.dof[0]], m.joint_limit_upper.numpy()[self.dof[0]]
        self.low = np.asarray(self._low if self._low is not None else lo, np.float32)
        self.high = np.asarray(self._high if self._high is not None else hi, np.float32)
        self.proprio_size = 2 * self.size if self._proprio else 0

    def _array(self, sim: Any) -> Any:
        return sim.control.joint_target_q if self.newton else sim.control.mujoco.ctrl

    def read(self, sim: Any) -> np.ndarray:
        return self._array(sim).numpy().reshape(-1)[self.act].astype(np.float32)

    def write(self, sim: Any, a: np.ndarray) -> None:
        arr = self._array(sim)
        now = arr.numpy()
        flat = now.reshape(-1).copy()
        new = np.asarray(a, dtype=flat.dtype).reshape(self.act.shape)
        if np.array_equal(flat[self.act], new):
            return
        flat[self.act] = new
        arr.assign(flat.reshape(now.shape))

    def proprio(self, sim: Any) -> np.ndarray:
        if not self._proprio:
            return np.zeros((self.q.shape[0], 0), np.float32)
        q, qd = sim.state.joint_q.numpy()[self.q], sim.state.joint_qd.numpy()[self.dof]
        return np.concatenate([q, qd], axis=1).astype(np.float32)


class KinematicBody:
    """A kinematic tool body (label suffix ``body``) the port places each period: the action is its pose (x, y, z,
    qx, qy, qz, qw) and velocity (vx, vy, vz, wx, wy, wz), as the script writes them; proprio is the same."""

    POS, VEL, SPIN = 3.0, 10.0, 200.0            # m (about the start pose), m/s, rad/s: the action box

    def __init__(self, body: str) -> None:
        self.body, self.size, self.proprio_size = body, 13, 13
        self.low = np.array([-self.POS] * 3 + [-1.0] * 4 + [-self.VEL] * 3 + [-self.SPIN] * 3, np.float32)
        self.high = -self.low

    def bind(self, sim: Any) -> None:
        m = sim.model
        self.idx = _match([str(lbl) for lbl in m.body_label], m.body_world.numpy(), self.body, "body")
        self.low, self.high = self.low.copy(), self.high.copy()
        self.low[:3], self.high[:3] = around_start(sim.state.body_q.numpy()[self.idx][:, :3], self.POS)

    def read(self, sim: Any) -> np.ndarray:
        q, qd = sim.state.body_q.numpy()[self.idx], sim.state.body_qd.numpy()[self.idx]
        return np.concatenate([q, qd], axis=1).astype(np.float32)

    def write(self, sim: Any, a: np.ndarray) -> None:
        a = np.asarray(a, np.float32).reshape(len(self.idx), 13)
        for arr, part in ((sim.state.body_q, a[:, :7]), (sim.state.body_qd, a[:, 7:])):
            now = arr.numpy()
            if np.array_equal(now[self.idx], part):
                continue
            now = now.copy()
            now[self.idx] = part
            arr.assign(now)

    def proprio(self, sim: Any) -> np.ndarray:
        return self.read(sim)


class PluginInput:
    """A plug-in's command input ``name`` (a valve, a faucet's regulator, a hand's targets), every value of it per
    world, each in ``[low, high]``. ``plugin`` names the plug-in (its label) when more than one has that input. An
    input shared by every world takes world 0's action."""

    def __init__(self, name: str, low: float = 0.0, high: float = 1.0, plugin: str | None = None) -> None:
        self.name, self.plugin, self._lo, self._hi, self.proprio_size = name, plugin, low, high, 0

    def _has(self, p: Any) -> bool:
        return self.name in p.inputs()

    def bind(self, sim: Any) -> None:
        hosts = getattr(getattr(sim, "plugins", None), "plugins", {})
        found = [p for label, p in hosts.items() if (self.plugin in (None, label)) and self._has(p)]
        if len(found) != 1:
            raise EnvError(f"{len(found)} plug-ins have the input {self.name!r}",
                           "name the plug-in (plugin=...)", field="plugin")
        self.target, self.worlds = found[0], _worlds_of(sim)
        now = self._array().numpy()
        self.shape, self.shared = now.shape, not (now.ndim and now.shape[0] == self.worlds)
        self.size = now.size if self.shared else now.size // self.worlds
        self.low = np.full(self.size, self._lo, np.float32)
        self.high = np.full(self.size, self._hi, np.float32)

    def _array(self) -> Any:
        return self.target.inputs()[self.name]

    def read(self, sim: Any) -> np.ndarray:  # noqa: ARG002
        v = self._array().numpy().astype(np.float32)
        return np.broadcast_to(v.reshape(1, -1), (self.worlds, v.size)).copy() if self.shared else \
            v.reshape(self.worlds, -1)

    def _new(self, a: np.ndarray) -> np.ndarray | None:
        """The input's new values, or None when ``a`` is what it holds (compared as the float32 the action is, so the
        expert's action keeps a float64 input exact)."""
        now = self._array().numpy()
        a = np.asarray(a, dtype=np.float32).reshape(self.worlds, self.size)
        new = (a[0] if self.shared else a).reshape(self.shape)
        return None if np.array_equal(now.astype(np.float32), new) else new.astype(now.dtype)

    def write(self, sim: Any, a: np.ndarray) -> None:  # noqa: ARG002
        new = self._new(a)
        if new is not None:
            self.target.set_input(self.name, new)

    def proprio(self, sim: Any) -> np.ndarray:  # noqa: ARG002
        return np.zeros((self.worlds, 0), np.float32)


class PluginArray(PluginInput):
    """A plug-in's array attribute ``name`` that the port's script assigns directly (not a registered input: the
    surface film's per-tool ``tool_release``, for one), every value of it per world."""

    def _has(self, p: Any) -> bool:
        return hasattr(getattr(p, self.name, None), "assign")

    def _array(self) -> Any:
        return getattr(self.target, self.name)

    def write(self, sim: Any, a: np.ndarray) -> None:  # noqa: ARG002
        new = self._new(a)
        if new is not None:
            self._array().assign(new)


class HandTargets:
    """A pose hand's commands (``alloy.tasks.rinse_hand.PoseHand``, label ``plugin``): per held body its target
    position, orientation (x, y, z, w), feed-forward velocity and grip; proprio is the held bodies' pose and velocity.
    """

    POS = 3.0                                    # m either side of where each held body starts

    def __init__(self, plugin: str = "pose_hand") -> None:
        self.plugin = plugin
        self.parts = [PluginInput("target_pos", -self.POS, self.POS, plugin),
                      PluginInput("target_quat", -1.0, 1.0, plugin),
                      PluginInput("target_vel", -5.0, 5.0, plugin), PluginInput("grip", 0.0, 1.0, plugin)]

    def bind(self, sim: Any) -> None:
        for p in self.parts:
            p.bind(sim)
        self.held = self.parts[0].target.held.numpy()                       # (W, N) body indices
        pos, start = self.parts[0], sim.state.body_q.numpy()[self.held][..., :3]   # (W, N, 3)
        if pos.size == start[0].size:                                     # one target per held body and world
            pos.low, pos.high = around_start(start, self.POS)
        self.size = sum(p.size for p in self.parts)
        self.proprio_size = 13 * self.held.shape[1]
        self.low = np.concatenate([p.low for p in self.parts])
        self.high = np.concatenate([p.high for p in self.parts])

    def read(self, sim: Any) -> np.ndarray:
        return np.concatenate([p.read(sim) for p in self.parts], axis=1)

    def write(self, sim: Any, a: np.ndarray) -> None:
        k = 0
        for p in self.parts:
            p.write(sim, a[:, k:k + p.size])
            k += p.size

    def proprio(self, sim: Any) -> np.ndarray:
        q, qd = sim.state.body_q.numpy()[self.held], sim.state.body_qd.numpy()[self.held]     # (W, N, 7 / 6)
        return np.concatenate([q, qd], axis=2).reshape(len(self.held), -1).astype(np.float32)


class Mover:
    """Parts side by side: one action vector (the parts' in order), one proprio vector."""

    def __init__(self, *parts: Any) -> None:
        self.parts = list(parts)
        self.sim: Any = None

    def bind(self, sim: Any) -> None:
        for p in self.parts:
            p.bind(sim)
        self.sim = sim
        self.size = sum(p.size for p in self.parts)
        self.proprio_size = sum(p.proprio_size for p in self.parts)
        self.low = np.concatenate([p.low for p in self.parts]).astype(np.float32)
        self.high = np.concatenate([p.high for p in self.parts]).astype(np.float32)

    def read(self, sim: Any) -> np.ndarray:
        return np.concatenate([p.read(sim) for p in self.parts], axis=1)

    def write(self, sim: Any, a: np.ndarray) -> None:
        k = 0
        for p in self.parts:
            p.write(sim, a[:, k:k + p.size])
            k += p.size

    def proprio(self, sim: Any) -> np.ndarray:
        return np.concatenate([p.proprio(sim) for p in self.parts], axis=1)


__all__ = ["HandTargets", "KinematicBody", "around_start", "Mover", "PluginArray", "PluginInput", "ServoJoints"]
