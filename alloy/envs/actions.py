# SPDX-License-Identifier: Apache-2.0
"""Actions: how the policy's (W, A) array drives the robots (zero-copy into the engine's control arrays).

Example::

    ActionSpec()                                            # every actuator the robots have, [-1, 1]
    ActionSpec(["gripper/lift"], mode="delta", delta=0.01)  # move the lift target by up to 1 cm per step
    ActionSpec(["arm/elbow"], kind="velocity", low=[-2], high=[2], mode="absolute")

Modes: "normalized" maps [-1, 1] onto [low, high] (the joint limits or actuator control range unless given);
"absolute" takes targets in metres / radians (clipped to [low, high]); "delta" adds ``delta * action`` (action in
[-1, 1]) to the current target, first kept within ``window`` of the joint's measured position: a blocked joint (a
finger on a table, a door at its stop) then does not wind its target up, and its servo pushes with at most about
kp x ``window``.

Kinds:
- "position": joint position targets (MJCF ``<position>`` actuators);
- "velocity": joint velocity targets (``<velocity>``);
- "effort": joint force / torque;
- "ctrl": MuJoCo-native actuators (``<general>``, ``<motor>``, ...) by "robot/actuator" or the joint they drive; a
  ``<general>`` position servo's control is its joint target;
- "auto" (default): each joint or actuator its own, so one action can drive both: joint targets where the joint has
  an MJCF ``<position>`` actuator, MuJoCo ctrl for the rest (``<general>``, ``<motor>``); with no names, every
  actuated joint and every MuJoCo-native actuator of the scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.envs.layout import UNBOUNDED, Layout

#: action kind -> control array (a dotted path under ``engine.control``)
KINDS = {"position": "joint_target_q", "velocity": "joint_target_qd", "effort": "joint_f", "ctrl": "mujoco.ctrl"}
MODES = ("normalized", "absolute", "delta")
#: mode "delta" without ``window``: a target stays within this many ``delta`` steps of its joint's measured position
WINDOW_STEPS = 5.0


@dataclass
class ActionSpec:
    """Which joints / actuators the action drives and how. Example: ``ActionSpec(["gripper/lift"], mode="delta")``."""

    joints: Sequence[str] | None = None        # None: every actuated joint, or every MuJoCo-native actuator
    kind: str = "auto"
    mode: str = "normalized"
    low: Sequence[float] | None = None         # per action column; default: joint limits / actuator ctrlrange
    high: Sequence[float] | None = None
    delta: float | Sequence[float] = 0.02      # mode "delta": target change at action +-1, per step (per column)
    window: float | Sequence[float] | None = None   # mode "delta": target kept within this of the measured joint
    #                                            (per column; None: WINDOW_STEPS x delta; float("inf"): no window)

    def validate(self) -> None:
        """Raise ``EnvError`` on an unknown kind / mode. Example: ``ActionSpec(kind="force").validate()`` raises."""
        if self.kind not in (*KINDS, "auto"):
            raise EnvError(f"unknown action kind {self.kind!r}", f"use one of {(*KINDS, 'auto')}", field="kind")
        if self.mode not in MODES:
            raise EnvError(f"unknown action mode {self.mode!r}", f"use one of {MODES}", field="mode")
        if self.mode == "delta" and self.kind == "effort":
            raise EnvError("mode='delta' with kind='effort'", "use mode='normalized' or 'absolute' for efforts",
                           field="mode")
        if self.window is not None and not np.all(np.asarray(self.window, np.float64) > 0.0):
            raise EnvError(f"window is {self.window!r}", "pass a window > 0 per column (float('inf') for none), or "
                           "None for WINDOW_STEPS x delta", field="window")


def control_array(control: Any, kind: str) -> Any:
    """The engine control array an action kind writes (None if the scene has none). Example: internal."""
    obj = control
    for part in KINDS[kind].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def default_names(spec: ActionSpec, layout: Layout) -> list[str]:
    """What ``ActionSpec(joints=None)`` drives: every actuated joint (kind "position", "velocity", "effort"), every
    MuJoCo-native actuator ("ctrl"), or both ("auto"; an actuator on an actuated joint once, as the joint)."""
    joints = layout.actuated_joints()
    if spec.kind == "ctrl":
        return list(layout.actuators)
    if spec.kind != "auto":
        return joints
    return joints + [n for n, a in layout.actuators.items() if a.joint not in joints]


def resolve_kinds(spec: ActionSpec, layout: Layout, names: Sequence[str]) -> list[str]:
    """The kind of each named joint or actuator: ``spec.kind``, or for "auto" joint targets where the joint has a
    joint-target actuator and MuJoCo ctrl otherwise. Example: ``resolve_kinds(ActionSpec(), lay, ["arm/x"])``."""
    if spec.kind != "auto":
        return [spec.kind] * len(names)
    return ["position" if (n in layout.joints and layout.joints[n].actuated) or not layout.actuators else "ctrl"
            for n in names]


def resolve_kind(spec: ActionSpec, layout: Layout) -> str:
    """The one kind ``spec`` writes, or "mixed" when "auto" picks joint targets for some columns and ctrl for others.
    Example: ``resolve_kind(ActionSpec(), env.layout)``."""
    names = list(spec.joints) if spec.joints is not None else default_names(spec, layout)
    kinds = set(resolve_kinds(spec, layout, names)) or {"position"}
    return kinds.pop() if len(kinds) == 1 else "mixed"


class ActionWriter:
    """Writes actions into ``engine.control`` for every world, and puts targets back for reset worlds."""

    def __init__(self, spec: ActionSpec, layout: Layout, engine: Any, xp: Any) -> None:
        spec.validate()
        self.spec, self.layout, self.engine, self.xp = spec, layout, engine, xp
        self.joints = list(spec.joints) if spec.joints is not None else default_names(spec, layout)
        if not self.joints:
            raise EnvError("the scene has no actuated joints to act on", "add a robot whose MJCF has actuators, or "
                           "name joints in ActionSpec(joints=[...])", field="joints")
        self.kinds = resolve_kinds(spec, layout, self.joints)                  # per named joint / actuator
        self.kind = self.kinds[0] if len(set(self.kinds)) == 1 else "mixed"
        cols = [self._columns(j, k) for j, k in zip(self.joints, self.kinds)]
        self.rows_np = np.concatenate([c[0] for c in cols], axis=1)                     # (W, A), per column's array
        self.groups = self._groups([k for c, k in zip(cols, self.kinds) for _ in range(c[0].shape[1])])
        single = len(self.groups) == 1                      # one control array (every kind but "mixed")
        self.target = self.groups[0][1] if single else None
        self.rows = self.groups[0][2] if single else None
        lo, hi = self._limits(np.concatenate([c[1] for c in cols]), np.concatenate([c[2] for c in cols]))
        self.low, self.high, self.low_np, self.high_np = xp.asarray(lo), xp.asarray(hi), lo, hi
        self.delta = xp.asarray(np.broadcast_to(np.asarray(spec.delta, np.float32), (self.size,)).copy())
        self.q_rows_np, self.measured_np = self._measured_rows()           # (W, A) rows, (A,) has a joint
        self.q_rows = xp.asarray(self.q_rows_np, "int64")
        self.window = xp.asarray(self._window(self.measured_np))
        self._initial = {k: xp.copy(xp.view(a)) for k in KINDS
                         if (a := control_array(engine.control, k)) is not None and a.shape[0]}
        self._world = self._target_worlds()

    @property
    def size(self) -> int:
        """Action columns per world. Example: ``writer.size``."""
        return int(self.rows_np.shape[1])

    def column_names(self) -> list[str]:
        """One name per action column ("robot/joint", or "robot/joint[k]" for multi-dof joints)."""
        out = []
        for j, kind in zip(self.joints, self.kinds):
            w = self._columns(j, kind)[0].shape[1]
            out += [j] if w == 1 else [f"{j}[{k}]" for k in range(w)]
        return out

    def _groups(self, col_kinds: list[str]) -> list[tuple[str, Any, Any, Any]]:
        """Per control array written: (kind, array, rows (W, a), its columns (a,), or None for every column)."""
        out = []
        for kind in dict.fromkeys(col_kinds):
            target = control_array(self.engine.control, kind)
            if target is None:
                raise EnvError(f"this scene has no control.{KINDS[kind]}", "use kind='position' with MJCF "
                               "<position> actuators, or kind='ctrl' with MuJoCo <general>/<motor> actuators",
                               field="kind")
            cols = np.flatnonzero(np.asarray(col_kinds) == kind)
            every = len(cols) == len(col_kinds)
            out.append((kind, target, self.xp.asarray(self.rows_np[:, cols], "int64"),
                        None if every else self.xp.asarray(cols, "int64")))
        return out

    def _columns(self, name: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows (W, k), lower (k,), upper (k,)) of one named joint or actuator."""
        if kind == "ctrl":
            a = self.layout.actuator(name)
            return a.rows[:, None], np.array([a.lower]), np.array([a.upper])
        r = self.layout.joint(name)
        if kind in ("position", "velocity") and not r.actuated and self._ctrl_driven(name):
            raise EnvError(f"{name!r} is driven by a MuJoCo actuator (<general>, <motor>), not by a joint target, so "
                           f"kind={kind!r} would not move it", "use ActionSpec(kind='auto') (the default) or "
                           "kind='ctrl' for it", field="kind")
        rows = r.target_q if kind == "position" else r.qd
        if kind == "position" and rows.shape[1] != len(r.lower):
            return rows, np.full(rows.shape[1], -np.inf), np.full(rows.shape[1], np.inf)
        return rows, r.lower, r.upper

    def _ctrl_driven(self, joint: str) -> bool:
        """Whether a MuJoCo-native actuator drives ``joint``."""
        return any(a.joint is not None and joint in (a.joint, a.joint.split("/", 1)[1])
                   for a in self.layout.actuators.values())

    def _limits(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, s = self.size, self.spec
        if s.low is not None and s.high is not None:
            lo, hi = np.asarray(s.low, np.float32).reshape(-1), np.asarray(s.high, np.float32).reshape(-1)
            if lo.shape != (n,) or hi.shape != (n,) or np.any(hi <= lo):
                raise EnvError(f"low/high need {n} values each with high > low", f"pass {n} numbers per bound "
                               f"(columns: {self.column_names()})", field="low")
            return lo, hi
        if self.kind in ("velocity", "effort"):
            raise EnvError(f"kind={self.kind!r} needs explicit low= and high=", "pass ActionSpec(low=[...], "
                           f"high=[...]) with {n} values (columns: {self.column_names()})", field="low")
        unbounded = ~(np.isfinite(lo) & np.isfinite(hi)) | (np.abs(lo) >= UNBOUNDED) | (np.abs(hi) >= UNBOUNDED)
        if np.any(unbounded):
            free = [c for c, u in zip(self.column_names(), unbounded) if u]
            raise EnvError(f"joints without limits: {free}", "give the joints a range in the robot file, or pass "
                           "ActionSpec(low=[...], high=[...])", field="joints")
        return lo.astype(np.float32), hi.astype(np.float32)

    # --- stepping --------------------------------------------------------------------------------------------------

    def apply(self, actions: Any) -> Any:
        """Write (W, A) actions as targets; returns the targets written (W, A). Example: ``writer.apply(a)``."""
        xp, s = self.xp, self.spec
        a = self._check(actions)
        if s.mode == "absolute":
            values = xp.clip(a, self.low, self.high)
        else:
            a = xp.clip(a, -1.0, 1.0)
            if s.mode == "normalized":
                values = self.low + (a + 1.0) * 0.5 * (self.high - self.low)
            else:
                values = xp.clip(self.held(self.current(), self.window) + self.delta * a, self.low, self.high)
        for _kind, target, rows, cols in self.groups:
            part = values if cols is None else values[:, cols]
            xp.update(target, lambda t, rows=rows, part=part: t.__setitem__(rows, part))
        return values

    def current(self) -> Any:
        """(W, A) the targets (or ctrl values) in force now, one column per action column. Example: internal."""
        xp = self.xp
        if len(self.groups) == 1:
            _kind, target, rows, _cols = self.groups[0]
            return xp.view(target)[rows]
        out = xp.zeros((self.layout.worlds, self.size))
        for _kind, target, rows, cols in self.groups:
            out[:, cols] = xp.view(target)[rows]
        return out

    def held(self, current: Any, window: Any) -> Any:
        """(W, A) targets ``current`` kept within ``window`` (A,) of each column's measured joint position (an
        infinite window, or a column with no one-coordinate joint, leaves it as is, bit for bit).

        Example: ``writer.held(targets, writer.window)``."""
        xp = self.xp
        q = xp.view(self.engine.state.joint_q)[self.q_rows]
        return xp.clip(current, q - window, q + window)

    def _measured_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """(W, A) rows into ``state.joint_q`` of each column's joint, and (A,) whether it has one (one-coordinate
        joints: hinges and slides; a ``ctrl`` column needs a position servo on a joint, not a ``<motor>``)."""
        rows, has = np.zeros(self.rows_np.shape, np.int64), np.zeros(self.size, bool)
        col = 0
        for name, kind in zip(self.joints, self.kinds):
            width = self._columns(name, kind)[0].shape[1]
            act = self.layout.actuator(name) if kind == "ctrl" else None
            joint = name if act is None else (act.joint if act.position else None)
            r = self.layout.joints.get(joint) if joint is not None else None
            if r is not None and width == 1 and r.width == 1 and kind in ("position", "ctrl"):
                rows[:, col], has[col] = r.q[:, 0], True
            col += width
        return rows, has

    def _window(self, measured: np.ndarray) -> np.ndarray:
        """(A,) window per column: the spec's, or ``WINDOW_STEPS`` x delta; infinite where nothing is measured."""
        s = self.spec
        if s.mode != "delta":
            return np.full(self.size, np.inf, np.float32)
        w = (WINDOW_STEPS * np.asarray(s.delta, np.float64)) if s.window is None else np.asarray(s.window, np.float64)
        return np.where(measured, np.broadcast_to(w, (self.size,)), np.inf).astype(np.float32)

    def _check(self, actions: Any) -> Any:
        xp, w = self.xp, self.layout.worlds
        shape = tuple(getattr(actions, "shape", np.shape(actions)))
        if shape != (w, self.size):
            raise EnvError(f"actions have shape {shape}", f"pass an array of shape ({w}, {self.size}): one row per "
                           f"world, columns {self.column_names()}", field="actions")
        return xp.asarray(actions, "float32")

    def reset(self, mask: Any) -> None:
        """Put the build-time targets back for the worlds in the bool (W,) ``mask``. Example: ``writer.reset(done)``."""
        xp = self.xp
        for kind, (rows, world) in self._world.items():
            init = self._initial[kind][rows]

            def put(t: Any, rows: Any = rows, world: Any = world, init: Any = init) -> None:
                t[rows] = xp.where(mask[world], init, t[rows])
            xp.update(control_array(self.engine.control, kind), put)

    def _target_worlds(self) -> dict[str, tuple[Any, Any]]:
        """Per control array: the rows that belong to a world, and that world (rows shared by all worlds stay)."""
        lay, out = self.layout, {}
        for kind, init in self._initial.items():
            ids = np.full(int(init.shape[0]), -1, np.int64)
            if kind == "position":
                for r in lay.joints.values():
                    ids[r.target_q] = np.arange(lay.worlds)[:, None]
            elif kind == "ctrl":
                ids[:] = lay.world_of.get("ctrl", ids)
            elif len(ids) == len(lay.world_of["qd"]):
                ids[:] = lay.world_of["qd"]
            rows = np.flatnonzero(ids >= 0)
            out[kind] = (self.xp.asarray(rows, "int64"), self.xp.asarray(ids[rows], "int64"))
        return out


__all__ = ["ActionSpec", "ActionWriter", "KINDS", "MODES", "WINDOW_STEPS", "control_array", "default_names",
           "resolve_kind", "resolve_kinds"]
