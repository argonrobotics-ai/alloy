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
- "auto" (default): "position" when the scene has joint-target actuators, else "ctrl".
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


def resolve_kind(spec: ActionSpec, layout: Layout) -> str:
    """The concrete kind for ``spec.kind == "auto"``: joint targets when the named joints have them, else ctrl."""
    if spec.kind != "auto":
        return spec.kind
    names = list(spec.joints) if spec.joints is not None else None
    if names is None:
        return "position" if layout.actuated_joints() or not layout.actuators else "ctrl"
    targeted = all(n in layout.joints and layout.joints[n].actuated for n in names)
    return "position" if targeted or not layout.actuators else "ctrl"


class ActionWriter:
    """Writes actions into ``engine.control`` for every world, and puts targets back for reset worlds."""

    def __init__(self, spec: ActionSpec, layout: Layout, engine: Any, xp: Any) -> None:
        spec.validate()
        self.spec, self.layout, self.engine, self.xp = spec, layout, engine, xp
        self.kind = resolve_kind(spec, layout)
        self.target = control_array(engine.control, self.kind)
        if self.target is None:
            raise EnvError(f"this scene has no control.{KINDS[self.kind]}", "use kind='position' with MJCF "
                           "<position> actuators, or kind='ctrl' with MuJoCo <general>/<motor> actuators", field="kind")
        default = layout.actuated_joints() if self.kind != "ctrl" else list(layout.actuators)
        self.joints = list(spec.joints) if spec.joints is not None else default
        if not self.joints:
            raise EnvError("the scene has no actuated joints to act on", "add a robot whose MJCF has actuators, or "
                           "name joints in ActionSpec(joints=[...])", field="joints")
        cols = [self._columns(j) for j in self.joints]
        self.rows_np = np.concatenate([c[0] for c in cols], axis=1)                     # (W, A)
        self.rows = xp.asarray(self.rows_np, "int64")
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
        for j in self.joints:
            w = self._columns(j)[0].shape[1]
            out += [j] if w == 1 else [f"{j}[{k}]" for k in range(w)]
        return out

    def _columns(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows (W, k), lower (k,), upper (k,)) of one named joint or actuator."""
        if self.kind == "ctrl":
            a = self.layout.actuator(name)
            return a.rows[:, None], np.array([a.lower]), np.array([a.upper])
        r = self.layout.joint(name)
        rows = r.target_q if self.kind == "position" else r.qd
        if self.kind == "position" and rows.shape[1] != len(r.lower):
            return rows, np.full(rows.shape[1], -np.inf), np.full(rows.shape[1], np.inf)
        return rows, r.lower, r.upper

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
                values = xp.clip(self.held(xp.view(self.target)[self.rows], self.window) + self.delta * a,
                                 self.low, self.high)
        xp.update(self.target, lambda t: t.__setitem__(self.rows, values))
        return values

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
        for name in self.joints:
            width = self._columns(name)[0].shape[1]
            act = self.layout.actuator(name) if self.kind == "ctrl" else None
            joint = name if act is None else (act.joint if act.position else None)
            r = self.layout.joints.get(joint) if joint is not None else None
            if r is not None and width == 1 and r.width == 1 and self.kind in ("position", "ctrl"):
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


__all__ = ["ActionSpec", "ActionWriter", "KINDS", "MODES", "WINDOW_STEPS", "control_array", "resolve_kind"]
