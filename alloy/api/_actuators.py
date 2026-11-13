# SPDX-License-Identifier: Apache-2.0
"""MJCF actuators that read MuJoCo's ``ctrl`` directly (``<general>`` / ``<motor>`` and robot files that use them for
servos, such as MuJoCo Menagerie's arms) next to the ``<position>`` / ``<velocity>`` actuators Newton drives from
joint targets. ``Sim.set_targets`` writes both kinds; ``Sim.set_controls`` writes raw ``ctrl`` values by name, and
for a ``<position>`` / ``<velocity>`` actuator the joint target its ctrl stands for (``target_rows``; MuJoCo's own
``ctrl`` row is not read for those, so a value set there would be ignored).

A ``<general>`` actuator on a joint with an affine bias ``(0, -kp, -kv)`` and gain ``kp`` is a position servo:
its ctrl is the joint position the servo holds (MuJoCo's absolute qpos, i.e. the joint target plus the joint's ref).
With bias ``(0, 0, -kv)`` and gain ``kv`` it is a velocity servo. An ``<intvelocity>`` (a position servo whose
activation integrates ctrl) takes a velocity: its ctrl is the rate of the position it holds. A ``gear`` g scales
the joint: MuJoCo's actuator length is g x the joint value, so its ctrl is g x the joint position or velocity.

Names: an actuator is "<robot>/<name in its file>" (the name may hold "/", as in MuJoCo Menagerie's two-arm files),
its owner read from its joint, or for a tendon or site actuator from its articulation; the bare name works too
when exactly one robot has it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CTRL_DIRECT = 1
TRN_JOINT = 0
DYN_INTEGRATOR = 1          # MuJoCo's mjDYN_INTEGRATOR: the activation integrates ctrl


@dataclass
class Actuators:
    """World-0 actuator rows by name, the per-world row stride, and the joint servos among them."""

    names: dict[str, int] = field(default_factory=dict)          # "<robot>/<label>" (and a unique bare label) -> row
    position: dict[int, int] = field(default_factory=dict)       # template joint -> CTRL_DIRECT position-servo row
    velocity: dict[int, int] = field(default_factory=dict)       # template joint -> CTRL_DIRECT velocity-servo row
    joint_target: set[int] = field(default_factory=set)          # template joints driven from joint targets
    stride: int = 0                                              # rows per world
    dof_of_row: dict[int, int] = field(default_factory=dict)     # world-0 row -> template dof
    on_joint: set[int] = field(default_factory=set)              # template joints with any actuator
    target_rows: dict[int, tuple[int, str]] = field(default_factory=dict)   # joint-target row -> (joint, servo kind)
    target_kinds: dict[int, set[str]] = field(default_factory=dict)  # joint -> kinds of its joint-target servos
    ctrl_span: dict[int, tuple[int, float, float]] = field(default_factory=dict)   # joint -> (position-servo row,
    #                                                 its ctrlrange lo, hi), when the file limits the ctrl (MuJoCo qpos)
    gear: dict[int, float] = field(default_factory=dict)         # world-0 row -> its joint gear, where it is not 1
    ambiguous: dict[str, list[str]] = field(default_factory=dict)   # bare label in several robots -> full names
    # per template dof (world 0): a body with several MJCF joints loads as one joint of several dofs, each its own
    position_dof: dict[int, int] = field(default_factory=dict)   # dof -> CTRL_DIRECT position-servo row
    velocity_dof: dict[int, int] = field(default_factory=dict)   # dof -> CTRL_DIRECT velocity-servo row
    target_dofs: dict[int, set[str]] = field(default_factory=dict)   # dof -> kinds of its joint-target servos
    span_dof: dict[int, tuple[int, float, float]] = field(default_factory=dict)   # dof -> ctrl_span's tuple

    @classmethod
    def from_model(cls, model: Any, robots: dict[str, range], qd_start: np.ndarray,
                   joint_names: Sequence[str] | None = None) -> Actuators:
        """Read ``model.mujoco`` actuator attributes (an empty table without them). ``robots``: name -> joints;
        ``joint_names``: each joint's name in its file (names an unnamed actuator; default: the label's last part).

        Example: ``Actuators.from_model(sim.built.model, sim.index.robot_joints, qd_start).names``."""
        mj = getattr(model, "mujoco", None)
        out = cls()
        if mj is None or not hasattr(mj, "ctrl_source"):
            return out
        cols = _Columns.read(model, mj)
        rows0 = np.flatnonzero(cols.world == 0)
        out.stride = len(rows0)
        if joint_names is None:
            joint_names = [str(x).rsplit("/", 1)[-1] for x in getattr(model, "joint_label", [])]
        owned: list[tuple[str, str]] = []
        for k, row in enumerate(rows0):
            joint = _joint_of_dof(int(cols.trn[row][0]), qd_start) if int(cols.ttype[row]) == TRN_JOINT else None
            label = cols.label(row) or (joint_names[joint] if joint is not None and joint_names else f"actuator{k}")
            owner = cols.owner(int(row), joint, robots)
            out.names[f"{owner}/{label}" if owner else label] = int(row)
            if owner:
                owned.append((label, f"{owner}/{label}"))
            if joint is not None:
                out._servo(cols, int(row), joint)
        out._bare_names(owned)
        return out

    def _servo(self, cols: _Columns, row: int, joint: int) -> None:
        dof = self.dof_of_row[row] = int(cols.trn[row][0])
        self.on_joint.add(joint)
        kind = _servo_kind(cols.gain[row], cols.bias[row], cols.dyntype(row))
        if cols.gear_of(row) != 1.0:
            self.gear[row] = cols.gear_of(row)
        if kind == "position" and cols.limited(row):
            span = (row, float(cols.span[row][0]), float(cols.span[row][1]))      # limited: true / auto
            self.ctrl_span[joint] = self.span_dof[dof] = span
        if int(cols.src[row]) != CTRL_DIRECT:         # <position> / <velocity>: Newton drives it from joint targets
            self.joint_target.add(joint)
            self.target_rows[row] = (joint, kind or "")
            kinds = self.target_dofs.setdefault(dof, set())
            if kind:
                self.target_kinds.setdefault(joint, set()).add(kind)
                kinds.add(kind)
            return
        if kind is not None:
            getattr(self, kind)[joint] = row
            getattr(self, f"{kind}_dof")[dof] = row

    def _bare_names(self, owned: list[tuple[str, str]]) -> None:
        """A bare label names its actuator when one robot has it; in several robots it is ambiguous (no binding)."""
        by_label: dict[str, list[str]] = {}
        for label, full in owned:
            by_label.setdefault(label, []).append(full)
        for label, fulls in by_label.items():
            if label in self.names:
                continue
            if len(fulls) == 1:
                self.names[label] = self.names[fulls[0]]
            else:
                self.ambiguous[label] = fulls

    def rows(self, row: int, worlds: int) -> np.ndarray:
        """The row of a world-0 actuator in every world. Example: ``acts.rows(acts.names["arm/a1"], sim.worlds)``."""
        return row + np.arange(worlds, dtype=np.int64) * self.stride

    def actuated(self, joint: int) -> bool:
        """True if any actuator drives the template joint."""
        return joint in self.on_joint

    def gear_of(self, row: int) -> float:
        """The joint gear of a world-0 row (1 when the file gives none). Example: ``acts.gear_of(row)``."""
        return self.gear.get(row, 1.0)


@dataclass
class _Columns:
    """The actuator attributes ``Actuators.from_model`` reads, as host arrays (optional ones None)."""

    world: np.ndarray
    src: np.ndarray
    trn: np.ndarray
    ttype: np.ndarray
    gain: np.ndarray
    bias: np.ndarray
    labels: Any = None
    dyn: np.ndarray | None = None
    gears: np.ndarray | None = None
    span: np.ndarray | None = None
    lim: np.ndarray | None = None
    art: np.ndarray | None = None
    art_start: np.ndarray | None = None

    @classmethod
    def read(cls, model: Any, mj: Any) -> _Columns:
        def host(obj: Any, name: str) -> np.ndarray | None:
            arr = getattr(obj, name, None)
            return None if arr is None else np.asarray(arr.numpy())

        return cls(host(mj, "actuator_world"), host(mj, "ctrl_source"), host(mj, "actuator_trnid"),
                   host(mj, "actuator_trntype"), host(mj, "actuator_gainprm"), host(mj, "actuator_biasprm"),
                   getattr(mj, "actuator_label", None), host(mj, "actuator_dyntype"), host(mj, "actuator_gear"),
                   host(mj, "actuator_ctrlrange"), host(mj, "actuator_ctrllimited"), host(mj, "actuator_articulation"),
                   host(model, "articulation_start"))

    def label(self, row: int) -> str:
        raw = str(self.labels[row]) if self.labels is not None else ""
        return raw.rsplit("/", 1)[-1] if raw.startswith("/") else raw      # a USD prim path: its last part

    def owner(self, row: int, joint: int | None, robots: dict[str, range]) -> str | None:
        """The robot whose joints hold the actuator's joint, or for a tendon / site actuator its articulation's."""
        if joint is None and self.art is not None and self.art_start is not None:
            a = int(self.art[row])
            joint = int(self.art_start[a]) if 0 <= a < len(self.art_start) - 1 else None
        return next((r for r, joints in robots.items() if joint is not None and joint in joints), None)

    def dyntype(self, row: int) -> int:
        return int(self.dyn[row]) if self.dyn is not None else 0

    def gear_of(self, row: int) -> float:
        g = float(np.ravel(self.gears[row])[0]) if self.gears is not None else 1.0
        return g if g != 0.0 else 1.0

    def limited(self, row: int) -> bool:
        return (self.span is not None and self.lim is not None and int(self.lim[row]) != 0
                and self.span[row][0] < self.span[row][1])


def _joint_of_dof(dof: int, qd_start: np.ndarray) -> int | None:
    """Template joint whose dofs contain ``dof`` (world-0 dof index), or None."""
    j = int(np.searchsorted(qd_start, dof, side="right")) - 1
    return j if 0 <= j < len(qd_start) - 1 and qd_start[j] <= dof < qd_start[j + 1] else None


def _servo_kind(gain: np.ndarray, bias: np.ndarray, dyntype: int = 0) -> str | None:
    """"position", "velocity" or None from the gains; an integrator's position servo (``<intvelocity>``) takes a
    velocity, since its ctrl is the rate of the position it holds."""
    kp = float(gain[0])
    if kp <= 0.0:
        return None
    kind = None
    if abs(float(bias[1]) + kp) <= 1e-6 * kp and float(bias[0]) == 0.0:
        kind = "position"
    elif float(bias[1]) == 0.0 and abs(float(bias[2]) + kp) <= 1e-6 * kp and float(bias[0]) == 0.0:
        kind = "velocity"
    if dyntype == DYN_INTEGRATOR:
        return "velocity" if kind == "position" else None
    return kind
