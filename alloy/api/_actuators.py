# SPDX-License-Identifier: Apache-2.0
"""MJCF actuators that read MuJoCo's ``ctrl`` directly (``<general>`` / ``<motor>`` and robot files that use them for
servos, such as MuJoCo Menagerie's arms) next to the ``<position>`` / ``<velocity>`` actuators Newton drives from
joint targets. ``Sim.set_targets`` writes both kinds; ``Sim.set_controls`` writes raw ``ctrl`` values by name, and
for a ``<position>`` / ``<velocity>`` actuator the joint target its ctrl stands for (``target_rows``; MuJoCo's own
``ctrl`` row is not read for those: AUDIT-STACK AGENTS-4, a value set there was silently ignored).

A ``<general>`` actuator on a joint with an affine bias ``(0, -kp, -kv)`` and gain ``kp`` is a position servo:
its ctrl is the joint position the servo holds (MuJoCo's absolute qpos, i.e. the joint target plus the joint's ref).
With bias ``(0, 0, -kv)`` and gain ``kv`` it is a velocity servo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CTRL_DIRECT = 1
TRN_JOINT = 0


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
    ctrl_span: dict[int, tuple[int, float, float]] = field(default_factory=dict)   # joint -> (position-servo row,
    #                                                 its ctrlrange lo, hi), when the file limits the ctrl (MuJoCo qpos)

    @classmethod
    def from_model(cls, model: Any, robots: dict[str, range], qd_start: np.ndarray) -> Actuators:
        """Read ``model.mujoco`` actuator attributes (an empty table without them). ``robots``: name -> joints.

        Example: ``Actuators.from_model(sim.built.model, sim.index.robot_joints, qd_start).names``."""
        mj = getattr(model, "mujoco", None)
        joint_labels = [str(x) for x in getattr(model, "joint_label", [])]
        out = cls()
        if mj is None or not hasattr(mj, "ctrl_source"):
            return out
        world = mj.actuator_world.numpy()
        rows0 = np.flatnonzero(world == 0)
        out.stride = len(rows0)
        src, trn, ttype = mj.ctrl_source.numpy(), mj.actuator_trnid.numpy(), mj.actuator_trntype.numpy()
        gain, bias = mj.actuator_gainprm.numpy(), mj.actuator_biasprm.numpy()
        labels = getattr(mj, "actuator_label", None)
        span, limited = getattr(mj, "actuator_ctrlrange", None), getattr(mj, "actuator_ctrllimited", None)
        span, limited = (None, None) if span is None or limited is None else (span.numpy(), limited.numpy())
        for k, row in enumerate(rows0):
            joint = _joint_of_dof(int(trn[row][0]), qd_start) if int(ttype[row]) == TRN_JOINT else None
            label = str(labels[row]).rsplit("/", 1)[-1] if labels is not None else ""
            if not label:                     # an unnamed MJCF actuator: named after its joint, else its position
                label = joint_labels[joint].rsplit("/", 1)[-1] if joint is not None and joint_labels else f"actuator{k}"
            out._name(row, label, joint, robots)
            if joint is None:
                continue
            out.dof_of_row[int(row)] = int(trn[row][0])
            out.on_joint.add(joint)
            kind = _servo_kind(gain[row], bias[row])
            if kind == "position" and span is not None and int(limited[row]) != 0 and span[row][0] < span[row][1]:
                out.ctrl_span[joint] = (int(row), float(span[row][0]), float(span[row][1]))   # limited: true or auto
            if int(src[row]) != CTRL_DIRECT:         # <position> / <velocity>: Newton drives it from joint targets
                out.joint_target.add(joint)
                out.target_rows[int(row)] = (joint, kind or "")
                continue
            if kind is not None:
                getattr(out, kind)[joint] = int(row)
        return out

    def _name(self, row: int, label: str, joint: int | None, robots: dict[str, range]) -> None:
        owner = next((r for r, joints in robots.items() if joint is not None and joint in joints), None)
        self.names[f"{owner}/{label}" if owner else label] = int(row)
        if owner and label not in self.names:
            self.names[label] = int(row)

    def rows(self, row: int, worlds: int) -> np.ndarray:
        """The row of a world-0 actuator in every world. Example: ``acts.rows(acts.names["arm/a1"], sim.worlds)``."""
        return row + np.arange(worlds, dtype=np.int64) * self.stride

    def actuated(self, joint: int) -> bool:
        """True if any actuator drives the template joint."""
        return joint in self.on_joint


def _joint_of_dof(dof: int, qd_start: np.ndarray) -> int | None:
    """Template joint whose dofs contain ``dof`` (world-0 dof index), or None."""
    j = int(np.searchsorted(qd_start, dof, side="right")) - 1
    return j if 0 <= j < len(qd_start) - 1 and qd_start[j] <= dof < qd_start[j + 1] else None


def _servo_kind(gain: np.ndarray, bias: np.ndarray) -> str | None:
    kp = float(gain[0])
    if kp <= 0.0:
        return None
    if abs(float(bias[1]) + kp) <= 1e-6 * kp and float(bias[0]) == 0.0:
        return "position"
    if float(bias[1]) == 0.0 and abs(float(bias[2]) + kp) <= 1e-6 * kp and float(bias[0]) == 0.0:
        return "velocity"
    return None
