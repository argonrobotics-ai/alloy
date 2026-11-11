# SPDX-License-Identifier: Apache-2.0
"""Joint targets against the robot's own numbers: the joint's range and whether anything drives the joint.

- **Range.** ``Sim.set_targets`` (and ``run(targets=)``) hold a position target at the joint's range. MuJoCo's joint
  limits are soft, so a stiff servo sent past one rested past it: a hinge of range +-1 rad with ``<position kp="300">``
  and a target of 3 rad stopped at 1.32 rad, health OK (AUDIT-STACK API-3), while the docs said targets stop at the
  limit. A position servo whose file gives it a ``ctrlrange`` wider than the joint's range keeps that range (WP13's
  review: a gripper servo that squeezes by aiming past its closed stop; the error past the stop is the grip force).
  The first target held back for a joint is told to the user (``_say.warn``).
- **Driven.** A robot whose file has no ``<actuator>`` block has no actuator table, and its targets went to Newton's
  joint target arrays, which nothing reads when the joint's own gains are zero: the call returned and the arm fell
  (AUDIT-STACK AGENTS-3 / ERR-4). ``driven`` is False for such a joint, and ``set_targets`` raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api import _say


@dataclass
class Limits:
    """Per template dof (world 0): the joint range and Newton's own servo gains."""

    lower: np.ndarray
    upper: np.ndarray
    ke: np.ndarray
    kd: np.ndarray
    qd_start: np.ndarray
    told: set[str] = field(default_factory=set)

    @classmethod
    def from_model(cls, model: Any, qd_start: np.ndarray) -> Limits:
        """Read the model's joint limits and target gains. Example: ``Limits.from_model(sim.built.model, qd_start)``."""
        def host(name: str, fill: float) -> np.ndarray:
            arr = getattr(model, name, None)
            return arr.numpy().astype(np.float64) if arr is not None and len(arr) else np.full(1, fill)

        return cls(host("joint_limit_lower", -np.inf), host("joint_limit_upper", np.inf), host("joint_target_ke", 0.0),
                   host("joint_target_kd", 0.0), np.asarray(qd_start))

    def _dof(self, joint: int) -> int:
        return int(self.qd_start[joint])

    def clamp(self, name: str, joint: int, values: np.ndarray, ctrl: tuple[float, float] | None = None) -> np.ndarray:
        """``values`` (worlds,) held inside the joint's range, each world's target on its own (the range is the robot
        file's, one for every world); a joint without a range (or one whose lower limit is not below its upper)
        passes through. ``ctrl``: the position servo's own ctrl range in the targets' frame, when the file gives
        one; the wider of the two holds. Example: ``limits.clamp("elbow", j, vals)``."""
        d = self._dof(joint)
        if d >= len(self.lower) or d >= len(self.upper) or not self.lower[d] < self.upper[d]:
            return values
        lo, hi = float(self.lower[d]), float(self.upper[d])
        if ctrl is not None:
            lo, hi = min(lo, ctrl[0]), max(hi, ctrl[1])
        out = np.clip(values, lo, hi)
        if name not in self.told and not np.array_equal(out, values):
            self.told.add(name)
            _say.warn("target for joint %r is outside its range [%.4g, %.4g]: held at the limit (set_targets keeps "
                      "position targets inside the joint's range, or the servo's ctrlrange when that is wider)",
                      name, lo, hi)
        return out

    def driven(self, joint: int, kind: str, actuators: int) -> bool:
        """Whether Newton's own joint target moves the joint when the model has no actuator table (``actuators`` 0):
        its position gain for a position target, its velocity gain for a velocity one.
        Example: ``limits.driven(j, "position", 0)``."""
        d = self._dof(joint)
        gain = self.ke if kind == "position" else self.kd
        return not actuators and d < len(gain) and float(gain[d]) > 0.0


def write(array: Any, rows: np.ndarray, values: np.ndarray) -> None:
    """Assign ``values`` to ``rows`` of a Warp array through a host copy.

    Example: ``write(ctrl.joint_target_q, rows, v)``."""
    host = array.numpy()
    host[rows] = values
    array.assign(host)
