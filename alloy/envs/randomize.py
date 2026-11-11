# SPDX-License-Identifier: Apache-2.0
"""Random starts: each reset moves objects and robot joints by a fresh random amount, per world.

The engine resets a world to the same start every time; these jitters make every episode start differently (the
values are drawn on the device from the environment's seed). Example::

    from alloy.envs import Task, randomize as R
    task = Task(..., randomize=[R.jitter_body("cube", pos=(0.02, 0.02, 0.0), yaw_deg=30),
                                R.jitter_joint("arm/elbow", by=0.1)])

``jitter_body`` shifts a free object (uniform in +-pos metres, and +-yaw_deg about the vertical); ``jitter_joint``
offsets a robot joint (uniform in +-by, kept inside its limits) and moves its position target (or the control of the
MuJoCo servo driving it) there too, so the servo holds the new start instead of snapping back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.envs.layout import Layout


@dataclass(frozen=True)
class Jitter:
    """One random change applied at reset. Make it with ``jitter_body`` or ``jitter_joint``."""

    name: str
    kind: str                                   # "body" or "joint"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    by: float = 0.0
    at: float | None = None                     # joints: the centre of the jitter (None: the reset value)
    carry: tuple[str, ...] = ()                 # slide joints: free bodies that move with it (a held object)
    target: float | None = None                 # joints: the servo target to set (None: the new joint value)


def jitter_body(name: str, pos: Sequence[float] = (0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> Jitter:
    """Shift a free object by up to +-pos (m) and turn it up to +-yaw_deg about the vertical at each reset.

    Example: ``jitter_body("cube", pos=(0.02, 0.02, 0), yaw_deg=45)``."""
    p = tuple(float(v) for v in pos)
    if len(p) != 3 or min(p) < 0 or yaw_deg < 0:
        raise EnvError(f"jitter_body({name!r}) needs pos=(dx, dy, dz) >= 0 and yaw_deg >= 0",
                       "pass half-ranges, e.g. pos=(0.02, 0.02, 0.0), yaw_deg=30", field="pos")
    return Jitter(name, "body", pos=p, yaw_deg=float(yaw_deg))  # type: ignore[arg-type]


def jitter_joint(name: str, by: float = 0.0, at: float | None = None, carry: Sequence[str] = (),
                 target: float | None = None) -> Jitter:
    """Start a robot joint at ``at`` (default: its reset value) plus up to +-by (m or rad), within its limits.
    ``carry``: free objects that move with a slide joint (an object the gripper holds at the start). ``target``: the
    servo target to start with (default: the joint's new value; set it past a held object to squeeze it).

    Example: ``jitter_joint("gripper/lift", by=0.01)``; a home pose: ``jitter_joint("arm/elbow", at=-1.57)``;
    a held peg: ``jitter_joint("g/x", by=0.02, carry=["peg"])``."""
    if by < 0:
        raise EnvError(f"jitter_joint({name!r}, by={by})", "pass by >= 0 (a half-range)", field="by")
    return Jitter(name, "joint", by=float(by), at=None if at is None else float(at), carry=tuple(carry),
                  target=None if target is None else float(target))


class Randomizer:
    """Applies a task's jitters to the reset worlds: writes joint coordinates and targets, then forward kinematics."""

    def __init__(self, jitters: Sequence[Jitter], layout: Layout, engine: Any, xp: Any) -> None:
        self.jitters, self.layout, self.engine, self.xp = list(jitters), layout, engine, xp
        self._plans = [self._plan(j) for j in self.jitters]
        self._joint_world = xp.asarray(np.maximum(layout.world_of.get("joint", np.zeros(0)), 0), "int64")
        self._joint_valid = xp.asarray(layout.world_of.get("joint", np.zeros(0)) >= 0, "bool")

    def _plan(self, j: Jitter) -> dict[str, Any]:
        lay, xp = self.layout, self.xp
        if j.kind == "body":
            if j.name not in lay.free:
                raise EnvError(f"{j.name!r} is not a free object", f"jitter one of {sorted(lay.free) or 'none'}; "
                               "use jitter_joint for robot joints", field="randomize")
            rot = _matrix(lay.free_frame[j.name]).T                       # world -> joint frame
            return {"q0": xp.asarray(lay.free[j.name], "int64"), "rot": xp.asarray(rot),
                    "axis": rot @ np.array([0.0, 0.0, 1.0])}
        rows = lay.joint(j.name)
        if rows.width != 1:
            raise EnvError(f"{j.name!r} has {rows.width} coordinates", "jitter 1-dof joints (hinge or slide)",
                           field="randomize")
        full = next((n for n, r in lay.joints.items() if r is rows and "/" in n), j.name)
        ctrl = [a.rows for a in lay.actuators.values() if a.joint == full]      # a MuJoCo servo driving it
        return {"q": xp.asarray(rows.q[:, 0], "int64"), "t": xp.asarray(rows.target_q[:, 0], "int64"),
                "c": xp.asarray(ctrl[0], "int64") if ctrl else None,
                "lo": float(rows.lower[0]), "hi": float(rows.upper[0]),
                "carry": [self._plan(Jitter(b, "body")) for b in j.carry],
                "axis": self._slide_axis(j.name, int(rows.qd[0, 0])) if j.carry else None}

    def _slide_axis(self, name: str, dof: int) -> np.ndarray:
        """World direction of a slide joint (world 0, at the build pose): what a carried object moves along."""
        m = self.engine.model
        j = int(np.searchsorted(m.joint_qd_start.numpy(), dof, side="right") - 1)
        if int(m.joint_type.numpy()[j]) != 0:                         # newton.JointType.PRISMATIC
            raise EnvError(f"carry= needs a slide joint, {name!r} is not one", "jitter the slide joints that carry "
                           "the object (x, y, z) with carry=[...]", field="carry")
        axis = m.joint_axis.numpy()[dof].astype(np.float64)
        rot = _matrix(m.joint_X_p.numpy()[j, 3:7])
        parent = int(m.joint_parent.numpy()[j])
        if parent >= 0:
            rot = _matrix(self.engine.state.body_q.numpy()[parent, 3:7]) @ rot
        return rot @ axis

    def apply(self, mask: Any) -> None:
        """Jitter the worlds in the bool (W,) ``mask`` (just reset). Example: ``randomizer.apply(done)``."""
        if not self.jitters:
            return
        xp, ctrl = self.xp, self.engine.control
        for j, plan in zip(self.jitters, self._plans):
            if j.kind == "body":
                xp.update(self.engine.state.joint_q, lambda q, j=j, p=plan: self._body(q, j, p, mask))
            else:
                values, delta = self._joint_values(j, plan, mask)
                xp.update(self.engine.state.joint_q, lambda q, p=plan, v=values: q.__setitem__(p["q"], v))
                for held in plan["carry"]:
                    xp.update(self.engine.state.joint_q, lambda q, h=held, p=plan, d=delta: self._shift(q, h, p, d))
                hold = values if j.target is None else values * 0.0 + j.target
                xp.update(ctrl.joint_target_q, lambda t, p=plan, v=hold: _put(xp, t, p["t"], v, mask))
                if plan["c"] is not None:
                    xp.update(ctrl.mujoco.ctrl, lambda c, p=plan, v=hold: _put(xp, c, p["c"], v, mask))
        self._forward_kinematics(mask)

    def _body(self, q: Any, j: Jitter, plan: dict[str, Any], mask: Any) -> None:
        xp, w = self.xp, self.layout.worlds
        m = xp.float(mask)[:, None]
        half = xp.asarray(j.pos)
        d_world = xp.uniform((w, 3), -1.0, 1.0) * half[None, :] * m
        d_joint = (plan["rot"][None, :, :] * d_world[:, None, :]).sum(-1)    # R^T d per world
        rows = plan["q0"][:, None] + xp.asarray(np.arange(3), "int64")[None, :]
        q[rows] = q[rows] + d_joint
        if j.yaw_deg:
            angle = xp.uniform((w,), -1.0, 1.0) * math.radians(j.yaw_deg) * m[:, 0]
            qrows = plan["q0"][:, None] + xp.asarray(np.arange(3, 7), "int64")[None, :]
            q[qrows] = _quat_mul(xp, _axis_angle(xp, plan["axis"], angle), q[qrows])

    def _joint_values(self, j: Jitter, plan: dict[str, Any], mask: Any) -> Any:
        xp = self.xp
        current = xp.view(self.engine.state.joint_q)[plan["q"]]
        centre = current if j.at is None else current * 0.0 + j.at
        noise = xp.uniform((self.layout.worlds,), -j.by, j.by) if j.by else 0.0
        new = xp.where(mask, xp.clip(centre + noise, plan["lo"], plan["hi"]), current)
        return new, new - current

    def _shift(self, q: Any, held: dict[str, Any], plan: dict[str, Any], delta: Any) -> None:
        """Move a carried free body by ``delta`` (W,) along the slide joint's world axis."""
        xp = self.xp
        d_world = delta[:, None] * xp.asarray(plan["axis"])[None, :]
        d_joint = (held["rot"][None, :, :] * d_world[:, None, :]).sum(-1)
        rows = held["q0"][:, None] + xp.asarray(np.arange(3), "int64")[None, :]
        q[rows] = q[rows] + d_joint

    def _forward_kinematics(self, mask: Any) -> None:
        """Body poses of the masked worlds from their (jittered) joint coordinates; other worlds are untouched."""
        from alloy._vendor import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        eng = self.engine
        sel = mask[self._joint_world] & self._joint_valid
        with wp.ScopedDevice(eng.model.device):
            newton.eval_fk(eng.model, eng.state.joint_q, eng.state.joint_qd, eng.state,
                           joint_mask=self.xp.to_warp_bool(sel))


def _put(xp: Any, arr: Any, rows: Any, values: Any, mask: Any) -> None:
    """``arr[rows] = values`` for the masked worlds only (the others keep their running targets)."""
    arr[rows] = xp.where(mask, values, arr[rows])


def _matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(c) for c in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_angle(xp: Any, axis: np.ndarray, angle: Any) -> Any:
    """(W, 4) xyzw quaternions for rotations by ``angle`` (W,) about a fixed unit ``axis``."""
    s, c = _sin_cos(xp, angle * 0.5)
    a = xp.asarray(axis / np.linalg.norm(axis))
    return xp.cat([a[None, :] * s[:, None], c[:, None]], -1)


def _sin_cos(xp: Any, x: Any) -> tuple[Any, Any]:
    if xp.name == "torch":
        return xp.torch.sin(x), xp.torch.cos(x)
    return np.sin(x), np.cos(x)


def _quat_mul(xp: Any, a: Any, b: Any) -> Any:
    """Hamilton product a * b of (W, 4) xyzw quaternions."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return xp.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


__all__ = ["Jitter", "Randomizer", "jitter_body", "jitter_joint"]
