# SPDX-License-Identifier: Apache-2.0
"""The robot facts of the rigid step, read from the template builder (``alloy.scene.rigid_step_robots``).

Where each number comes from (Newton's MJCF, URDF and USD importers fill them; nothing is compiled yet):
- servo gains: ``joint_target_ke`` of a dof in a position mode (MJCF ``<position>``), and the MuJoCo actuator rows
  ``mujoco:actuator_gainprm / biasprm`` of an affine position servo on a joint (``<position>`` again, and ``<general>``
  as MuJoCo Menagerie writes it: fixed gain kp, affine bias with biasprm[1] = -kp), times gear squared. A servo on a
  tendon, site or body is not checked.
- damping: the servo's kv (``joint_target_kd``, or -biasprm[2] of its actuator row; a positive biasprm[2] is a
  damping ratio the file asked for, kv = 2 ratio sqrt(kp I)) plus the dof's ``joint_damping``. A separate velocity
  actuator on the same joint is not counted.
- joint inertia: the joint's own diagonal entry of the joint-space inertia at the template pose: every body below
  the joint about its axis (parallel-axis theorem; total mass for a slide), plus the dof's armature. With the other
  joints free to move the inertia a servo feels is smaller, so this is the frequency with the other joints held.
- time constants: a shape's ``mujoco:solref`` when the file wrote one (mode RAW), else 2 / kd of its Newton gains
  (MuJoCo's default 20 ms for an MJCF shape); a limited dof's ``mujoco:solreflimit`` (RAW) or MuJoCo's default 20 ms;
  each equality row's ``mujoco:eq_solref``. Rows written as direct stiffness (negative solref) and limits written as
  force-space gains (URDF, USD) are not clamped by the step in the same way and are not checked.
- loose bodies: a body of an MJCF robot file on a free joint with no jointed body below it (the light-body rule's
  scope); its shapes' rows go to the "loose_body" cap with the body's mass and the shape's impedance (solimp dmax).
- link speed: the corners of each collision shape's box below a joint, at the declared joint speed or the file's
  joint velocity limit when lower, against the robot's ``contact_gap`` (5 mm by default).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.contact.light_body import GEOM_FILE
from alloy.core.defaults import RIGID_GAP_M
from alloy.scene import rigid_step_bodies as RB
from alloy.scene import rigid_step_robots as RR
from alloy.scene.template import quat_matrix

JT = newton.JointType
SOLREF_RAW, SOLREF_FILE_DEFAULT = 1, 2      # Newton's mujoco solref modes (0: from the Newton gains)
DEFAULT_TAU = 0.02                          # s: MuJoCo's default solref time constant
DEFAULT_DMAX = 0.95                         # MuJoCo's default solimp dmax
UNLIMITED = 1.0e9                           # joint limits beyond this are none (Newton writes 1e10)
POSITION_MODES = (1, 3)                     # JointTargetMode POSITION, POSITION_VELOCITY
JOINT_TRN = 0                               # MuJoCo actuator transmission: a joint
FIXED_GAIN, AFFINE_BIAS = 0, 1
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)


@dataclass(frozen=True)
class Dof:
    """One hinge or slide dof of a robot, in the world frame at the template pose."""

    index: int              # template dof id
    name: str               # "<robot>/<joint>"
    robot: str
    hinge: bool
    axis: np.ndarray        # (3,) unit
    anchor: np.ndarray      # (3,) a point on the axis
    bodies: tuple[int, ...]     # every body below the joint


def collect(b: newton.ModelBuilder, objs: Sequence[Any], scene: Any, fixed: float) -> RR.RobotFacts:
    """``RobotFacts`` of every robot template in ``objs`` (module docstring)."""
    robots = [t for t in objs if t.obj is None]
    dofs = [d for t in robots for d in robot_dofs(b, t)]
    inertia = {d.index: joint_inertia(b, d) for d in dofs}
    gaps = {r.name: RIGID_GAP_M if r.contact_gap is None else float(r.contact_gap)
            for r in getattr(scene, "robots", None) or ()}
    points = {body: body_points(b, body) for body in {x for d in dofs for x in d.bodies}}
    links = [k for d in dofs for k in [link_of(b, d, points, gaps.get(d.robot, RIGID_GAP_M))] if k is not None]
    contacts = [row for t in robots for row in shape_rows(b, t)]
    rows = [*(r for r in contacts if isinstance(r, RR.Row)), *limit_rows(b, dofs), *equality_rows(b)]
    loose = tuple(r for r in contacts if isinstance(r, RR.Loose))
    drives = servos(b, dofs, inertia)
    return RR.RobotFacts(fixed, tuple(drives), tuple(rows), tuple(links), loose,
                         tuple(damped(b, dofs, inertia, drives, {k.name for k in links})))


# --- builder access ---------------------------------------------------------------------------------------------

def attribute(b: newton.ModelBuilder, key: str, index: int, default: Any = None) -> Any:
    """Custom attribute ``key`` at ``index``: its value, the attribute's default, or ``default`` without one."""
    attr = b.custom_attributes.get(key)
    if attr is None:
        return default
    values = attr.values
    if isinstance(values, dict):
        return values.get(index, attr.default)
    return values[index] if values is not None and index < len(values) else attr.default


def _count(b: newton.ModelBuilder, key: str) -> int:
    attr = b.custom_attributes.get(key)
    return len(attr.values or ()) if attr is not None else 0


def _leaf(label: Any) -> str:
    return str(label).rsplit("/", 1)[-1]


def _pose(xform: Any) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix) of a transform (x, y, z, qx, qy, qz, qw)."""
    v = np.asarray([float(x) for x in xform], np.float64)
    return v[:3], quat_matrix(v[3:7] / np.linalg.norm(v[3:7]))


# --- joints -----------------------------------------------------------------------------------------------------

def subtree(b: newton.ModelBuilder, root: int) -> tuple[int, ...]:
    """``root`` and every body below it in the joint tree."""
    children: dict[int, list[int]] = {}
    for p, c in zip(b.joint_parent, b.joint_child, strict=True):
        children.setdefault(int(p), []).append(int(c))
    out, stack = [], [root]
    while stack:
        body = stack.pop()
        out.append(body)
        stack.extend(children.get(body, ()))
    return tuple(out)


def robot_dofs(b: newton.ModelBuilder, t: Any) -> Iterator[Dof]:
    """The hinge and slide dofs of robot ``t`` (free and ball joints have no servo axis here)."""
    for j in t.ranges.joint:
        if b.joint_type[j] in (JT.FREE, JT.BALL, JT.FIXED, JT.DISTANCE):
            continue
        child = int(b.joint_child[j])
        p, R = _pose(b.body_q[child])
        xp, xR = _pose(b.joint_X_c[j])
        linear, angular = b.joint_dof_dim[j]
        bodies = subtree(b, child)
        for k in range(linear + angular):
            d = int(b.joint_qd_start[j]) + k
            axis = R @ xR @ np.asarray([float(x) for x in b.joint_axis[d]], np.float64)
            norm = np.linalg.norm(axis)
            if norm <= 0.0:
                continue
            name = f"{t.name}/{_leaf(b.joint_label[j])}"
            yield Dof(d, name, t.name, k >= linear, axis / norm, p + R @ xp, bodies)


def joint_inertia(b: newton.ModelBuilder, d: Dof) -> float:
    """The dof's own joint-space inertia [kg m2, or kg for a slide] at the template pose, armature included."""
    total = float(b.joint_armature[d.index])
    for body in d.bodies:
        mass = float(b.body_mass[body])
        if not d.hinge:
            total += mass
            continue
        p, R = _pose(b.body_q[body])
        com = p + R @ np.asarray([float(x) for x in b.body_com[body]], np.float64)
        about_com = R @ np.asarray(b.body_inertia[body], np.float64).reshape(3, 3) @ R.T
        arm = np.cross(d.axis, com - d.anchor)
        total += float(d.axis @ about_com @ d.axis) + mass * float(arm @ arm)
    return total


# --- servos -----------------------------------------------------------------------------------------------------

def _target_gains(b: newton.ModelBuilder) -> dict[int, tuple[float, float]]:
    """{dof: (kp, kv)} of the dofs Newton drives as position targets."""
    return {d: (float(ke), float(b.joint_target_kd[d])) for d, ke in enumerate(b.joint_target_ke)
            if float(ke) > 0.0 and int(b.joint_target_mode[d]) in POSITION_MODES}


def _actuator_gains(b: newton.ModelBuilder) -> dict[int, tuple[float, float]]:
    """{dof: (kp, kv) gear^2} of MuJoCo's affine position servos on joints (the stiffest per dof); a negative kv
    stands for a damping ratio the file asked for (``_kv``)."""
    out: dict[int, tuple[float, float]] = {}
    for a in range(_count(b, "mujoco:actuator_gainprm")):
        kp = float(attribute(b, "mujoco:actuator_gainprm", a)[0])
        bias = float(attribute(b, "mujoco:actuator_biasprm", a)[1])
        servo = (int(attribute(b, "mujoco:actuator_gaintype", a, FIXED_GAIN)) == FIXED_GAIN
                 and int(attribute(b, "mujoco:actuator_biastype", a, 0)) == AFFINE_BIAS
                 and kp > 0.0 and abs(bias + kp) <= 1e-6 * kp)
        if not servo or int(attribute(b, "mujoco:actuator_trntype", a, JOINT_TRN)) != JOINT_TRN:
            continue
        dof = int(attribute(b, "mujoco:actuator_trnid", a)[0])
        gear = float(attribute(b, "mujoco:actuator_gear", a, (1.0,))[0])
        kv = -float(attribute(b, "mujoco:actuator_biasprm", a)[2])
        out[dof] = max(out.get(dof, (0.0, 0.0)), (kp * gear * gear, kv * gear * gear if kv > 0.0 else kv))
    return out


def _kv(kp: float, kv: float, inertia: float) -> float:
    """kv [N m s/rad]; a negative ``kv`` is minus a damping ratio: kv = 2 ratio sqrt(kp I)."""
    return kv if kv >= 0.0 else -2.0 * kv * math.sqrt(kp * inertia)


def servos(b: newton.ModelBuilder, dofs: Sequence[Dof], inertia: dict[int, float]) -> list[RR.Servo]:
    """One ``Servo`` per robot dof under position control: the stiffer of Newton's target gain and MuJoCo's row,
    with its kv (the larger of the two when both carry the same kp) plus the joint's damping."""
    target, actuator = _target_gains(b), _actuator_gains(b)
    out = []
    for d in dofs:
        rows = [g for g in (target.get(d.index), actuator.get(d.index)) if g is not None]
        if not rows:
            continue
        kp = max(g[0] for g in rows)
        kv = max(_kv(g[0], g[1], inertia[d.index]) for g in rows if g[0] == kp)
        out.append(RR.Servo(d.name, kp, inertia[d.index], kv + float(b.joint_damping[d.index])))
    return out


def damped(b: newton.ModelBuilder, dofs: Sequence[Dof], inertia: dict[int, float], drives: Sequence[RR.Servo],
           touching: set[str]) -> list[RR.Damped]:
    """One ``Damped`` per robot dof with damping (its servo's, else the joint's own) whose links collide
    (``touching``: the joints with a ``Link``)."""
    servo = {s.name: s.damping for s in drives}
    out = []
    for d in dofs:
        damping = servo.get(d.name, float(b.joint_damping[d.index]))
        if damping > 0.0 and d.name in touching:
            out.append(RR.Damped(d.name, damping, inertia[d.index]))
    return out


# --- time constants ---------------------------------------------------------------------------------------------

def shape_solref(b: newton.ModelBuilder, s: int) -> tuple[float, float] | None:
    """The contact (time constant [s], damping ratio) of shape ``s``, or None when the file wrote direct stiffness
    (no clamp)."""
    if int(attribute(b, "mujoco:solref_mode", s, SOLREF_FILE_DEFAULT)) == SOLREF_RAW:
        tau, zeta = (float(x) for x in attribute(b, "mujoco:solref", s, (DEFAULT_TAU, 1.0)))
        return (tau, zeta) if tau > 0.0 else None
    ke, kd = float(b.shape_material_ke[s]), float(b.shape_material_kd[s])
    return (2.0 / kd, 0.5 * kd / math.sqrt(ke)) if ke > 0.0 and kd > 0.0 else (DEFAULT_TAU, 1.0)


def loose_masses(b: newton.ModelBuilder, t: Any) -> dict[int, float]:
    """{body: its weld group's mass [kg]} of the loose bodies of MJCF robot ``t``: on a free joint, with no jointed
    body below (the light-body rule's scope: a box beside an arm, not a floating base)."""
    out: dict[int, float] = {}
    for j in t.ranges.joint:
        root = int(b.joint_child[j])
        if b.joint_type[j] != JT.FREE or not GEOM_FILE.match(str(b.body_label[root])):
            continue
        group = subtree(b, root)
        jointed = any(int(p) in group and b.joint_type[k] != JT.FIXED for k, p in enumerate(b.joint_parent))
        mass = sum(float(b.body_mass[body]) for body in group)
        if not jointed and mass > 0.0:
            out.update(dict.fromkeys(group, mass))
    return out


def shape_rows(b: newton.ModelBuilder, t: Any) -> Iterator[RR.Row | RR.Loose]:
    """The contact row of every collision shape of robot ``t``: a ``Loose`` on a loose body, else a ``Row``."""
    loose = loose_masses(b, t)
    for s in t.ranges.shape:
        solref = shape_solref(b, s) if int(b.shape_flags[s]) & RB.COLLIDE else None
        if solref is None:
            continue
        name, mass = f"{t.name}/{_leaf(b.shape_label[s])}", loose.get(int(b.shape_body[s]))
        if mass is None:
            yield RR.Row(name, "contact", solref[0])
        else:
            dmax = float(attribute(b, "mujoco:geom_solimp", s, (0.0, DEFAULT_DMAX))[1])
            yield RR.Loose(name, solref[0], solref[1], dmax, mass)


def limit_rows(b: newton.ModelBuilder, dofs: Sequence[Dof]) -> Iterator[RR.Row]:
    for d in dofs:
        if float(b.joint_limit_lower[d.index]) <= -UNLIMITED and float(b.joint_limit_upper[d.index]) >= UNLIMITED:
            continue
        mode = int(attribute(b, "mujoco:solreflimit_mode", d.index, 0))
        if mode == SOLREF_FILE_DEFAULT:
            yield RR.Row(d.name, "joint limit", DEFAULT_TAU)
        elif mode == SOLREF_RAW:
            tau = float(attribute(b, "mujoco:solreflimit", d.index, (DEFAULT_TAU, 1.0))[0])
            if tau > 0.0:
                yield RR.Row(d.name, "joint limit", tau)


def equality_rows(b: newton.ModelBuilder) -> Iterator[RR.Row]:
    for e in range(_count(b, "mujoco:eq_solref")):
        tau = float(attribute(b, "mujoco:eq_solref", e)[0])
        if tau > 0.0:
            label = attribute(b, "mujoco:equality_constraint_label", e, "") or f"equality {e}"
            yield RR.Row(_leaf(label), "equality", tau)


# --- link speed -------------------------------------------------------------------------------------------------

def body_points(b: newton.ModelBuilder, body: int) -> np.ndarray:
    """World corners (k, 3) of the boxes of ``body``'s collision shapes at the template pose (0 rows without any)."""
    p, R = _pose(b.body_q[body])
    corners = [x.centre + CORNERS * x.half for x in RB.body_boxes(b, body)]
    return p + np.concatenate(corners) @ R.T if corners else np.zeros((0, 3))


def link_of(b: newton.ModelBuilder, d: Dof, points: dict[int, np.ndarray], gap: float) -> RR.Link | None:
    """The fastest collision point below dof ``d`` at its declared speed, or None when nothing below it collides."""
    pts = np.concatenate([points[body] for body in d.bodies])
    if not len(pts):
        return None
    limit = float(b.joint_velocity_limit[d.index])
    if not d.hinge:
        return RR.Link(d.name, min(RR.LINK_SPEED_M_S, limit), gap)
    reach = float(np.linalg.norm(np.cross(pts - d.anchor, d.axis), axis=1).max())
    return RR.Link(d.name, min(RR.LINK_SPEED_RAD_S, limit) * reach, gap, reach)


__all__ = ["Dof", "attribute", "body_points", "collect", "damped", "equality_rows", "joint_inertia", "limit_rows",
           "link_of", "loose_masses", "robot_dofs", "servos", "shape_rows", "shape_solref", "subtree"]
