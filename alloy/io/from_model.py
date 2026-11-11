# SPDX-License-Identifier: Apache-2.0
"""Read a finalized one-world ``newton.Model`` into an ``IOModel``.

The model can come from Alloy's scene assembler (then ``names`` carries object names and card materials) or from
Newton's own MJCF/URDF/USD importers (then materials are made from the per-shape surface values).

Example::

    from alloy._vendor import newton
    from alloy.io.from_model import read_model
    b = newton.ModelBuilder()
    b.add_urdf("robot.urdf")
    io = read_model(b.finalize("cpu"), "robot")
    print([body.name for body in io.bodies])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from alloy._vendor import newton
from alloy._vendor.newton import GeoType, JointType

from alloy.io.model import IOBody, IOEquality, IOJoint, IOMaterial, IOMesh, IOModel, IOShape, Pose

log = logging.getLogger("alloy.io")
UNLIMITED = 1e5          # |limit| at or above this is "no limit" (Newton stores 1e10)
_KINDS = {GeoType.BOX: "box", GeoType.SPHERE: "sphere", GeoType.CAPSULE: "capsule", GeoType.CYLINDER: "cylinder",
          GeoType.CONE: "cone", GeoType.ELLIPSOID: "ellipsoid", GeoType.PLANE: "plane", GeoType.MESH: "mesh",
          GeoType.CONVEX_MESH: "mesh"}
_JOINTS = {JointType.FREE: "free", JointType.FIXED: "fixed", JointType.REVOLUTE: "revolute",
           JointType.PRISMATIC: "prismatic", JointType.BALL: "ball"}
_WELD, _CONNECT = 1, 0   # MuJoCo mjtEq values stored in mujoco:equality_constraint_type


@dataclass
class ModelNames:
    """Optional metadata from an assembled scene: group (object/robot) of each body and shape, and materials."""

    groups: dict[str, str] = field(default_factory=dict)          # group -> "object" | "robot"
    body_group: dict[int, str] = field(default_factory=dict)
    shape_group: dict[int, str] = field(default_factory=dict)
    shape_material: dict[int, str] = field(default_factory=dict)
    materials: dict[str, IOMaterial] = field(default_factory=dict)


@dataclass
class _Arrays:
    """Host copies of the Model arrays the reader needs."""

    m: newton.Model
    a: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.a:
            self.a[key] = getattr(self.m, key).numpy()
        return self.a[key]


def clean_name(text: str, fallback: str) -> str:
    """A name safe in XML and USD paths: the last path element of ``text``, non-word characters as ``_``.

    >>> clean_name("robot/link 1", "x")
    'link_1'
    """
    tail = str(text or "").replace("\\", "/").rstrip("/").split("/")[-1]
    name = re.sub(r"\W", "_", tail)
    if not name:
        return fallback
    return name if not name[0].isdigit() else f"_{name}"


def unique_names(raw: list[str], prefix: str) -> list[str]:
    """Cleaned, de-duplicated names (``a``, ``a_1``, ...); empty labels become ``<prefix><index>``."""
    out, seen = [], set()
    for i, text in enumerate(raw):
        base = clean_name(text, f"{prefix}{i}")
        name, k = base, 1
        while name in seen:
            name, k = f"{base}_{k}", k + 1
        seen.add(name)
        out.append(name)
    return out


def _pose(v) -> Pose:
    return tuple(float(x) for x in np.asarray(v, np.float64).reshape(7))


def _material_from_shape(arr: _Arrays, s: int, density: float) -> IOMaterial:
    """A material made from one shape's surface values (files carry no static friction: μs = μd)."""
    mu = float(arr["shape_material_mu"][s])
    e = float(arr["shape_material_restitution"][s])
    tors, roll = float(arr["shape_material_mu_torsional"][s]), float(arr["shape_material_mu_rolling"][s])
    name = f"mat_mu{mu:.4g}_e{e:.3g}_t{tors:.3g}_r{roll:.3g}".replace(".", "p").replace("-", "m")
    return IOMaterial(name, density, mu, mu, e, tors, roll, origin="file")


def _shape(arr: _Arrays, s: int, name: str) -> IOShape | None:
    gt = GeoType(int(arr["shape_type"][s]))
    kind = _KINDS.get(gt)
    if kind is None:
        return None
    scale = np.asarray(arr["shape_scale"][s], np.float64)
    flags = int(arr["shape_flags"][s])
    size: tuple[float, ...] = {"box": tuple(scale), "sphere": (scale[0],), "ellipsoid": tuple(scale),
                               "plane": (scale[0], scale[1])}.get(kind, (scale[0], scale[1]))
    mesh = None
    if kind == "mesh":
        src = arr.m.shape_source[s]
        mesh = IOMesh(np.asarray(src.vertices, np.float64) * scale, np.asarray(src.indices, np.int32))
        size = ()
    return IOShape(name=name, kind=kind, size=tuple(float(x) for x in size), pose=_pose(arr["shape_transform"][s]),
                   mesh=mesh, convex=gt != GeoType.MESH, collide=bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)),
                   visible=bool(flags & int(newton.ShapeFlags.VISIBLE)))


def _limit(x: float) -> float | None:
    return None if abs(x) >= UNLIMITED else float(x)


def _joint(arr: _Arrays, j: int, body_names: list[str], name: str, io: IOModel) -> IOJoint | None:
    jt = JointType(int(arr["joint_type"][j]))
    kind = _JOINTS.get(jt)
    child = body_names[int(arr["joint_child"][j])]
    if kind is None:
        io.note("model", f"joint {name}", f"{jt.name} joints have no writer mapping; {child} is written as free")
        return None
    p = int(arr["joint_parent"][j])
    common = {"name": name, "kind": kind, "parent": body_names[p] if p >= 0 else None, "child": child,
              "parent_pose": _pose(arr["joint_X_p"][j]), "child_pose": _pose(arr["joint_X_c"][j])}
    if kind not in ("revolute", "prismatic"):
        return IOJoint(**common)
    d, q = int(arr["joint_qd_start"][j]), int(arr["joint_q_start"][j])
    eff, vel = float(arr["joint_effort_limit"][d]), float(arr["joint_velocity_limit"][d])
    return IOJoint(**common, axis=tuple(float(x) for x in arr["joint_axis"][d]),
                   lower=_limit(float(arr["joint_limit_lower"][d])), upper=_limit(float(arr["joint_limit_upper"][d])),
                   effort=_limit(eff), velocity=_limit(vel), friction=float(arr["joint_friction"][d]),
                   armature=float(arr["joint_armature"][d]), q0=float(arr["joint_q"][q]),
                   stiffness=float(arr["joint_target_ke"][d]), drive_damping=float(arr["joint_target_kd"][d]),
                   actuated=int(arr["joint_target_mode"][d]) != 0)


def _actuators(m: newton.Model, io: IOModel, joint_index: dict[int, int]) -> None:
    """Mark joints driven by MuJoCo actuators; position servos also give the drive gains and force limit."""
    ns = getattr(m, "mujoco", None)
    if ns is None or not hasattr(ns, "actuator_trnid") or not len(ns.actuator_trnid.numpy()):
        return
    trn, typ = ns.actuator_trnid.numpy(), ns.actuator_trntype.numpy()
    gain, bias = ns.actuator_gainprm.numpy(), ns.actuator_biasprm.numpy()
    frc, has_frc = ns.actuator_forcerange.numpy(), ns.actuator_has_forcerange.numpy()
    starts = m.joint_qd_start.numpy()
    dof_joint = {d: j for j in range(m.joint_count) for d in range(int(starts[j]), int(starts[j + 1]))}
    for a in range(len(trn)):
        k = joint_index.get(dof_joint.get(int(trn[a][0]), -1)) if int(typ[a]) == 0 else None
        if k is None:
            io.note("model", f"actuator {a}", "only joint actuators are read (tendon/site actuators dropped)")
            continue
        j = io.joints[k]
        servo = gain[a][0] > 0 and abs(bias[a][1] + gain[a][0]) < 1e-6 * gain[a][0]
        io.joints[k] = IOJoint(**{**j.__dict__, "actuated": True,
                                  "stiffness": float(gain[a][0]) if servo else j.stiffness,
                                  "drive_damping": float(-bias[a][2]) if servo else j.drive_damping,
                                  "effort": float(max(abs(frc[a][0]), abs(frc[a][1]))) if has_frc[a] else j.effort})


def _body_motion(arr: _Arrays, i: int) -> str:
    return "kinematic" if int(arr["body_flags"][i]) & int(newton.BodyFlags.KINEMATIC) else "dynamic"


def _equalities(m: newton.Model, body_names: list[str], io: IOModel) -> None:
    ns = getattr(m, "mujoco", None)
    if ns is None or not hasattr(ns, "equality_constraint_type"):
        return
    kinds = ns.equality_constraint_type.numpy()
    b1, b2 = ns.equality_constraint_body1.numpy(), ns.equality_constraint_body2.numpy()
    for k, (t, x, y) in enumerate(zip(kinds, b1, b2)):
        if int(t) not in (_WELD, _CONNECT) or (x < 0 and y < 0):
            io.note("model", f"equality {k}", "only body welds and connects are read (joint/tendon couplings dropped)")
            continue
        first, second = (int(y), int(x)) if x < 0 else (int(x), int(y))
        io.equalities.append(IOEquality(name=f"eq_{k}", kind="weld" if int(t) == _WELD else "connect",
                                        body1=body_names[first], body2=body_names[second] if second >= 0 else None))


def _density_by_body(arr: _Arrays, shapes_of: dict[int, list[int]]) -> dict[int, float]:
    """Uniform density that reproduces each body's mass from its collision shapes (0 if no volume)."""
    from alloy.io.geometry import shape_volume  # noqa: PLC0415 - avoids a cycle at import

    out = {}
    for b, shapes in shapes_of.items():
        read = [_shape(arr, s, "v") for s in shapes]
        vol = sum(shape_volume(x) for x in read if x is not None and x.collide)
        out[b] = float(arr["body_mass"][b]) / vol if vol > 0 else 0.0
    return out


def _add_shapes(arr: _Arrays, io: IOModel, names: ModelNames, shape_names: list[str]) -> dict[int, list[IOShape]]:
    body_of = arr["shape_body"]
    shapes_of: dict[int, list[int]] = {}
    for s in range(len(body_of)):
        shapes_of.setdefault(int(body_of[s]), []).append(s)
    density = _density_by_body(arr, {b: v for b, v in shapes_of.items() if b >= 0})
    out: dict[int, list[IOShape]] = {}
    for b, shapes in shapes_of.items():
        for s in shapes:
            shp = _shape(arr, s, shape_names[s])
            if shp is None:
                io.note("model", f"shape {shape_names[s]}", f"type {GeoType(int(arr['shape_type'][s])).name} skipped")
                continue
            mat = names.shape_material.get(s)
            if mat is None:
                made = _material_from_shape(arr, s, density.get(b, 0.0))
                io.materials.setdefault(made.name, made)
                mat = made.name
            out.setdefault(b, []).append(IOShape(**{**shp.__dict__, "material": mat}))
            if b < 0:
                io.shape_groups[shape_names[s]] = names.shape_group.get(s, shape_names[s])
    return out


def read_model(m: newton.Model, name: str, names: ModelNames | None = None) -> IOModel:
    """``IOModel`` of world 0 of ``m`` (build the model with one world). Particles, springs, triangles and
    tetrahedra are not read; each kind present is listed in ``IOModel.unmapped``."""
    names = names or ModelNames()
    arr = _Arrays(m)
    io = IOModel(name=name, materials=dict(names.materials), groups=dict(names.groups))
    g = np.asarray(m.gravity.numpy(), np.float64).reshape(-1, 3)[0] if hasattr(m, "gravity") else (0, 0, -9.81)
    io.gravity = tuple(float(x) for x in g)
    body_names = unique_names(list(m.body_label), "body")
    shape_names = unique_names(list(m.shape_label), "shape")
    by_body = _add_shapes(arr, io, names, shape_names)
    io.world_shapes = by_body.get(-1, [])
    for i, bn in enumerate(body_names):
        io.bodies.append(IOBody(name=bn, pose=_pose(arr["body_q"][i]), mass=float(arr["body_mass"][i]),
                                com=tuple(float(x) for x in arr["body_com"][i]),
                                inertia=tuple(tuple(float(x) for x in r) for r in arr["body_inertia"][i]),
                                shapes=tuple(by_body.get(i, [])), motion=_body_motion(arr, i),
                                group=names.body_group.get(i, "")))
    joint_names = unique_names(list(m.joint_label), "joint")
    joint_index = {}
    for j in range(m.joint_count):
        jt = _joint(arr, j, body_names, joint_names[j], io)
        if jt is not None:
            joint_index[j] = len(io.joints)
            io.joints.append(jt)
    _actuators(m, io, joint_index)
    _equalities(m, body_names, io)
    for kind, count in (("particles", m.particle_count), ("triangles", m.tri_count), ("tetrahedra", m.tet_count),
                        ("springs", getattr(m, "spring_count", 0))):
        if count:
            io.note("model", kind, f"{count} {kind} (deformables, liquids, grains) are not exported")
    log.info("read model %s: %d bodies, %d world shapes, %d joints", name, len(io.bodies), len(io.world_shapes),
             len(io.joints))
    return io
