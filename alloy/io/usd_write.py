# SPDX-License-Identifier: Apache-2.0
"""``IOModel`` to USD (``.usda`` text) with the UsdPhysics schema; needs no USD library to write.

Layout: ``/<scene>`` (default prim, Z up, metres, kilograms) holds a ``PhysicsScene``, a ``Materials`` scope with
``PhysicsMaterialAPI`` materials, one Xform per object or robot, rigid bodies (``PhysicsRigidBodyAPI`` +
``PhysicsMassAPI``) with collision prims under them, and joints. Joint limits and drives use UsdPhysics units
(degrees for angles). Values UsdPhysics has no field for use Newton's open ``newton:`` attributes (torsional and
rolling friction, contact stiffness, joint armature, friction and velocity limit), which Newton's USD importer reads.

Example::

    from alloy.io.usd_write import write_usd
    write_usd(io, "out/scene.usda")     # io: an IOModel (alloy.io.build.scene_model)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from alloy.io.from_model import clean_name
from alloy.io.geometry import compose, inverse, matrix_quat, tessellate
from alloy.io.groups import Group, split_groups
from alloy.io.model import IDENTITY, IOBody, IOJoint, IOMaterial, IOModel, IOShape, Pose
from alloy.io.usd_text import Prim, f3, num, quat_text

log = logging.getLogger("alloy.io")
DEG = 180.0 / math.pi


def _xform(prim: Prim, pose: Pose, scale=None) -> None:
    prim.attr("double3 xformOp:translate", f3(pose[:3]))
    prim.attr("quatd xformOp:orient", quat_text(pose[3:]))
    if scale is None:
        prim.attr("uniform token[] xformOpOrder", '["xformOp:translate", "xformOp:orient"]')
        return
    prim.attr("double3 xformOp:scale", f3(scale))
    prim.attr("uniform token[] xformOpOrder", '["xformOp:translate", "xformOp:orient", "xformOp:scale"]')


def material_prim(m: IOMaterial) -> Prim:
    """A ``Material`` prim with ``PhysicsMaterialAPI`` values (and Newton's lever and stiffness attributes)."""
    p = Prim("Material", clean_name(m.name, "material"), ["PhysicsMaterialAPI", "NewtonMaterialAPI"])
    p.attr("float physics:staticFriction", num(m.mu_static))
    p.attr("float physics:dynamicFriction", num(m.mu_dynamic))
    p.attr("float physics:restitution", num(m.restitution))
    if m.density > 0:
        p.attr("float physics:density", num(m.density))
    p.attr("float newton:torsionalFriction", num(m.torsional_lever))
    p.attr("float newton:rollingFriction", num(m.rolling_lever))
    if m.contact_stiffness is not None:
        p.attr("float newton:contactStiffness", num(m.contact_stiffness))
    return p


def _mesh_prim(p: Prim, shape: IOShape) -> None:
    mesh = tessellate(shape)
    p.attr("int[] faceVertexCounts", "[" + ", ".join(["3"] * len(mesh.faces)) + "]")
    p.attr("int[] faceVertexIndices", "[" + ", ".join(str(int(i)) for i in mesh.faces.reshape(-1)) + "]")
    p.attr("point3f[] points", "[" + ", ".join(f3(v) for v in mesh.vertices) + "]")
    p.attr("uniform token physics:approximation", '"convexHull"' if shape.convex else '"none"')


def shape_prim(shape: IOShape, mat_path: str | None, io: IOModel) -> Prim:
    """A collision prim for one shape (Cube/Sphere/Capsule/Cylinder/Cone/Plane natively, else Mesh)."""
    s, kind, schemas = shape.size, shape.kind, ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    native = {"box": "Cube", "sphere": "Sphere", "capsule": "Capsule", "cylinder": "Cylinder", "cone": "Cone",
              "plane": "Plane"}
    typ = native.get(kind, "Mesh")
    if typ == "Mesh":
        schemas.insert(1, "PhysicsMeshCollisionAPI")
        if kind != "mesh":
            io.note("usd", shape.name, f"USD physics has no {kind}: written as a convex mesh")
    p = Prim(typ, clean_name(shape.name, "shape"), schemas if shape.collide else ["MaterialBindingAPI"])
    _xform(p, shape.pose, s if typ == "Cube" else None)
    if typ == "Cube":
        p.attr("double size", "2")
    elif typ == "Sphere":
        p.attr("double radius", num(s[0]))
    elif typ in ("Capsule", "Cylinder", "Cone"):
        p.attr("double radius", num(s[0]))
        p.attr("double height", num(2 * s[1]))
        p.attr("uniform token axis", '"Z"')
    elif typ == "Plane":
        p.attr("double width", num(2 * s[0] if s[0] > 0 else 100))
        p.attr("double length", num(2 * s[1] if s[1] > 0 else 100))
        p.attr("uniform token axis", '"Z"')
    else:
        _mesh_prim(p, shape)
    if not shape.visible:
        p.attr("token purpose", '"guide"')
    if mat_path:
        p.attr("rel material:binding:physics", f"<{mat_path}>")
    return p


def _mass(p: Prim, body: IOBody) -> None:
    """Mass, COM and principal inertia (eigen-decomposed, as UsdPhysics wants)."""
    w, v = np.linalg.eigh(np.asarray(body.inertia, np.float64))
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1.0
    p.attr("float physics:mass", num(body.mass))
    p.attr("point3f physics:centerOfMass", f3(body.com))
    p.attr("float3 physics:diagonalInertia", f3(np.maximum(w, 0.0)))
    p.attr("quatf physics:principalAxes", quat_text(matrix_quat(v)))


def body_prim(body: IOBody, io: IOModel, mats: dict[str, str], *, in_world: bool = True) -> Prim:
    """A rigid body Xform with its collision shapes (pose is the world pose when ``in_world``)."""
    p = Prim("Xform", clean_name(body.name, "body"), ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    _xform(p, body.pose if in_world else IDENTITY)
    if body.motion == "kinematic":
        p.attr("bool physics:kinematicEnabled", "1")
    _mass(p, body)
    for s in body.shapes:
        p.children.append(shape_prim(s, mats.get(s.material), io))
    return p


def _axis_frames(j: IOJoint) -> tuple[Pose, Pose]:
    """Joint frames rotated so the joint axis is the frame's X (UsdPhysics axes are tokens X/Y/Z)."""
    a = np.asarray(j.axis, np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.dot(x, a))
    if c > 1.0 - 1e-12:
        r = np.array([0.0, 0.0, 0.0, 1.0])
    elif c < -1.0 + 1e-12:
        r = np.array([0.0, 0.0, 1.0, 0.0])
    else:
        ax = np.cross(x, a)
        ax /= np.linalg.norm(ax)
        h = 0.5 * math.acos(c)
        r = np.array([*(ax * math.sin(h)), math.cos(h)])
    rot = (0.0, 0.0, 0.0, *r)
    return compose(j.parent_pose, rot), compose(j.child_pose, rot)


def joint_prim(j: IOJoint, paths: dict[str, str], io: IOModel) -> Prim | None:
    """A UsdPhysics joint prim (revolute/prismatic/spherical/fixed; free joints are just unjointed bodies)."""
    typ = {"revolute": "PhysicsRevoluteJoint", "prismatic": "PhysicsPrismaticJoint", "ball": "PhysicsSphericalJoint",
           "fixed": "PhysicsFixedJoint"}.get(j.kind)
    if typ is None:
        return None
    angular = j.kind == "revolute"
    schemas = [f"PhysicsDriveAPI:{'angular' if angular else 'linear'}"] if j.stiffness or j.drive_damping else []
    p = Prim(typ, clean_name(j.name, "joint"), schemas)
    p0, p1 = _axis_frames(j) if j.kind in ("revolute", "prismatic") else (j.parent_pose, j.child_pose)
    if j.parent is not None:
        p.attr("rel physics:body0", f"<{paths[j.parent]}>")
    p.attr("rel physics:body1", f"<{paths[j.child]}>")
    for k, pose in (("0", p0), ("1", p1)):
        p.attr(f"point3f physics:localPos{k}", f3(pose[:3]))
        p.attr(f"quatf physics:localRot{k}", quat_text(pose[3:]))
    if j.kind not in ("revolute", "prismatic"):
        return p
    unit = DEG if angular else 1.0
    p.attr("uniform token physics:axis", '"X"')
    if j.lower is not None and j.upper is not None:
        p.attr("float physics:lowerLimit", num(j.lower * unit))
        p.attr("float physics:upperLimit", num(j.upper * unit))
    _joint_extras(p, j, unit, angular)
    return p


def _joint_extras(p: Prim, j: IOJoint, unit: float, angular: bool) -> None:
    kind = "angular" if angular else "linear"
    if j.stiffness or j.drive_damping:
        p.attr(f"float drive:{kind}:physics:stiffness", num(j.stiffness / unit))
        p.attr(f"float drive:{kind}:physics:damping", num(j.drive_damping / unit))
        p.attr(f"float drive:{kind}:physics:targetPosition", num(j.q0 * unit))
        if j.effort is not None:
            p.attr(f"float drive:{kind}:physics:maxForce", num(j.effort))
    if j.armature or j.friction or j.velocity is not None:
        p.schemas.append("NewtonJointAPI")
    if j.armature:
        p.attr("float newton:armature", num(j.armature))
    if j.friction:
        p.attr("float newton:friction", num(j.friction))
    if j.velocity is not None:
        p.attr("float newton:velocityLimit", num(j.velocity * unit))
    if j.q0:
        p.attr(f"float state:{kind}:physics:position", num(j.q0 * unit))


def group_prim(g: Group, io: IOModel, mats: dict[str, str], root: str) -> tuple[Prim, list[Prim]]:
    """One object/robot Xform with its bodies (world poses) and static shapes; returns it and its joints."""
    gp = Prim("Xform", clean_name(g.name, "object"), ["PhysicsArticulationRootAPI"] if g.articulated else [])
    base = f"{root}/{gp.name}"
    paths = {b.name: f"{base}/{clean_name(b.name, 'body')}" for b in g.bodies}
    for b in g.bodies:
        gp.children.append(body_prim(b, io, mats))
    for s in g.world_shapes:
        gp.children.append(shape_prim(s, mats.get(s.material), io))
    joints = []
    for j in ([g.root_joint] if g.root_joint is not None and g.root_joint.kind != "free" else []) + g.joints:
        jp = joint_prim(j, paths, io)
        if jp is not None:
            joints.append(jp)
    return gp, joints


def usd_text(io: IOModel) -> str:
    """The whole ``.usda`` document of ``io``."""
    name = clean_name(io.name, "scene")
    root = f"/{name}"
    top = Prim("Xform", name)
    g = np.asarray(io.gravity, np.float64)
    mag = float(np.linalg.norm(g))
    scene = Prim("PhysicsScene", "physicsScene")
    scene.attr("vector3f physics:gravityDirection", f3(g / mag if mag else (0.0, 0.0, -1.0)))
    scene.attr("float physics:gravityMagnitude", num(mag))
    if io.timestep:
        scene.schemas.append("NewtonSceneAPI")
        scene.attr("uniform int newton:timeStepsPerSecond", str(int(round(1.0 / io.timestep))))
    top.children.append(scene)
    looks = Prim("Scope", "Materials")
    mats = {}
    for m in io.materials.values():
        looks.children.append(material_prim(m))
        mats[m.name] = f"{root}/Materials/{looks.children[-1].name}"
    top.children.append(looks)
    joints = Prim("Scope", "Joints")
    for grp in split_groups(io):
        gp, js = group_prim(grp, io, mats, root)
        top.children.append(gp)
        joints.children.extend(js)
    _equality_joints(io, joints, root)
    top.children.append(joints)
    header = ('#usda 1.0\n(\n    defaultPrim = "%s"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n'
              '    upAxis = "Z"\n    doc = "Exported by Alloy (alloy.io)"\n)\n\n' % name)
    return header + top.text()


def _equality_joints(io: IOModel, joints: Prim, root: str) -> None:
    """Welds become fixed joints excluded from articulations (the body keeps its current offset)."""
    paths = {}
    for grp in split_groups(io):
        for b in grp.bodies:
            paths[b.name] = f"{root}/{clean_name(grp.name, 'object')}/{clean_name(b.name, 'body')}"
    for e in io.equalities:
        if e.kind != "weld":
            io.note("usd", e.name, "connect (ball) equality not written; add a PhysicsSphericalJoint")
            continue
        b1 = io.body(e.body1)
        rel = IDENTITY if e.body2 is None else compose(inverse(io.body(e.body2).pose), b1.pose)
        jp = Prim("PhysicsFixedJoint", clean_name(e.name, "weld"))
        if e.body2 is not None:
            jp.attr("rel physics:body0", f"<{paths[e.body2]}>")
        jp.attr("rel physics:body1", f"<{paths[e.body1]}>")
        pos0 = rel if e.body2 is not None else b1.pose
        jp.attr("point3f physics:localPos0", f3(pos0[:3]))
        jp.attr("quatf physics:localRot0", quat_text(pos0[3:]))
        jp.attr("point3f physics:localPos1", f3((0.0, 0.0, 0.0)))
        jp.attr("quatf physics:localRot1", quat_text((0.0, 0.0, 0.0, 1.0)))
        jp.attr("bool physics:excludeFromArticulation", "1")
        joints.children.append(jp)


def write_usd(io: IOModel, path: str | Path) -> Path:
    """Write ``io`` as a ``.usda`` file at ``path`` (parents created) and return the path."""
    path = Path(path)
    if path.suffix != ".usda":
        raise ValueError(f"{path}: alloy.io writes text USD; use a .usda file name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(usd_text(io))
    log.info("wrote USD %s", path)
    return path

