# SPDX-License-Identifier: Apache-2.0
"""Write the kinematic tree of a compiled ``mujoco.MjModel`` as explicit MJCF elements.

A model loaded from ``.mjb`` has no spec, so ``mj_saveLastXML`` raises "No XML model loaded", and
``MjSpec.to_xml`` rounds numbers to 6 significant digits (timestep 0.0008333333333 -> 0.000833333; MuJoCo 3.16.0).
This writer emits every compiled value with ``repr`` precision (float64 round-trip) and no defaults or classes, so the
MJCF reads the same in MuJoCo and in Newton's importer. Mesh vertices are written in the compiled (recentred)
source frame (recentring undone) to binary ``.msh`` files. :mod:`alloy.tasks.mjb_verify` reloads the XML and checks it field by field.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from alloy.tasks.msh import MeshData, store_msh

log = logging.getLogger("alloy.tasks")

_OBJ = mujoco.mjtObj
_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def fmt(values: float | Iterable[float]) -> str:
    """Space-separated ``repr`` of float64 values (shortest exact round trip)."""
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    return " ".join(repr(float(v)) for v in arr)


def fmt_int(values: int | Iterable[int]) -> str:
    return " ".join(str(int(v)) for v in np.atleast_1d(np.asarray(values)))


def flag(value: bool) -> str:
    return "true" if value else "false"


def ranged(name: str, limited: bool, values: np.ndarray) -> dict[str, str]:
    """``{name: values}`` unless the range is unused (not limited and all zero): Newton's importer rejects
    ``range="0 0"`` on an unlimited joint (r1_t14 toy-car wheels) although MuJoCo ignores it."""
    return {name: fmt(values)} if limited or np.any(np.asarray(values) != 0.0) else {}


@dataclass(frozen=True)
class Names:
    """Unique names per object type; unnamed objects get ``<kind><index>``."""

    body: tuple[str, ...]
    joint: tuple[str, ...]
    geom: tuple[str, ...]
    site: tuple[str, ...]
    mesh: tuple[str, ...]
    actuator: tuple[str, ...]
    equality: tuple[str, ...]
    tendon: tuple[str, ...]


def _names(m: mujoco.MjModel, obj: mujoco.mjtObj, count: int, kind: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for i in range(count):
        name = mujoco.mj_id2name(m, obj, i) or f"{kind}{i}"
        if name in seen:
            name = f"{name}__{kind}{i}"
        seen.add(name)
        out.append(name)
    return tuple(out)


def model_names(m: mujoco.MjModel) -> Names:
    """Collect (and make unique) the names of every object the converter writes."""
    return Names(
        body=_names(m, _OBJ.mjOBJ_BODY, m.nbody, "body"),
        joint=_names(m, _OBJ.mjOBJ_JOINT, m.njnt, "joint"),
        geom=_names(m, _OBJ.mjOBJ_GEOM, m.ngeom, "geom"),
        site=_names(m, _OBJ.mjOBJ_SITE, m.nsite, "site"),
        mesh=_names(m, _OBJ.mjOBJ_MESH, m.nmesh, "mesh"),
        actuator=_names(m, _OBJ.mjOBJ_ACTUATOR, m.nu, "actuator"),
        equality=_names(m, _OBJ.mjOBJ_EQUALITY, m.neq, "equality"),
        tendon=_names(m, _OBJ.mjOBJ_TENDON, m.ntendon, "tendon"),
    )


def mesh_data(m: mujoco.MjModel, i: int) -> MeshData:
    """Compiled vertices and faces of mesh ``i`` (already scaled and recentred by the original compile)."""
    va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
    fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
    return MeshData(vertices=m.mesh_vert[va : va + vn].copy(), faces=m.mesh_face[fa : fa + fn].copy())


def _quat_mat(q: np.ndarray) -> np.ndarray:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(q, dtype=np.float64))
    return rot.reshape(3, 3)


def source_frame_mesh(m: mujoco.MjModel, i: int) -> MeshData:
    """Mesh ``i`` with the compiler's recentring undone: ``v_src = R(mesh_quat)·v + mesh_pos``.

    MuJoCo recentres every mesh on its inertial frame and folds the shift into the geom pose
    (``mjuu_frameaccum``). Writing the source-frame vertices and the matching geom pose (:func:`source_geom_pose`)
    lets the recompile see the data it saw first, which keeps qhull's choices: on r3_t11 it restores 3 of the 4
    hulls that lost one near-coplanar vertex when the recentred vertices were written."""
    md = mesh_data(m, i)
    verts = md.vertices.astype(np.float64) @ _quat_mat(m.mesh_quat[i]).T + m.mesh_pos[i]
    return MeshData(vertices=verts.astype(np.float32), faces=md.faces)


def source_geom_pose(m: mujoco.MjModel, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Spec-frame pose of mesh geom ``k``: inverse of ``pos = p + R(q)·mesh_pos``, ``quat = q ⊗ mesh_quat``."""
    mesh = int(m.geom_dataid[k])
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, m.mesh_quat[mesh])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, m.geom_quat[k], conj)
    pos = m.geom_pos[k] - _quat_mat(quat) @ m.mesh_pos[mesh]
    return pos, quat


def hull_vertex_count(m: mujoco.MjModel, i: int) -> int:
    """Vertex count of mesh ``i``'s compiled convex hull, or -1 if the model holds no hull for it."""
    adr = int(m.mesh_graphadr[i])
    return int(m.mesh_graph[adr]) if adr >= 0 else -1


def write_assets(root: ET.Element, m: mujoco.MjModel, names: Names, store: Path, rel: str) -> None:
    """Store each mesh in the content-addressed ``store`` and add its ``<mesh>`` asset as ``<rel>/<file>``.

    Meshes that have a hull in the recording get ``maxhullvert="-1"``: no limit, MuJoCo's default, which the recorder
    compiled with (hulls up to 1,423 vertices). Without the attribute Newton's importer applies its own default of 64
    (newton@ac7f7830:newton/_src/geometry/types.py:175, passed to MuJoCo at solver_mujoco.py:7469-7496), which cut 74
    of T12's hulls and caused BUL-0002: on the 64-vertex wrist-mount and finger-tip hulls MuJoCo 3.16's native-CCD
    multi-contact reports 41.6 mm for a 0.06 mm touch (docs/notes/wp13-bul0002.md). Newton reads -1 as no limit
    (geometry/utils.py remesh_convex_hull, sim/builder.py:9666). The recorded count itself is not used as the limit:
    a binding-equal limit still changes qhull's path (T14 replay floor 0.3 -> 1.3 µm)."""
    asset = ET.SubElement(root, "asset")
    for i in range(m.nmesh):
        fname = store_msh(store, names.mesh[i], source_frame_mesh(m, i))
        attrs = {"name": names.mesh[i], "file": f"{rel}/{fname}"}
        if hull_vertex_count(m, i) >= 4:
            attrs["maxhullvert"] = "-1"
        ET.SubElement(asset, "mesh", attrs)


def _body_elem(parent: ET.Element, m: mujoco.MjModel, i: int, name: str) -> ET.Element:
    attrs = {"name": name, "pos": fmt(m.body_pos[i]), "quat": fmt(m.body_quat[i])}
    if m.body_mocapid[i] >= 0:
        attrs["mocap"] = "true"
    if m.body_gravcomp[i] != 0.0:
        attrs["gravcomp"] = fmt(m.body_gravcomp[i])
    b = ET.SubElement(parent, "body", attrs)
    ET.SubElement(b, "inertial", pos=fmt(m.body_ipos[i]), quat=fmt(m.body_iquat[i]),
                  mass=fmt(m.body_mass[i]), diaginertia=fmt(m.body_inertia[i]))  # fmt: skip
    return b


def _poly(base: float, poly: np.ndarray) -> np.ndarray:
    """``[linear, poly...]`` when the polynomial terms are used, else the plain linear value."""
    return np.array([base, *poly]) if np.any(poly != 0.0) else np.array([base])


def _dof_attrs(m: mujoco.MjModel, k: int) -> dict[str, str]:
    dadr = int(m.jnt_dofadr[k])
    ndof = {_FREE: 6, _BALL: 3}.get(int(m.jnt_type[k]), 1)
    for field in ("dof_armature", "dof_damping", "dof_frictionloss", "dof_dampingpoly"):
        vals = getattr(m, field)[dadr : dadr + ndof]
        if not np.all(vals == vals[0]):
            log.warning("joint %d: %s differs across its %d dofs; writing the first", k, field, ndof)
    return {
        "armature": fmt(m.dof_armature[dadr]),
        "damping": fmt(_poly(m.dof_damping[dadr], m.dof_dampingpoly[dadr])),
        "frictionloss": fmt(m.dof_frictionloss[dadr]),
        "solreffriction": fmt(m.dof_solref[dadr]),
        "solimpfriction": fmt(m.dof_solimp[dadr]),
    }


def _joint_elem(b: ET.Element, m: mujoco.MjModel, k: int, name: str) -> None:
    jtype = int(m.jnt_type[k])
    attrs = {"name": name, "type": JOINT_TYPES[jtype]}
    if jtype != _FREE:
        attrs |= {"pos": fmt(m.jnt_pos[k]), "axis": fmt(m.jnt_axis[k]), "limited": flag(m.jnt_limited[k]),
                  "margin": fmt(m.jnt_margin[k]), "solreflimit": fmt(m.jnt_solref[k]),
                  "solimplimit": fmt(m.jnt_solimp[k])} | ranged("range", m.jnt_limited[k], m.jnt_range[k])  # fmt: skip
    if jtype not in (_FREE, _BALL):
        qadr = int(m.jnt_qposadr[k])
        attrs |= {"ref": fmt(m.qpos0[qadr]), "springref": fmt(m.qpos_spring[qadr])}
    attrs |= {
        "stiffness": fmt(_poly(m.jnt_stiffness[k], m.jnt_stiffnesspoly[k])),
        "actuatorfrclimited": flag(m.jnt_actfrclimited[k]),
        "actuatorgravcomp": flag(m.jnt_actgravcomp[k]),
        "group": fmt_int(m.jnt_group[k]),
    }
    attrs |= ranged("actuatorfrcrange", m.jnt_actfrclimited[k], m.jnt_actfrcrange[k])
    ET.SubElement(b, "joint", attrs | _dof_attrs(m, k))


def _geom_elem(b: ET.Element, m: mujoco.MjModel, k: int, names: Names) -> None:
    gtype = int(m.geom_type[k])
    attrs = {"name": names.geom[k], "type": GEOM_TYPES[gtype], "pos": fmt(m.geom_pos[k]), "quat": fmt(m.geom_quat[k])}
    if gtype == _MESH:
        pos, quat = source_geom_pose(m, k)
        attrs |= {"pos": fmt(pos), "quat": fmt(quat), "mesh": names.mesh[int(m.geom_dataid[k])]}
    else:
        attrs["size"] = fmt(m.geom_size[k])
    attrs |= {
        "contype": fmt_int(m.geom_contype[k]), "conaffinity": fmt_int(m.geom_conaffinity[k]),
        "condim": fmt_int(m.geom_condim[k]), "priority": fmt_int(m.geom_priority[k]),
        "friction": fmt(m.geom_friction[k]), "solmix": fmt(m.geom_solmix[k]), "solref": fmt(m.geom_solref[k]),
        "solimp": fmt(m.geom_solimp[k]), "margin": fmt(m.geom_margin[k]), "gap": fmt(m.geom_gap[k]),
        "group": fmt_int(m.geom_group[k]), "rgba": fmt(m.geom_rgba[k]),
    }  # fmt: skip
    if np.any(m.geom_fluid[k] != 0.0):
        log.warning("geom %s: fluid interaction parameters are not converted", names.geom[k])
    ET.SubElement(b, "geom", attrs)


def _site_elem(b: ET.Element, m: mujoco.MjModel, k: int, name: str) -> None:
    ET.SubElement(b, "site", name=name, type=GEOM_TYPES[int(m.site_type[k])], pos=fmt(m.site_pos[k]),
                  quat=fmt(m.site_quat[k]), size=fmt(m.site_size[k]), group=fmt_int(m.site_group[k]),
                  rgba=fmt(m.site_rgba[k]))  # fmt: skip


def _fill_body(b: ET.Element, m: mujoco.MjModel, i: int, names: Names) -> None:
    ja, jn = int(m.body_jntadr[i]), int(m.body_jntnum[i])
    for k in range(ja, ja + jn):
        _joint_elem(b, m, k, names.joint[k])
    ga, gn = int(m.body_geomadr[i]), int(m.body_geomnum[i])
    for k in range(ga, ga + gn):
        _geom_elem(b, m, k, names)
    for k in np.nonzero(m.site_bodyid == i)[0]:
        _site_elem(b, m, int(k), names.site[int(k)])


def write_tree(root: ET.Element, m: mujoco.MjModel, names: Names) -> None:
    """Add every body; compiled order is depth-first, so each parent element exists before its children."""
    world = ET.SubElement(root, "worldbody")
    elems: list[ET.Element] = [world]
    _fill_body(world, m, 0, names)
    for i in range(1, m.nbody):
        b = _body_elem(elems[int(m.body_parentid[i])], m, i, names.body[i])
        _fill_body(b, m, i, names)
        elems.append(b)
