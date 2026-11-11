# SPDX-License-Identifier: Apache-2.0
"""P3-2 scene port pack: the scene every engine loads for SB-7 / SB-11 / SB-5 / SB-6 (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/scene_pack.py --target task_t12 \
        --scene library/robots/task_t12/scene.xml --initial trace:bench/traces/t12 --out bench/xbench/ours/packs/t12
    PYTHONPATH=. python bench/xbench/ours/scene_pack.py --target <bully target> --scene <scene.xml> \
        --initial keyframe:init --out bench/xbench/ours/packs/<name>          # P3-2b embodiments (WP13)

Input: one MJCF scene file (robot, fixtures and objects in it; collision meshes convex, as MuJoCo collides them) and
the registered Alloy bully target (``alloy.bully.targets.get_target``) that runs that scene on the Alloy route: it
gives the action space (B0 legal box, rate limit, initial command), the robot bodies and the route's dt. For T12 the
scene is ``library/robots/task_t12/scene.xml`` (WP13's split hulls in it) and the target ``task_t12``. The pack holds:

- ``scene.xml``: that MJCF with every collision mesh as a convex-hull OBJ under ``meshes/`` (MuJoCo collides the
  convex hull of a mesh, so the hull is the collision geometry exactly), visual-only geoms (contype = conaffinity = 0)
  and their meshes dropped; every other element (bodies, inertials, joints, actuators, equality, pairs, excludes,
  option) unchanged.
- ``<name>.json`` (``packs/t12/t12.json``): the compiled facts a port must reproduce and its static check compares against (MuJoCo conventions:
  quaternions w x y z, body inertia principal + ``iquat``): bodies, joints, collision geoms, the contact pairs MuJoCo
  generates, folded per body pair, with their effective sliding friction, equality (finger coupling), actuators
  (affine PD: kp, kv), the initial state (a recorded trace's, an MJCF keyframe's, or qpos0), the bully action space,
  body roles, the S1 limits, Alloy's route timing and the target name (``alloy_target``) the scorers rebuild.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from alloy.tasks.msh import read_msh  # noqa: E402

log = logging.getLogger("xbench.ours")
PACKS = Path(__file__).resolve().parent / "packs"
_J = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_G = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def _visual_only(g: ET.Element) -> bool:
    return g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0"


def _write_obj(path: Path, verts: np.ndarray) -> int:
    """Convex hull of ``verts`` as an OBJ with outward triangles; returns the hull vertex count."""
    v = np.asarray(verts, dtype=np.float64)
    hull = ConvexHull(v)
    used = np.unique(hull.simplices)
    remap = {int(o): i for i, o in enumerate(used)}
    centre = v[used].mean(axis=0)
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v[used]]
    for tri in hull.simplices:
        a, b, c = (v[i] for i in tri)
        if np.dot(np.cross(b - a, c - a), a - centre) < 0.0:
            tri = tri[[0, 2, 1]]
        lines.append("f " + " ".join(str(remap[int(i)] + 1) for i in tri))
    path.write_text("\n".join(lines) + "\n")
    return int(used.size)


def _mesh_vertices(path: Path) -> np.ndarray:
    if path.suffix == ".msh":
        return read_msh(path).vertices
    import trimesh  # noqa: PLC0415 - obj / stl sources (menagerie)

    return np.asarray(trimesh.load(path, force="mesh", process=False).vertices)


def rewrite_scene(scene: Path, out: Path) -> dict[str, int]:
    """``scene.xml`` with collision meshes as hull OBJs; returns {mesh name: hull vertex count}."""
    tree = ET.parse(scene)
    root = tree.getroot()
    for body in root.iter("body"):
        for g in [g for g in body.findall("geom") if _visual_only(g)]:
            body.remove(g)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("type") == "mesh"}
    asset = root.find("asset")
    mesh_dir = (root.find("compiler").get("meshdir", "") if root.find("compiler") is not None else "")
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    hulls: dict[str, int] = {}
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name")
        if name not in used:
            asset.remove(mesh)
            continue
        verts = _mesh_vertices((scene.parent / mesh_dir / mesh.get("file")).resolve())
        scale = np.asarray([float(s) for s in mesh.get("scale", "1 1 1").split()])
        hulls[name] = _write_obj(out / "meshes" / f"{name}.obj", verts * scale)
        mesh.set("file", f"meshes/{name}.obj")
        mesh.attrib.pop("scale", None)
    if root.find("compiler") is not None:
        root.find("compiler").attrib.pop("meshdir", None)
    tree.write(out / "scene.xml")
    return hulls


def _name(m: mujoco.MjModel, kind: mujoco.mjtObj, i: int) -> str | None:
    return mujoco.mj_id2name(m, kind, i) if i >= 0 else None


def _bodies(m: mujoco.MjModel, d: mujoco.MjData) -> list[dict]:
    B = mujoco.mjtObj.mjOBJ_BODY
    return [{"name": _name(m, B, b), "parent": _name(m, B, int(m.body_parentid[b])), "mass": float(m.body_mass[b]),
             "ipos": m.body_ipos[b].tolist(), "iquat_wxyz": m.body_iquat[b].tolist(),
             "inertia_principal": m.body_inertia[b].tolist(), "pos": m.body_pos[b].tolist(),
             "quat_wxyz": m.body_quat[b].tolist(), "xpos0": d.xpos[b].tolist(), "xquat0_wxyz": d.xquat[b].tolist()}
            for b in range(1, m.nbody)]


def _joints(m: mujoco.MjModel) -> list[dict]:
    J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
    out = []
    for j in range(m.njnt):
        dof = int(m.jnt_dofadr[j])
        out.append({"name": _name(m, J, j), "type": _J[int(m.jnt_type[j])], "body": _name(m, B, int(m.jnt_bodyid[j])),
                    "qposadr": int(m.jnt_qposadr[j]), "dofadr": dof, "axis": m.jnt_axis[j].tolist(),
                    "pos": m.jnt_pos[j].tolist(), "limited": bool(m.jnt_limited[j]), "range": m.jnt_range[j].tolist(),
                    "armature": float(m.dof_armature[dof]), "damping": float(m.dof_damping[dof]),
                    "frictionloss": float(m.dof_frictionloss[dof]), "stiffness": float(m.jnt_stiffness[j])})
    return out


def _geoms(m: mujoco.MjModel, d: mujoco.MjData, hulls: dict[str, int]) -> list[dict]:
    G, B, M = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_MESH
    out = []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        mesh = _name(m, M, int(m.geom_dataid[g])) if int(m.geom_type[g]) == 7 else None
        out.append({"name": _name(m, G, g), "body": _name(m, B, int(m.geom_bodyid[g])), "type": _G[int(m.geom_type[g])],
                    "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(), "quat_wxyz": m.geom_quat[g].tolist(),
                    "mesh_obj": f"meshes/{mesh}.obj" if mesh else None, "hull_verts": hulls.get(mesh),
                    **(_obj_pose(m, g) if mesh else {}),
                    "friction": m.geom_friction[g].tolist(), "condim": int(m.geom_condim[g]),
                    "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
                    "margin": float(m.geom_margin[g]), "priority": int(m.geom_priority[g]),
                    "aabb_local": m.geom_aabb[g].tolist(), "rbound": float(m.geom_rbound[g]),
                    "xpos0": d.geom_xpos[g].tolist()})
    return out


def _obj_pose(m: mujoco.MjModel, g: int) -> dict:
    """Pose of the OBJ's own frame in the body frame: MuJoCo re-centres a mesh about its inertial frame (``mesh_pos``,
    ``mesh_quat``) and the compiled geom pose refers to that frame, while the OBJ keeps the source vertices, so
    body vertex = (geom pose * mesh pose^-1) * obj vertex. Checked against MuJoCo's compiled ``mesh_vert``."""
    k = int(m.geom_dataid[g])
    neg_p, neg_q = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(neg_p, neg_q, m.mesh_pos[k], m.mesh_quat[k])
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(pos, quat, m.geom_pos[g], m.geom_quat[g], neg_p, neg_q)
    return {"mesh_pos": m.mesh_pos[k].tolist(), "mesh_quat_wxyz": m.mesh_quat[k].tolist(), "obj_pos": pos.tolist(),
            "obj_quat_wxyz": quat.tolist(), "obj_pose_note": "place the OBJ at obj_pos / obj_quat_wxyz in the body "
            "frame (pos / quat_wxyz are MuJoCo's compiled pose of the re-centred mesh)"}


def check_obj_poses(m: mujoco.MjModel, out: Path) -> float:
    """Worst body-frame distance (m) between the written OBJ hull vertices placed with ``obj_pos`` / ``obj_quat_wxyz``
    and the nearest of MuJoCo's compiled vertices placed with the compiled geom pose (every OBJ vertex is a source
    vertex, so it must land on a compiled one)."""
    from scipy.spatial import cKDTree

    G, M = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_MESH
    worst = 0.0
    for g in np.flatnonzero(m.geom_type == 7):
        k = int(m.geom_dataid[g])
        name = _name(m, M, k)
        obj = out / "meshes" / f"{name}.obj"
        if not obj.exists():
            continue
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in obj.read_text().splitlines() if ln.startswith("v ")])
        comp = m.mesh_vert[int(m.mesh_vertadr[k]):int(m.mesh_vertadr[k]) + int(m.mesh_vertnum[k])].astype(np.float64)
        p = _obj_pose(m, int(g))
        a = np.array([_rot(np.asarray(p["obj_quat_wxyz"]), x) for x in v]) + p["obj_pos"]
        b = np.array([_rot(m.geom_quat[g], x) for x in comp]) + m.geom_pos[g]
        d, _ = cKDTree(b).query(a)
        worst = max(worst, float(d.max()))
        if d.max() > 1e-6:
            log.warning("geom %s: OBJ placement off by %.3g m", _name(m, G, int(g)), d.max())
    return worst


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = np.zeros(3)
    mujoco.mju_rotVecQuat(r, np.asarray(v, np.float64), np.asarray(q, np.float64))
    return r


def _filtered(m: mujoco.MjModel, g1: int, g2: int, excluded: set[tuple[int, int]]) -> bool:
    """MuJoCo's dynamic pair filter (engine_collision_driver.c ``filterBodyPair`` + contype/conaffinity + excludes):
    bodies welded together never collide (this covers static-static: weld id 0), nor a weld body and its parent's
    weld body (filterparent), so a jointless camera on link 6 does not touch link 5."""
    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
    if w1 == w2:
        return True
    p1, p2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
    filterparent = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    if filterparent and w1 != 0 and w2 != 0 and (w1 == p2 or w2 == p1):
        return True
    if not (m.geom_contype[g1] & m.geom_conaffinity[g2] or m.geom_contype[g2] & m.geom_conaffinity[g1]):
        return True
    return (min(b1, b2), max(b1, b2)) in excluded


def _contact_pairs(m: mujoco.MjModel) -> list[dict]:
    """Every geom pair MuJoCo generates contacts for and its sliding friction (explicit pair, else max rule)."""
    G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
    excluded = {(min(b1, b2), max(b1, b2)) for b1, b2 in
                ((int(s) >> 16, int(s) & 0xFFFF) for s in m.exclude_signature[: m.nexclude])}
    explicit = {(min(int(a), int(b)), max(int(a), int(b))): float(m.pair_friction[k][0])
                for k, (a, b) in enumerate(zip(m.pair_geom1, m.pair_geom2))}
    out = []
    for g1, g2 in itertools.combinations(range(m.ngeom), 2):
        key, src = (g1, g2), None
        if key in explicit:
            mu, src = explicit[key], "pair"
        elif not _filtered(m, g1, g2, excluded):
            p1, p2 = int(m.geom_priority[g1]), int(m.geom_priority[g2])
            f1, f2 = float(m.geom_friction[g1][0]), float(m.geom_friction[g2][0])
            mu, src = (f1 if p1 > p2 else f2 if p2 > p1 else max(f1, f2)), "dynamic"
        if src:
            out.append({"geom1": _name(m, G, g1), "geom2": _name(m, G, g2), "body1": _name(m, B, int(m.geom_bodyid[g1])),
                        "body2": _name(m, B, int(m.geom_bodyid[g2])), "mu": mu, "source": src})
    return out


def _body_pairs(pairs: list[dict]) -> list[dict]:
    """Geom pairs folded per body pair: effective sliding friction values and how many geom pairs carry each."""
    acc: dict[tuple[str, str], dict] = {}
    for c in pairs:
        key = tuple(sorted((c["body1"], c["body2"])))
        e = acc.setdefault(key, {"body1": key[0], "body2": key[1], "mu": {}, "geom_pairs": 0, "explicit": 0})
        e["mu"][str(c["mu"])] = e["mu"].get(str(c["mu"]), 0) + 1
        e["geom_pairs"] += 1
        e["explicit"] += c["source"] == "pair"
    return sorted(acc.values(), key=lambda e: (e["body1"], e["body2"]))


def force_clamp(m: mujoco.MjModel, a: int) -> list[float] | None:
    """The force actuator ``a`` can put on its joint, as MuJoCo clamps it: the actuator's ``forcerange`` (times gear)
    when ``forcelimited``, then the joint's ``actuatorfrcrange`` when ``actuatorfrclimited`` (a per-joint clamp on
    the summed actuator force; one actuator per joint in these packs). None: unclamped."""
    j = int(m.actuator_trnid[a][0])
    lo, hi = -np.inf, np.inf
    if m.actuator_forcelimited[a]:
        lo, hi = sorted(float(x) * float(m.actuator_gear[a][0]) for x in m.actuator_forcerange[a])
    if m.jnt_actfrclimited[j]:
        lo, hi = max(lo, float(m.jnt_actfrcrange[j][0])), min(hi, float(m.jnt_actfrcrange[j][1]))
    return None if np.isinf(lo) and np.isinf(hi) else [lo, hi]


def _actuators(m: mujoco.MjModel) -> list[dict]:
    A, J = mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT
    return [{"name": _name(m, A, a), "joint": _name(m, J, int(m.actuator_trnid[a][0])), "kp": float(m.actuator_gainprm[a][0]),
             "kv": float(-m.actuator_biasprm[a][2]), "bias_kp": float(-m.actuator_biasprm[a][1]),
             "gear": float(m.actuator_gear[a][0]), "forcelimited": bool(m.actuator_forcelimited[a]),
             "forcerange": m.actuator_forcerange[a].tolist(), "ctrllimited": bool(m.actuator_ctrllimited[a]),
             "ctrlrange": m.actuator_ctrlrange[a].tolist(),
             "joint_actfrclimited": bool(m.jnt_actfrclimited[int(m.actuator_trnid[a][0])]),
             "joint_actfrcrange": m.jnt_actfrcrange[int(m.actuator_trnid[a][0])].tolist(),
             "force_clamp": force_clamp(m, a),
             "law": "force = clip(gear * (kp * ctrl - bias_kp * q - kv * qdot), *force_clamp) (no clip when null)"}
            for a in range(m.nu)]


def _equality(m: mujoco.MjModel) -> list[dict]:
    """Equality constraints with their objects named by their own type (connect / weld: bodies or sites; joint:
    joints; tendon: tendons) and the data that type uses."""
    kinds = {0: "connect", 1: "weld", 2: "joint", 3: "tendon"}
    out = []
    for e in range(m.neq):
        kind, d = kinds.get(int(m.eq_type[e]), str(int(m.eq_type[e]))), m.eq_data[e]
        own = {"joint": mujoco.mjtObj.mjOBJ_JOINT, "tendon": mujoco.mjtObj.mjOBJ_TENDON}   # eq_objtype: connect/weld
        ot = int(own.get(kind, m.eq_objtype[e]))
        row = {"type": kind, "objtype": mujoco.mjtObj(ot).name.removeprefix("mjOBJ_").lower(),
               "obj1": _name(m, ot, int(m.eq_obj1id[e])), "obj2": _name(m, ot, int(m.eq_obj2id[e])),
               "solref": m.eq_solref[e].tolist(), "active": bool(m.eq_active0[e])}
        if kind in ("joint", "tendon"):
            row.update(polycoef=d[:5].tolist(), law="q1 = poly(q2), polycoef[0] + polycoef[1] * q2 + ...")
        elif kind == "connect":
            row.update(anchor1=d[0:3].tolist(), anchor2=d[3:6].tolist(),
                       law="anchor1 in obj1's frame coincides with anchor2 in obj2's frame (point, 3 rows)")
        elif kind == "weld":
            row.update(anchor=d[0:3].tolist(), relpos=d[3:6].tolist(), relquat_wxyz=d[6:10].tolist(),
                       torquescale=float(d[10]), law="obj2 held at relpos / relquat in obj1's frame (6 rows)")
        out.append(row)
    return out


def _alloy_target(name: str) -> dict:
    """Action space, robot bodies, limits and route timing of Alloy's bully target ``name``."""
    import dataclasses

    from alloy.bully.targets import get_target
    from alloy.bully.types import BullyLimits

    tg = get_target(name)
    a = tg.actions
    sim = tg.make_sim(1, "cpu")
    model = sim.model
    names = [str(lbl).rsplit("/", 1)[-1] for lbl in model.body_label]   # Newton body order, MJCF names
    robot = {names[b] for b in (tg.robot_bodies or ())}
    jt, child = model.joint_type.numpy(), model.joint_child.numpy()
    from alloy._vendor import newton  # noqa: PLC0415

    free = [names[c] for j, c in enumerate(child) if int(jt[j]) == int(newton.JointType.FREE) and names[c] not in robot]
    space = {"actuators": [str(x) for x in getattr(tg, "action_names", ())] or None, "kind": tg.action_kind,
             "index": list(tg.action_index), "low": a.low.tolist(), "high": a.high.tolist(),
             "rate_hz": float(a.rate_hz), "max_delta_per_tick": a.max_delta_per_tick.tolist(),
             "initial": None if a.initial is None else np.asarray(a.initial).tolist(),
             "hold": "zero-order hold: command = trace[world, tick] for the whole tick"}
    roles = {"robot": sorted(robot, key=names.index), "free": free,
             "static": [n for n in names if n not in robot and n not in free], "thin_parts": dict(tg.thin_parts or {}),
             "non_target": [names[b] for b in tg.non_target_bodies]}
    dt = float(getattr(sim, "dt", 0.0))
    return {"alloy_target": name, "action_space": space, "roles": roles, "limits": dataclasses.asdict(BullyLimits()),
            "alloy_route": {"dt": dt, "control_dt": 1.0 / float(a.rate_hz), "substeps": int(getattr(sim, "substeps", 0)),
                            "solver": type(getattr(sim, "solver", None)).__name__}}


def _initial(m: mujoco.MjModel, how: str) -> dict:
    """``trace:<dir>`` (a recorded trace's first state), ``keyframe:<name>`` or ``qpos0``."""
    kind, _, arg = how.partition(":")
    if kind == "trace":
        from alloy.tasks.trace import load_trace

        tt = load_trace(ROOT / arg)
        qpos, qvel = (np.asarray(tt.trace.initial[k], dtype=np.float64) for k in ("qpos", "qvel"))
    elif kind == "keyframe":
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, arg)
        if k < 0:
            raise ValueError(f"no keyframe {arg!r} in the scene")
        qpos, qvel = np.asarray(m.key_qpos[k], np.float64), np.asarray(m.key_qvel[k], np.float64)
    elif kind == "qpos0":
        qpos, qvel = np.asarray(m.qpos0, np.float64), np.zeros(m.nv)
    else:
        raise ValueError(f"--initial must be trace:<dir>, keyframe:<name> or qpos0, got {how!r}")
    return {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "source": how + " (MuJoCo order)"}


def build(scene: Path, target: str, initial: str, out: Path, licence: str = "") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    hulls = rewrite_scene(scene, out)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    init = _initial(m, initial)
    d.qpos[:], d.qvel[:] = init["qpos"], init["qvel"]
    mujoco.mj_forward(m, d)
    err = check_obj_poses(m, out)
    if err > 1e-6:
        raise ValueError(f"OBJ frame poses do not reproduce MuJoCo's compiled vertices ({err:.3g} m)")
    pack = {"schema": "alloy.xbench.ours.scene/1", "source": str(scene.resolve().relative_to(ROOT)),
            "licence": licence,
            "option": {"timestep": float(m.opt.timestep), "gravity": m.opt.gravity.tolist(),
                       "integrator": int(m.opt.integrator), "cone": int(m.opt.cone), "impratio": float(m.opt.impratio)},
            "bodies": _bodies(m, d), "joints": _joints(m), "geoms": _geoms(m, d, hulls),
            "contact_body_pairs": _body_pairs(_contact_pairs(m)), "equality": _equality(m), "actuators": _actuators(m),
            "initial": init, **_alloy_target(target),
            "conventions": {"quat": "w x y z in this file; trajectory npz use x y z w (Newton)",
                            "body_q": "body frame origin, world frame", "body_qd": "COM linear velocity, angular "
                            "velocity, both world frame (Newton State.body_qd)"}}
    if pack["action_space"]["actuators"] is None:
        pack["action_space"]["actuators"] = [a["name"] for a in pack["actuators"]]
    path = out / f"{out.name}.json"
    path.write_text(json.dumps(pack, indent=1))
    log.info("wrote %s (%d bodies, %d collision geoms, %d contact body pairs, %d hull meshes)", path,
             len(pack["bodies"]), len(pack["geoms"]), len(pack["contact_body_pairs"]), len(hulls))
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="task_t12", help="registered Alloy bully target that runs the scene")
    p.add_argument("--scene", type=Path, default=ROOT / "library/robots/task_t12/scene.xml")
    p.add_argument("--initial", default="trace:bench/traces/t12")
    p.add_argument("--out", type=Path, default=PACKS / "t12")
    p.add_argument("--licence", default="Apache-2.0 AND MIT")
    a = p.parse_args()
    build(a.scene, a.target, a.initial, a.out, a.licence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
