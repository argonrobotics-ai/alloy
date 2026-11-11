# SPDX-License-Identifier: Apache-2.0
"""Assemble the P3-2b embodiment scenes with ``mujoco.MjSpec`` (offline, CPU).

A scene is the robot from its pinned source file, unchanged except where noted, plus its world:
- **tabletop** (arms, bimanual, mobile): a 1.2 x 1.2 m table with its top at z = 0 and the robot base on it, a floor
  0.75 m below, and three objects on the table: a 50 mm box, a 60 x 100 mm cylinder and T14's mug (the recorded
  reference-task mug, already convex pieces);
- **mobile**: the same table and objects, top at 0.75 m, 0.7-1.9 m in front of the robot on the floor;
- **legged**: a floor at z = 0 and a 0.3 x 0.3 x 0.1 m box obstacle 1 m ahead, the robot standing;
- **hand**: a 40 mm cube resting in the upturned palm.

Robot loaders and their changes to the sources: :mod:`alloy.tasks.emb_robots`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from alloy.core.errors import TraceError
from alloy.tasks.emb_firm import FIRM_CONTACTS, FIRM_SOLMIX, FIRM_SOLREF, firm_contacts, firm_world
from alloy.tasks.emb_robots import ROBOTS, StartPose, strip_visuals
from alloy.tasks.emb_specs import CONTACT_SOLREF, Embodiment
from alloy.cards.library import default_root

log = logging.getLogger("alloy.tasks")

MUG_SCENE = default_root() / "robots" / "task_t14" / "scene.xml"
TABLE_HALF = (0.6, 0.6, 0.025)      # [m] 1.2 x 1.2 m top, 50 mm thick


@dataclass(frozen=True)
class TableLayout:
    centre_x: float                 # [m] table centre
    top_z: float                    # [m] table top
    floor_z: float
    objects_x: float                # [m] box, cylinder and mug at y = -0.2, 0, 0.2


ARM_TABLE = TableLayout(0.3, 0.0, -0.75, 0.5)        # robot base on the table top (x -0.3..0.9)
MOBILE_TABLE = TableLayout(1.3, 0.75, 0.0, 0.95)    # robot on the floor, table x 0.7..1.9 in front, top 0.75 m
CUBE_HALF = 0.02                    # [m] the hand scene's cube (40 mm, 0.05 kg)


def _static_box(spec: mujoco.MjSpec, name: str, half: tuple[float, ...], pos: tuple[float, ...]) -> None:
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half))


def _floor(spec: mujoco.MjSpec, z: float) -> None:
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05], pos=[0, 0, z])


def _free_body(spec: mujoco.MjSpec, name: str, pos: tuple[float, ...]) -> mujoco.MjsBody:
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_free")
    return body


def _mug(spec: mujoco.MjSpec, pos: tuple[float, ...]) -> None:
    """T14's mug body (convex pieces, explicit inertial; collision geoms only), attached at ``pos``."""
    src = mujoco.MjSpec.from_file(str(MUG_SCENE))
    for mesh in src.meshes:                          # absolute paths: the attached copy resolves them from here
        mesh.file = str((MUG_SCENE.parent / mesh.file).resolve())
    frame = spec.worldbody.add_frame(pos=list(pos))
    frame.attach_body(src.body("mug"), "", "")


def table_scene(spec: mujoco.MjSpec, t: TableLayout) -> None:
    _floor(spec, t.floor_z)
    _static_box(spec, "table", TABLE_HALF, (t.centre_x, 0.0, t.top_z - TABLE_HALF[2]))
    box = _free_body(spec, "box", (t.objects_x, -0.2, t.top_z + 0.025))
    box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025] * 3, mass=0.1)
    cyl = _free_body(spec, "cylinder", (t.objects_x, 0.0, t.top_z + 0.05))
    cyl.add_geom(name="cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.05, 0.0], mass=0.15)
    _mug(spec, (t.objects_x, 0.2, t.top_z))


def tabletop(spec: mujoco.MjSpec) -> None:
    table_scene(spec, ARM_TABLE)


def mobile(spec: mujoco.MjSpec) -> None:
    table_scene(spec, MOBILE_TABLE)


def legged(spec: mujoco.MjSpec) -> None:
    _floor(spec, 0.0)
    _static_box(spec, "obstacle", (0.15, 0.15, 0.05), (1.0, 0.0, 0.05))


def hand(spec: mujoco.MjSpec) -> None:
    """Floor at z = 0 and the cube above the palm centre; :func:`_rest_in_palm` lowers it onto the palm."""
    _floor(spec, 0.0)
    cube = _free_body(spec, "cube", (0.0, 0.0, 0.3))
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=0.05)


WORLDS = {"tabletop": tabletop, "legged": legged, "hand": hand, "mobile": mobile}
TABLES = {"tabletop": ARM_TABLE, "mobile": MOBILE_TABLE}


def _rest_on_table(m: mujoco.MjModel, spec: mujoco.MjSpec, body: str, top_z: float) -> None:
    """Lift or lower ``body`` so its lowest collision vertex is at ``top_z`` (the table top)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b = m.body(body).id
    low = np.inf
    for g in (g for g in range(m.ngeom) if m.geom_bodyid[g] == b):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        k = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        low = min(low, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
    if not np.isfinite(low):
        raise TraceError(f"body {body!r} has no collision mesh to rest on the table")
    spec.body(body).pos = list(np.asarray(spec.body(body).pos) - np.array([0.0, 0.0, low - top_z]))


def _robot_geoms(m: mujoco.MjModel, skip: int) -> list[int]:
    return [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
            and m.geom_bodyid[g] not in (0, skip)]  # fmt: skip


def _rest_in_palm(m: mujoco.MjModel, spec: mujoco.MjSpec, body: str, start: StartPose) -> None:
    """Centre ``body`` over the hand's collision geoms and lower it until it just touches them (exact
    ``mj_geomDistance``; each drop is the current distance, so it never sinks in)."""
    qpos, _ = init_key(m, start)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    b = m.body(body).id
    hand_geoms = _robot_geoms(m, b)
    centre = np.mean([d.geom_xpos[g] for g in hand_geoms], axis=0)
    top = max(d.geom_xpos[g][2] + m.geom_rbound[g] for g in hand_geoms)
    pos = np.array([centre[0], centre[1], top + 2 * CUBE_HALF])
    cube = int(m.body_geomadr[b])
    for _ in range(64):
        spec.body(body).pos = pos.tolist()
        m = spec.compile()
        d = mujoco.MjData(m)
        d.qpos[:] = init_key(m, start)[0]
        mujoco.mj_forward(m, d)
        gap = min(mujoco.mj_geomDistance(m, d, cube, g, 0.5, None) for g in hand_geoms)
        if gap < 1e-7:
            return
        pos[2] -= gap
    raise TraceError(f"{body!r} did not come to rest on the hand (gap {gap:.3g} m)")


OVERLAP_EXCLUDE_M = 5e-4           # robot self-pairs deeper than this at the start pose or qpos0 are excluded


def exclude_start_overlaps(m: mujoco.MjModel, spec: mujoco.MjSpec, start: StartPose) -> list[tuple[str, str]]:
    """Exclude robot body pairs that overlap by more than 0.5 mm at the start pose, or at qpos0, where ``alloy.build``
    starts a robot (it does not apply keyframe ``init``): the source does not mean them to collide.
    - H1-2's URDF: the wrist yaw link and the thumb proximal link (grandparent and grandchild once the fixed hand base
      is fused, so MuJoCo's parent filter misses them) overlap 3-4.6 mm at the start pose.
    - The Franka arms (Menagerie's convex hulls): link5 and the hand (Panda) or link7 (FR3) overlap 30 / 43 / 5.6 mm at
      qpos0 (Panda / FR3 Duo / FR3) and 15-33 mm at joint6's lower limit, inside the joint ranges (release red,
      2026-11-09: the shipped Panda and Duo scenes failed check() at the start)."""
    robot = set(_robot_tree(m))
    pairs = set()
    for qpos in (init_key(m, start)[0], m.qpos0):
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            b1, b2 = (int(m.geom_bodyid[g]) for g in c.geom)
            if c.dist < -OVERLAP_EXCLUDE_M and b1 in robot and b2 in robot:
                pairs.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
    for b1, b2 in sorted(pairs):
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
        log.info("excluded self-pair %s | %s (overlaps at the start pose or qpos0)", b1, b2)
    return sorted(pairs)


def _robot_tree(m: mujoco.MjModel) -> list[int]:
    """Bodies in the kinematic trees of the actuated joints."""
    roots = {int(m.body_rootid[m.jnt_bodyid[j]]) for j in m.actuator_trnid[:, 0]}
    return [b for b in range(1, m.nbody) if int(m.body_rootid[b]) in roots]


def init_key(m: mujoco.MjModel, start: StartPose) -> tuple[np.ndarray, np.ndarray]:
    """``init`` keyframe qpos and ctrl: the robot's start pose, objects at qpos0, commands from ``start``. Both are
    clipped into the joint ranges (and ctrl ranges): G1's "stand" puts one joint 3 mrad past its limit."""
    qpos = m.qpos0.copy()
    for name, value in start.qpos.items():
        j = m.joint(name).id
        lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf)
        qpos[m.jnt_qposadr[j]] = float(np.clip(value, lo, hi))
    ctrl = np.array([start.ctrl[m.actuator(a).name] for a in range(m.nu)], dtype=np.float64)
    joints = m.actuator_trnid[:, 0]
    lo = np.where(m.jnt_limited[joints], m.jnt_range[joints, 0], -np.inf)
    hi = np.where(m.jnt_limited[joints], m.jnt_range[joints, 1], np.inf)
    lo = np.where(m.actuator_ctrllimited, np.maximum(lo, m.actuator_ctrlrange[:, 0]), lo)
    hi = np.where(m.actuator_ctrllimited, np.minimum(hi, m.actuator_ctrlrange[:, 1]), hi)
    return qpos, np.clip(ctrl, lo, hi)


def _lift_to_touch(m: mujoco.MjModel, spec: mujoco.MjSpec, start: StartPose) -> bool:
    """Raise the free-floating robot so its deepest floor contact at the start pose just touches (dist 0); returns
    whether it moved. A source keyframe may stand the feet a little into the floor (G1 "stand": 1.9 mm)."""
    qpos, _ = init_key(m, start)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    floor = m.geom("floor").id
    dist = [c.dist for c in d.contact[: d.ncon] if floor in (int(c.geom[0]), int(c.geom[1]))]
    if not dist or min(dist) >= 0.0:
        return False
    free = next(j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE)
    body = next(b for b in spec.bodies if free.name in {j.name for j in b.joints})
    body.pos = list(np.asarray(body.pos) + np.array([0.0, 0.0, -min(dist)]))
    log.info("lifted %s by %.3f mm to touch the floor", body.name, -1e3 * min(dist))
    return True


def stiffen_contacts(spec: mujoco.MjSpec, solref: tuple[float, float]) -> None:
    """A contact variant: every geom's solref = ``solref``; a geom whose solimp is softer than MuJoCo's default
    (dmin < 0.9: Go2's feet, 0.015 1 0.022) gets the default solimp. Friction, margins and every stiffer solimp
    (T14's mug, the 2F-85 pads: 0.95 0.99 0.001) stay as shipped. The ``_route`` variant needs the solimp floor too:
    with Go2's shipped feet at solref 1.25 ms the feet still rest 6.9 mm deep (64 CPU episodes: p50 6.87, max 9.54 mm;
    0.93 mm max with the floor), so SB-7 would measure the source's soft feet, not Alloy."""
    from alloy.tasks.emb_specs import DEFAULT_SOLIMP

    for g in spec.geoms:
        g.solref = list(solref)
        if g.solimp[0] < DEFAULT_SOLIMP[0]:
            g.solimp = list(DEFAULT_SOLIMP)


def set_options(spec: mujoco.MjSpec) -> None:
    """The sparse Jacobian (module doc); every other option as the source robot ships it."""
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE


def build_spec(e: Embodiment) -> mujoco.MjSpec:
    """The scene spec with its ``init`` keyframe (compiles; nothing written)."""
    if e.builder not in ROBOTS:
        raise TraceError(f"no builder for embodiment {e.key!r}")
    spec, start = ROBOTS[e.builder](e)
    if e.key in FIRM_CONTACTS:
        log.info("%s: %d robot geoms firm (solref %s, solmix %g)", e.name, firm_contacts(spec), FIRM_SOLREF,
                 FIRM_SOLMIX)  # fmt: skip
    WORLDS[e.world](spec)
    if e.key in FIRM_CONTACTS:
        log.info("%s: %d world geoms firm (solref %s)", e.name, firm_world(spec), FIRM_SOLREF)
    strip_visuals(spec)
    set_options(spec)
    if e.contact:
        stiffen_contacts(spec, CONTACT_SOLREF[e.contact])
    spec.modelname = e.name
    m = spec.compile()
    if e.world in TABLES:
        _rest_on_table(m, spec, "mug", TABLES[e.world].top_z)
        m = spec.compile()
    if e.world in ("legged", "mobile") and _lift_to_touch(m, spec, start):
        m = spec.compile()
    if e.world == "hand":
        _rest_in_palm(m, spec, "cube", start)
        m = spec.compile()
    if exclude_start_overlaps(m, spec, start):
        m = spec.compile()
    qpos, ctrl = init_key(m, start)
    spec.add_key(name="init", qpos=qpos.tolist(), qvel=[0.0] * m.nv, ctrl=ctrl.tolist())
    log.info("%s: %d bodies, %d dof, %d actuators, %d geoms, %d meshes", e.name, m.nbody, m.nv, m.nu, m.ngeom,
             m.nmesh)  # fmt: skip
    return spec
