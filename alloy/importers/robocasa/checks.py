# SPDX-License-Identifier: Apache-2.0
"""Physics checks for imported models, in MuJoCo (CPU, float64): a drop test for objects, a joint sweep for fixtures.

``drop_test`` drops the object 2 cm onto a plane, upright and on its side, and checks the run stays finite, nothing
flies off, penetration stays under 2 mm, and (upright) it comes to rest without sliding away. ``fixture_test`` holds
the base fixed and checks for deep self-penetration closed, fully open, with each door alone half and fully open,
and with each drawer pulled out behind open doors; then lets gravity act for a second from closed and checks the
joints stay finite and inside their limits. These are importer gates;
the engine's own check tool (``alloy.check``) runs on top once it lands.

Example::

    >>> from alloy.importers.robocasa.checks import drop_test
    >>> drop_test(Path("out/mjcf/glass_cup/robocasa_glass_cup_003/model.xml")).ok      # doctest: +SKIP
    True
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.importers.robocasa")

DT = 0.001
DROP_GAP = 0.02            # m, lowest collision point above the plane at the start
MAX_PENETRATION = 0.002    # m, at rest (end of the run)
MAX_PEAK_PENETRATION = 0.025   # m, during the impact (MuJoCo's default soft contact lets a 2 cm drop sink briefly)
MAX_RISE = 0.05            # m above the start height = flew off
REST_SPEED = 0.02          # m/s
REST_SPIN = 0.5            # rad/s
MAX_SLIDE = 0.2            # m horizontal travel when dropped upright


@dataclass
class CheckResult:
    """Outcome of one check: ``ok`` and the failures, plus the measured numbers (JSON-able)."""

    name: str
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    numbers: dict[str, float | str] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.failures.append(message)


def _mujoco():
    import mujoco  # noqa: PLC0415 - heavy import, only when a check runs

    return mujoco


def _scene_xml(model_path: Path, free: bool, quat: tuple[float, ...]) -> str:
    """The model's XML with a ground plane added, its root body freed (or fixed) and rotated by ``quat``."""
    root = ET.parse(model_path).getroot()
    comp = root.find("compiler")
    comp = comp if comp is not None else ET.SubElement(root, "compiler")
    comp.set("meshdir", str(model_path.parent))
    comp.set("texturedir", str(model_path.parent))
    ET.SubElement(root, "option", timestep=str(DT), integrator="implicitfast")
    world = root.find("worldbody")
    body = world.find("body")
    body.set("quat", " ".join(map(str, quat)))
    if free:
        body.insert(0, ET.Element("freejoint", name="drop_free"))
    if free:      # a fixed fixture is centred on its origin: a floor at z = 0 would cut through it
        world.insert(0, ET.Element("geom", name="ground", type="plane", size="2 2 0.1", friction="0.5 0.005 0.0001"))
    return ET.tostring(root, encoding="unicode")


def _collision_min_z(m, d) -> float:
    """Lowest world z over every colliding geom's mesh vertices (or AABB corners for primitives)."""
    mj = _mujoco()
    low = np.inf
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 or m.geom_bodyid[g] == 0:
            continue
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            low = min(low, float(world[:, 2].min()))
        else:
            low = min(low, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
    return low


def _load(model_path: Path, free: bool, quat=(1.0, 0.0, 0.0, 0.0)):
    mj = _mujoco()
    m = mj.MjModel.from_xml_string(_scene_xml(model_path, free, quat))
    return m, mj.MjData(m)


def _drop_once(model_path: Path, quat: tuple[float, ...], seconds: float, res: CheckResult, tag: str) -> None:
    mj = _mujoco()
    m, d = _load(model_path, True, quat)
    mj.mj_forward(m, d)
    d.qpos[2] += DROP_GAP - _collision_min_z(m, d)
    mj.mj_forward(m, d)
    start = d.qpos[:3].copy()
    top, deepest = start[2], 0.0
    for _ in range(int(seconds / DT)):
        mj.mj_step(m, d)
        top = max(top, d.qpos[2])
        if d.ncon:
            deepest = min(deepest, float(d.contact.dist[: d.ncon].min()))
    speed, spin = float(np.linalg.norm(d.qvel[:3])), float(np.linalg.norm(d.qvel[3:6]))
    slide = float(np.linalg.norm(d.qpos[:2] - start[:2]))
    rest = -min([0.0, *(float(d.contact[i].dist) for i in range(d.ncon))])
    res.numbers.update({f"{tag}_speed": speed, f"{tag}_spin": spin, f"{tag}_slide": slide, f"{tag}_penetration": rest,
                        f"{tag}_peak_penetration": float(-deepest), f"{tag}_rise": float(top - start[2])})
    if not np.isfinite(d.qpos).all():
        res.fail(f"{tag}: state went non-finite")
    if top - start[2] > MAX_RISE:
        res.fail(f"{tag}: rose {top - start[2]:.3f} m (flew off; check the hulls or the mass)")
    if rest > MAX_PENETRATION:
        res.fail(f"{tag}: resting penetration {rest * 1000:.1f} mm > {MAX_PENETRATION * 1000:.0f} mm")
    if -deepest > MAX_PEAK_PENETRATION:
        res.fail(f"{tag}: impact penetration {-deepest * 1000:.1f} mm > {MAX_PEAK_PENETRATION * 1000:.0f} mm "
                 "(hulls too thin for the contact?)")


def drop_test(model_path: Path, seconds: float = 2.0) -> CheckResult:
    """Drop the object upright and on its side (see module doc)."""
    res = CheckResult("drop")
    try:
        _drop_once(model_path, (1.0, 0.0, 0.0, 0.0), seconds, res, "upright")
        _drop_once(model_path, (0.7071068, 0.7071068, 0.0, 0.0), seconds, res, "side")
    except ValueError as exc:
        res.fail(f"MuJoCo cannot load {model_path}: {exc}")
        return res
    n = res.numbers
    if n.get("upright_speed", 0) > REST_SPEED or n.get("upright_spin", 0) > REST_SPIN:
        res.fail(f"upright: not at rest after {seconds:g} s (speed {n['upright_speed']:.3f} m/s, "
                 f"spin {n['upright_spin']:.2f} rad/s)")
    if n.get("upright_slide", 0) > MAX_SLIDE:
        res.fail(f"upright: slid {n['upright_slide']:.2f} m")
    return res


def _self_penetration(m, d) -> float:
    """Deepest contact between two bodies of the fixture (ground excluded), m (positive = overlap)."""
    deepest = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0:
            deepest = min(deepest, float(c.dist))
    return -deepest


def _home(m) -> np.ndarray:
    """Closed pose: the model's qpos0 clamped into every joint's range."""
    q = m.qpos0.copy()
    for j in range(m.njnt):
        if m.jnt_limited[j]:
            a = m.jnt_qposadr[j]
            q[a] = np.clip(q[a], *m.jnt_range[j])
    return q


def _open_value(m, j: int, home: np.ndarray) -> float:
    """The end of joint ``j``'s range farther from its closed value."""
    lo, hi = m.jnt_range[j]
    q = home[m.jnt_qposadr[j]]
    return float(hi if abs(hi - q) >= abs(q - lo) else lo)


def _poses(m) -> list[tuple[str, np.ndarray]]:
    """Closed; all open; each hinge alone half and fully open; each slide fully out with every hinge open."""
    home = _home(m)
    limited = [j for j in range(m.njnt) if m.jnt_limited[j]]
    hinges = [j for j in limited if m.jnt_type[j] == 3]   # mjJNT_HINGE
    opened = home.copy()
    for j in limited:
        opened[m.jnt_qposadr[j]] = _open_value(m, j, home)
    doors_open = home.copy()
    for j in hinges:
        doors_open[m.jnt_qposadr[j]] = opened[m.jnt_qposadr[j]]
    poses = [("closed", home), ("all open", opened)]
    for j in limited:
        a, name = m.jnt_qposadr[j], m.joint(j).name
        base = home if j in hinges else doors_open
        fracs = (0.5, 1.0) if j in hinges else (1.0,)
        for frac in fracs:
            q = base.copy()
            q[a] = home[a] + frac * (opened[a] - home[a])
            poses.append((f"{name} at {frac:g} open", q))
    return poses


def _sweep(m, d, res: CheckResult) -> float:
    """Worst body-body overlap over ``_poses``; records the pose that gave it."""
    mj = _mujoco()
    worst = 0.0
    for label, q in _poses(m):
        d.qpos[:] = q
        mj.mj_forward(m, d)
        depth = _self_penetration(m, d)
        if depth > worst:
            worst, res.numbers["worst_pose"] = depth, label
    return worst


def fixture_test(model_path: Path, max_overlap: float = 0.01) -> CheckResult:
    """Fixed base: self-penetration with each joint swept alone, then 1 s under gravity from closed (module doc)."""
    mj = _mujoco()
    res = CheckResult("fixture")
    try:
        m, d = _load(model_path, False)
    except ValueError as exc:
        res.fail(f"MuJoCo cannot load {model_path}: {exc}")
        return res
    res.numbers["joints"] = float(m.njnt)
    worst = res.numbers["overlap"] = _sweep(m, d, res)
    if worst > max_overlap:
        res.fail(f"pose '{res.numbers['worst_pose']}': parts overlap by {worst * 1000:.1f} mm")
    d.qpos[:], d.qvel[:] = _home(m), 0.0
    for _ in range(int(1.0 / DT)):
        mj.mj_step(m, d)
    if not np.isfinite(d.qpos).all():
        res.fail("joint state went non-finite under gravity")
        return res
    lim = m.jnt_limited.astype(bool)
    q = d.qpos[m.jnt_qposadr[lim]]
    over = np.maximum(m.jnt_range[lim, 0] - q, q - m.jnt_range[lim, 1])
    res.numbers["limit_violation"] = float(max(over.max(), 0.0)) if len(over) else 0.0
    if res.numbers["limit_violation"] > 0.05:
        name = m.joint(int(np.flatnonzero(lim)[int(np.argmax(over))])).name
        res.fail(f"joint {name!r} left its range by {res.numbers['limit_violation']:.3f} under gravity")
    return res


__all__ = ["CheckResult", "drop_test", "fixture_test"]
