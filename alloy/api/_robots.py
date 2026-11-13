# SPDX-License-Identifier: Apache-2.0
"""Robots in the Scene API: per-world base positions and the contact gap of a robot's shapes.

A robot whose file gives its root body a free joint (MJCF ``<freejoint/>``: a carton with flaps, a mobile base, a
loose gripper) can start somewhere else in every world (``add_robot(pos=(worlds, 3))``, ``scene.randomize(robot, "x",
lo, hi)``) and be moved while running (``sim.set_pose(robot, pos=...)``). These write the root free joint and run
forward kinematics over the robot's joints, so every link follows its base. A robot welded to the world has one base
pose for all worlds: move the other objects per world instead.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from alloy.api.errors import BuildError, SceneError, SimulationError

log = logging.getLogger("alloy.api")
AXES = {"x": 0, "y": 1, "z": 2, "pose.0": 0, "pose.1": 1, "pose.2": 2}
FREE_JOINT = 4      # newton.JointType.FREE
_WELDED_FIX = ("give the robot's root body a free joint (MJCF <freejoint/>) to place it per world, or move the other "
               "objects per world (add_object / add_box pos=(worlds, 3), scene.randomize)")


def contact_gap(value: Any) -> float | None:
    """A robot's ``contact_gap`` (m) checked, or None for the scene default (5 mm). Example: ``contact_gap(0.002)``."""
    if value is None:
        return None
    gap = float(value)
    if not math.isfinite(gap) or gap < 0.0:
        raise SceneError(f"contact_gap must be finite and >= 0 m, got {value!r}",
                         "pass contact_gap=0.005 (the default), or 0.1 for Newton's importer default",
                         field="contact_gap")
    return gap


def source(name: str, path: Any, mjcf: str | None) -> tuple[str, str | None, str | None]:
    """``(kind, MJCF text, absolute file path)`` of ``Scene.add_robot``'s ``path`` / ``mjcf`` (exactly one), with an
    XML source checked to parse: a malformed file reached the build as a bare ``xml`` ParseError from the importer,
    without the robot's name (AUDIT-STACK ERR-5). Example: ``source("arm", None, "<mujoco/>")[0]`` -> ``"mjcf"``."""
    from pathlib import Path  # noqa: PLC0415
    from xml.etree import ElementTree  # noqa: PLC0415

    from alloy.api._items import _robot_kind  # noqa: PLC0415

    if (path is None) == (mjcf is None):
        raise SceneError("add_robot needs exactly one of path / mjcf", "pass path='robot.xml' or mjcf='<mujoco>...'",
                         field="path")
    src = None if path is None else str(Path(path).expanduser().resolve())
    kind = "mjcf" if src is None else _robot_kind(src)
    try:
        if src is None:
            ElementTree.fromstring(mjcf)
        elif kind in ("mjcf", "urdf") and Path(src).is_file():
            ElementTree.parse(src)
    except ElementTree.ParseError as exc:
        line, column = exc.position
        raise SceneError(f"robot {name!r}: its {kind.upper()} is not well-formed XML ({exc.msg} at line {line}, "
                         f"column {column})", "close or fix the tag at that place (every <body>, <geom>, ... needs "
                         "its end tag or a closing '/>')", field="mjcf" if src is None else "path") from exc
    return kind, mjcf, src


def _sysid(scene: Any, call: str, arg: Any) -> None:
    if any(item.get("sysid") for item in scene.items.values()):     # without sysid= nothing is imported or run
        import alloy.sysid.apply as apply  # noqa: PLC0415

        getattr(apply, call)(scene, arg)


def sysid_refuse_rerun_worlds(scene: Any, rerun_worlds: Any) -> None:
    """``alloy.build(rerun_worlds=n)`` raises for a scene with ``add_robot(sysid=)``.
    Example: ``sysid_refuse_rerun_worlds(scene, None)`` does nothing."""
    _sysid(scene, "refuse_rerun_worlds", rerun_worlds)


def sysid_install(scene: Any, built: Any) -> None:
    """Write the fitted joint values into the built model (before the randomize draws).
    Example: ``sysid_install(scene, built)``."""
    _sysid(scene, "install", built)


def sysid_attach(scene: Any, engine: Any) -> None:
    """Hook the fitted action delay into the engine's control period. Example: ``sysid_attach(scene, engine)``."""
    _sysid(scene, "attach", engine)


def warn_unsaved_sysid(items: Any) -> None:
    """scene.save() writes cards, which carry no sysid values: say so once.

    Example: ``warn_unsaved_sysid(scene.items)`` logs "scene.save(): the cards do not keep sysid= for panda; ..."."""
    lost = sorted(n for n, it in items.items() if it.get("sysid"))
    if lost:
        log.warning("scene.save(): the cards do not keep sysid= for %s; a scene loaded from them has the robot "
                    "file's joint values. Keep the fit with scene.to_json() / Scene.from_dict, or pass sysid= "
                    "again.", ", ".join(lost))


def sysid_item(name: str, sysid: Any) -> dict[str, Any]:
    """The robot item's ``sysid`` entry for ``add_robot(sysid=)`` (``{}`` for None), checked.
    Example: ``sysid_item("panda", "my_panda.sysid.json")["sysid"]["joints"]``."""
    if sysid is None:
        return {}
    from alloy.sysid.apply import item_value  # noqa: PLC0415

    return {"sysid": item_value(name, sysid)}


def add_random(item: dict[str, Any], name: str, field: str, lo: float, hi: float) -> None:
    """Record a randomized start coordinate of robot ``name``'s base (absolute, m, as for objects).

    Example: ``add_random(scene.items["carton"], "carton", "x", -0.01, 0.01)``."""
    if field not in AXES:
        raise SceneError(f"robot {name!r}: cannot randomize {field!r}", "randomize a robot's base position: "
                         "'x', 'y' or 'z' (the robot's materials and masses come from its file)", field="field")
    item.setdefault("randomize", []).append({"field": f"pose.{AXES[field]}", "lo": float(lo), "hi": float(hi)})


def starts(scene: Any, seed: int) -> dict[str, np.ndarray]:
    """Per-world shifts (worlds, 3) of every robot that has per-world rows or randomized coordinates: each world's
    ``pos`` row (or the tuple with its drawn coordinates) minus the ``pos`` the robot was built at. A row moves the
    whole file as a ``pos`` tuple does, so the root body keeps where the file places it (0.1.1; it was put at the row).

    Example: ``starts(scene, seed=0)`` -> ``{"carton": array([[0, 0, 0], [0.01, 0, 0]])}``."""
    from alloy.api import _worlds  # noqa: PLC0415

    out = {}
    for name, item in scene.items.items():
        if item["kind"] != "robot" or not (item.get("per_world_pos") or item.get("randomize")):
            continue
        base = item["pose"] if item.get("pose") is not None else item["raw"]["base_pose"]
        rows = _worlds.rows(name, base, item.get("per_world_pos"), item.get("randomize", []), scene.worlds, seed)
        out[name] = rows - np.asarray(list(base)[:3], np.float64)
    return out


def place(built: Any, rows: dict[str, np.ndarray]) -> None:
    """Shift each robot's start by its per-world ``starts`` rows in the Model (root free joint, then FK of the robot),
    before the simulation snapshots its start state, so ``reset`` returns there. Example: ``place(built, rows)``."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.api.names import NameIndex  # noqa: PLC0415

    m = built.model
    idx = NameIndex.from_built(built)
    state = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, state)
    q, x_p, x_c = m.joint_q.numpy(), m.joint_X_p.numpy(), m.joint_X_c.numpy()
    body_q, q_start = state.body_q.numpy(), m.joint_q_start.numpy()
    for name, per_world in rows.items():
        root = base_joint(idx, m, name, BuildError)
        joints = idx.per_world(root, "joint")
        bodies = idx.per_world(idx.body(name), "body")
        for w, (j, b) in enumerate(zip(joints, bodies)):
            a = int(q_start[j])
            q[a:a + 7] = joint_coords(x_p[j], x_c[j], body_q[b, :3] + per_world[w], body_q[b, 3:7])
        log.info("robot %s: per-world starts for %d worlds", name, len(per_world))
    m.joint_q.assign(q)
    newton.eval_fk(m, m.joint_q, m.joint_qd, m)


def set_pose(sim: Any, name: str, pos: Any, quat_wxyz: Any, world: int | None) -> None:
    """Teleport robot ``name``'s free base now, its links with it (zero velocity, joint angles kept).

    Example: ``set_pose(sim, "carton", (0, 0, 0.2), None, world=None)``."""
    from alloy._vendor import newton  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    eng, idx = sim.engine, sim.index
    m, st = eng.model, eng.state
    worlds = list(range(sim.worlds)) if world is None else [int(world)]
    root = base_joint(idx, m, name, SimulationError)
    joints = idx.per_world(root, "joint")[worlds]
    bodies = idx.per_world(idx.body(name), "body")[worlds]
    jq, jqd, bq = st.joint_q.numpy(), st.joint_qd.numpy(), st.body_q.numpy()
    x_p, x_c, q_start = m.joint_X_p.numpy(), m.joint_X_c.numpy(), m.joint_q_start.numpy()
    p = bq[bodies, :3] if pos is None else np.broadcast_to(np.asarray(pos, np.float64), (len(worlds), 3))
    rot = bq[bodies, 3:7] if quat_wxyz is None else np.broadcast_to(
        np.asarray(quat_wxyz, np.float64)[..., [1, 2, 3, 0]], (len(worlds), 4))
    for k, j in enumerate(joints):
        a = int(q_start[j])
        jq[a:a + 7] = joint_coords(x_p[j], x_c[j], p[k], rot[k])
    mask = np.zeros(m.joint_count, bool)
    dof_start = m.joint_qd_start.numpy()
    for w in worlds:
        for j in robot_joint_ids(idx, name, w):
            mask[j] = True
            jqd[dof_start[j]:dof_start[j + 1]] = 0.0
    st.joint_q.assign(jq)
    st.joint_qd.assign(jqd)
    newton.eval_fk(m, st.joint_q, st.joint_qd, st, joint_mask=wp.array(mask, dtype=wp.bool, device=m.device))


def base_joint(idx: Any, model: Any, name: str, error: type = SimulationError) -> int:
    """The template free joint of robot ``name``'s root body; ``error`` when there is none (a welded base).

    Example: ``base_joint(sim.index, sim.engine.model, "carton")``."""
    joints = idx.robot_joints.get(name)
    if joints is None:
        raise error(f"{name!r} is not a robot", "pass a robot's name (scene.add_robot) or use set_pose on a free "
                    "body", field="name")
    parent, kind = model.joint_parent.numpy(), model.joint_type.numpy()
    roots = [j for j in joints if int(parent[j]) == -1]
    if len(roots) == 1 and int(kind[roots[0]]) == FREE_JOINT:
        return roots[0]
    raise error(f"robot {name!r} has no free base (its root is welded to the world or jointed), so it cannot be "
                "placed per world", _WELDED_FIX, field="name")


def robot_joint_ids(idx: Any, name: str, world: int) -> range:
    """Global joint ids of robot ``name`` in ``world``. Example: ``robot_joint_ids(sim.index, "carton", 1)``."""
    t = idx.robot_joints[name]
    shift = world * idx.stride["joint"]
    return range(t.start + shift, t.stop + shift)


def joint_coords(x_p: Any, x_c: Any, pos: Any, quat_xyzw: Any) -> np.ndarray:
    """Free-joint coordinates (x, y, z, qx, qy, qz, qw) that put the child body at ``pos`` / ``quat_xyzw`` in the
    world: X_p^-1 * T * X_c (Newton's FK: body = X_p * q * X_c^-1).

    Example: ``joint_coords(I, I, (0, 0, 1), (0, 0, 0, 1))`` with ``I = (0, 0, 0, 0, 0, 0, 1)``."""
    import warp as wp  # noqa: PLC0415

    def tf(row: Any) -> Any:
        r = np.asarray(row, np.float64)
        return wp.transform(wp.vec3(*r[:3]), wp.quat(*(r[3:] / np.linalg.norm(r[3:]))))

    target = wp.transform(wp.vec3(*np.asarray(pos, np.float64)), tf((0, 0, 0, *quat_xyzw)).q)
    out = wp.transform_multiply(wp.transform_multiply(wp.transform_inverse(tf(x_p)), target), tf(x_c))
    return np.asarray([*out.p, *out.q], np.float64)
