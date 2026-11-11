# SPDX-License-Identifier: Apache-2.0
"""Random starts at reset for a free-base robot: ``scene.randomize("go2", "start_yaw", -3.14, 3.14, when="reset")``.

The fields ``start_x``, ``start_y``, ``start_z`` (m) and ``start_yaw`` (rad, about the world's z through the base)
are offsets from the robot's start pose. ``place`` runs in the engine's reset, after the worlds are back at their
start and the rows are drawn: it moves the base of every reset world by its draws, the links with it (joint angles
and the zero start velocity kept), and takes the health check's energy reference from the new start. Host-side,
outside any captured graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.api import _robot_random as R
from alloy.api.errors import BuildError

COLUMN = {"start_x": 0, "start_y": 1, "start_z": 2, "start_yaw": 3}


def robots(rows: list[dict[str, Any]]) -> list[str]:
    """The robots with a start row, in row order. Example: ``robots(st.rows) == ["go2"]``."""
    return list(dict.fromkeys(r["target"] for r in rows if r["field"] in COLUMN))


def offsets(st: Any, robot: str, worlds: np.ndarray) -> np.ndarray:
    """(len(worlds), 4): the summed x, y, z and yaw draws of ``robot`` in ``worlds``."""
    out = np.zeros((len(worlds), 4))
    for r in st.rows:
        if r["target"] == robot and r["field"] in COLUMN:
            out[:, COLUMN[r["field"]]] += st.values[R.key(r)][worlds]
    return out


def place(engine: Any, worlds: np.ndarray | None = None) -> None:
    """Move each robot with start rows by its current draws, in ``worlds`` (all when None), from the pose the state
    holds now (a world just reset). Nothing without such rows. Example: ``place(sim.engine)`` after a reset."""
    st = getattr(engine.built, "robot_random", None)
    names = robots(st.rows) if st is not None else []
    if not names:
        return
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.api import _robots  # noqa: PLC0415

    model, state, idx = engine.model, engine.state, st.index
    worlds = np.arange(len(st.counts)) if worlds is None else np.asarray(worlds, np.int64)
    if not len(worlds):
        return
    jq, bq = state.joint_q.numpy().copy(), state.body_q.numpy()
    x_p, x_c, q_start = model.joint_X_p.numpy(), model.joint_X_c.numpy(), model.joint_q_start.numpy()
    moved = np.zeros(model.joint_count, bool)
    for robot in names:
        root = free_root(idx, model, robot)
        body, off = int(model.joint_child.numpy()[root]), offsets(st, robot, worlds)
        for k, w in enumerate(worlds):
            j, b = root + int(w) * idx.stride["joint"], body + int(w) * idx.stride["body"]
            a = int(q_start[j])
            jq[a:a + 7] = _robots.joint_coords(x_p[j], x_c[j], bq[b, :3] + off[k, :3], _yawed(bq[b, 3:7], off[k, 3]))
            ids = _robots.robot_joint_ids(idx, robot, int(w))
            moved[ids.start:ids.stop] = True
    state.joint_q.assign(jq)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state,
                   joint_mask=wp.array(moved, dtype=wp.bool, device=model.device))
    health = engine.health
    if health is not None and hasattr(health, "reset_reference"):
        mask = np.zeros(len(st.counts), np.int32)
        mask[worlds] = 1
        health.reset_reference(state, wp.array(mask, dtype=wp.int32, device=model.device))


def free_root(idx: Any, model: Any, robot: str) -> int:
    """The template joint of ``robot``'s free base; ``BuildError`` when its file has none, or more than one free
    body at its root. Example: ``free_root(idx, model, "go2")``."""
    parent, kind = model.joint_parent.numpy(), model.joint_type.numpy()
    roots = [j for j in idx.robot_joints[robot] if int(parent[j]) == -1 and int(kind[j]) == 4]   # JointType.FREE
    if len(roots) == 1:
        return roots[0]
    raise BuildError(f"robot {robot!r} has {'no' if not roots else 'more than one'} free base, so its start cannot "
                     "be randomized", "use a robot whose root body alone has a free joint (MJCF <freejoint/>), such "
                     "as the Go2", field="name")


def _yawed(quat_xyzw: np.ndarray, yaw: float) -> np.ndarray:
    """``quat_xyzw`` turned by ``yaw`` rad about the world's z. Example: ``_yawed((0, 0, 0, 1), 0.0)`` is the same."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    s, c = np.sin(0.5 * yaw), np.cos(0.5 * yaw)
    return np.asarray([c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z])


__all__ = ["COLUMN", "free_root", "offsets", "place", "robots"]
