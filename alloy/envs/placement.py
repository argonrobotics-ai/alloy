# SPDX-License-Identifier: Apache-2.0
"""Put library objects at rest on the floor, whatever their origin convention (a YCB scan's origin sits under the
object, a primitive card's at its centre): measure the lowest point of the object's collision shapes in the built
model, lift or lower its free joint so that point touches the floor, and make that the reset start.

Example (a ``Task.on_build`` hook)::

    Task(..., on_build=lambda sim: rest_on_floor(sim, ["obj"]))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError

CYLINDER_POINTS = 16


def _rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(c) for c in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _shape_points(kind: int, scale: np.ndarray, src: Any) -> tuple[np.ndarray, float]:
    """Points of one shape in its own frame and a radius to subtract (spheres, capsules)."""
    from alloy._vendor import newton  # noqa: PLC0415

    g = newton.GeoType
    if kind in (int(g.MESH), int(g.CONVEX_MESH)) and getattr(src, "vertices", None) is not None:
        return np.asarray(src.vertices, np.float64) * scale, 0.0
    if kind == int(g.BOX):
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
        return corners * scale, 0.0
    if kind == int(g.SPHERE):
        return np.zeros((1, 3)), float(scale[0])
    if kind == int(g.CAPSULE):
        return np.array([[0.0, 0.0, -scale[1]], [0.0, 0.0, scale[1]]]), float(scale[0])
    if kind in (int(g.CYLINDER), int(g.CONE)):
        a = np.linspace(0.0, 2 * np.pi, CYLINDER_POINTS, endpoint=False)
        ring = np.stack([scale[0] * np.cos(a), scale[0] * np.sin(a), np.zeros_like(a)], 1)
        return np.concatenate([ring + [0, 0, -scale[1]], ring + [0, 0, scale[1]]]), 0.0
    raise EnvError(f"cannot measure a shape of type {kind}", "use primitive or mesh collision shapes", field="shape")


def lowest_z(sim: Any, name: str, world: int = 0) -> float:
    """Lowest point (m) of a body's collision shapes in one world. Example: ``lowest_z(sim, "obj")``."""
    from alloy._vendor import newton  # noqa: PLC0415

    m, idx = sim.engine.model, sim.index
    body = int(idx.per_world(idx.body(name), "body")[world])
    sb, flags = m.shape_body.numpy(), m.shape_flags.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    tf, scale, kind = m.shape_transform.numpy(), m.shape_scale.numpy(), m.shape_type.numpy()
    bq = sim.engine.state.body_q.numpy()[body]
    r_wb, low = _rot(bq[3:7]), np.inf
    for s in np.flatnonzero((sb == body) & ((flags & collide) != 0)):
        pts, radius = _shape_points(int(kind[s]), scale[s].astype(np.float64), m.shape_source[s])
        world_pts = (pts @ _rot(tf[s, 3:7]).T + tf[s, :3]) @ r_wb.T + bq[:3]
        low = min(low, float(world_pts[:, 2].min()) - radius)
    if not np.isfinite(low):
        raise EnvError(f"{name!r} has no collision shapes", "pass a body with collision geometry", field=name)
    return low


def rest_on_floor(sim: Any, names: Sequence[str], floor: float = 0.0, gap: float = 0.0005) -> dict[str, float]:
    """Move each named free object up or down so its lowest point is ``gap`` above ``floor`` in every world, and
    make that the state ``reset`` returns to. Returns the shift per object (m).

    Example: ``rest_on_floor(sim, ["obj"])``."""
    from alloy._vendor import newton  # noqa: PLC0415

    eng, idx = sim.engine, sim.index
    joint_q, q_start = eng.state.joint_q.numpy(), eng.model.joint_q_start.numpy()
    x_p, shifts = eng.model.joint_X_p.numpy(), {}
    for name in names:
        if name not in idx.free_joint:
            raise EnvError(f"{name!r} is not a free object", "rest_on_floor moves objects added with add_object / "
                           "add_box, not robots or static parts", field=name)
        dz = floor + gap - lowest_z(sim, name)
        for j in idx.per_world(idx.free_joint[name], "joint"):
            q0 = int(q_start[j])
            joint_q[q0:q0 + 3] += _rot(x_p[j, 3:7]).T @ np.array([0.0, 0.0, dz])
        shifts[name] = dz
    eng.state.joint_q.assign(joint_q)
    newton.eval_fk(eng.model, eng.state.joint_q, eng.state.joint_qd, eng.state)
    _make_start(eng)
    return shifts


def _make_start(eng: Any) -> None:
    """The current state becomes what ``reset`` restores (as ``alloy.Sim.settle`` does)."""
    eng._initial.save(eng.state)
    latch = getattr(eng, "_latch_snap", None)
    if latch is not None:
        latch.save(eng.state)
    if eng.health is not None and hasattr(eng.health, "reset_reference"):
        eng.health.reset_reference(eng.state, None)


__all__ = ["lowest_z", "rest_on_floor"]
