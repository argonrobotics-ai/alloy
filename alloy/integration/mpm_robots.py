# SPDX-License-Identifier: Apache-2.0
"""Which bodies may touch an MPM material, and whether they are servo-driven robot links (speed-now-1109 S5).

``mpm_clock``'s "auto" pair clock is fast only for a robot (or tool) working in sand or paste: every coupled pair ends
in MPM, the scene has a servo-driven articulation, and no other moving body starts near the MPM material. Decided once,
at build, from the build state:
- near: a body with a colliding shape whose bounding sphere (``shape_collision_radius`` about the shape's origin) comes
  within ``MARGIN`` of the MPM particles' bounding box;
- a near body is fine when it is welded to the world (every joint up to the world fixed) or a link of an articulation
  that has an actuator (a MuJoCo actuator, or a joint target stiffness); it is not when it is kinematic, has a free
  (or ball, distance, D6) joint on its way to the world, or belongs to an articulation without an actuator.
A free body that starts away from the material and reaches it later (a dropped cup) is not seen: it then runs on the
fast clock (docs/known_gaps/mpm-solve-speed.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy._vendor import newton

__all__ = ["MARGIN", "robot_only"]

MARGIN = 0.02                    # m around the MPM material's build-time bounding box
_COLLIDES = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
_KINEMATIC = int(newton.BodyFlags.KINEMATIC)
_FIXED = int(newton.JointType.FIXED)
_SERVO = (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE))


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate rows of ``v`` by the xyzw quaternions in rows of ``q``."""
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _mpm_box(built: Any) -> tuple[np.ndarray, np.ndarray] | None:
    ids = [i for e in built.entries if e.capability == "mpm" for i in e.particles]
    if not ids:
        return None
    x = built.model.particle_q.numpy()[np.asarray(ids, dtype=np.int64)]
    return x.min(0) - MARGIN, x.max(0) + MARGIN


def near_bodies(built: Any) -> np.ndarray:
    """Body ids with a colliding shape that starts within ``MARGIN`` of the MPM material (module doc)."""
    m, box = built.model, _mpm_box(built)
    if box is None or not int(m.shape_count or 0) or not int(m.body_count or 0):
        return np.zeros(0, dtype=np.int64)
    body, flags = m.shape_body.numpy(), m.shape_flags.numpy().astype(np.int64)
    keep = (body >= 0) & ((flags & _COLLIDES) != 0)
    s, b = np.flatnonzero(keep), body[keep]
    bq = m.body_q.numpy().astype(np.float64)[b]
    local = m.shape_transform.numpy().astype(np.float64)[s, :3]
    centre = bq[:, :3] + _rotate(bq[:, 3:7], local)
    gap = np.linalg.norm(np.maximum(box[0] - centre, 0.0) + np.maximum(centre - box[1], 0.0), axis=1)
    return np.unique(b[gap <= m.shape_collision_radius.numpy()[s]])


def _actuated(m: Any) -> set[int]:
    """Articulations with an actuator: a MuJoCo actuator, or a joint DOF with a target stiffness."""
    out: set[int] = set()
    owners = (getattr(m, "custom_frequency_articulation", None) or {}).get("mujoco:actuator")
    if owners is not None and owners.shape[0]:
        out |= {int(a) for a in owners.numpy() if a >= 0}
    ke = getattr(m, "joint_target_ke", None)
    if ke is not None and int(m.joint_count or 0):
        start, art = m.joint_qd_start.numpy(), m.joint_articulation.numpy()
        stiff = ke.numpy() > 0.0
        out |= {int(art[j]) for j in range(int(m.joint_count)) if art[j] >= 0 and stiff[start[j]:start[j + 1]].any()}
    return out


def _kind(m: Any, b: int, joints: dict[int, int], actuated: set[int]) -> str:
    """"welded", "robot", "kinematic", "free" or "passive" for body ``b`` (module doc)."""
    if int(m.body_flags.numpy()[b]) & _KINEMATIC:
        return "kinematic"
    jt, parent, art = m.joint_type.numpy(), m.joint_parent.numpy(), m.joint_articulation.numpy()
    if b not in joints:
        return "free"                              # a body without a joint floats free
    kinds, j, seen = set(), joints.get(b), 0
    while j is not None and seen < len(jt):
        kinds.add(int(jt[j]))
        p = int(parent[j])
        j, seen = (joints.get(p) if p >= 0 else None), seen + 1
    if kinds <= {_FIXED}:
        return "welded"
    if not kinds <= {_FIXED, *_SERVO}:
        return "free"
    return "robot" if int(art[joints[b]]) in actuated else "passive"


def robot_only(built: Any) -> tuple[bool, str]:
    """(True, why) when the scene has a servo-driven articulation and every body near its MPM material is welded or
    such a robot's link; else (False, why naming the first body that is not)."""
    m = built.model
    actuated = _actuated(m)
    if not actuated:
        return False, "no servo-driven robot in the scene"
    joints = {int(c): j for j, c in enumerate(m.joint_child.numpy()) if c >= 0}
    labels = list(getattr(m, "body_label", None) or [])
    for b in near_bodies(built):
        kind = _kind(m, int(b), joints, actuated)
        if kind not in ("welded", "robot"):
            name = labels[b].rsplit("/", 1)[-1] if b < len(labels) else f"body {b}"
            return False, f"{name} ({kind}) starts within {MARGIN * 100:g} cm of the MPM material"
    return True, "every body that starts near the MPM material is welded or a servo-driven robot link"
