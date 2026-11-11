# SPDX-License-Identifier: Apache-2.0
"""No contacts between shapes that can never push each other: kinematic vs static, kinematic vs kinematic.

A kinematic body (``kinematic=True``) moves only as it is driven (``sim.drive``, ``sim.set_pose``), never by a force,
and a static shape never moves. A contact between the two, or between two kinematic bodies, changes nothing, but it
still costs narrowphase work and fills the contact buffer. A kinematic ``add_open_cylinder`` is 25 kinematic boxes.
Resting on the ground they made 296 MuJoCo contacts, and their corners overlapped one another (START_PEN). Beside a
robot they overflowed MuJoCo Warp's contact buffer: CONTACT_OVERFLOW, every world INVALID, and Warp's overflow
warning about 120 times per step (an outside tester, 2026-11-09).

``filter_fixed_pairs`` adds a shape collision filter pair for every such pair of colliding shapes on the builder.
The assembler calls it before the explicit MuJoCo pair rows are written (``alloy.scene.mujoco_pairs`` skips filtered
pairs). Newton turns the filters into MuJoCo excludes and contact masks, and Newton's own pipeline honours them.
Particles are not shapes: liquid, cloth and rods still meet kinematic and static walls.
"""

from __future__ import annotations

import logging

from alloy._vendor import newton

log = logging.getLogger("alloy.scene")


def _colliding(b: newton.ModelBuilder, s: int) -> bool:
    return bool(int(b.shape_flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES))


def fixed_shapes(b: newton.ModelBuilder) -> tuple[list[int], dict[int, list[int]]]:
    """(static colliding shapes, {kinematic body: its colliding shapes}) on the builder."""
    kinematic = {i for i, f in enumerate(b.body_flags) if int(f) & int(newton.BodyFlags.KINEMATIC)}
    static: list[int] = []
    moved: dict[int, list[int]] = {}
    for s, body in enumerate(b.shape_body):
        if not _colliding(b, s):
            continue
        if int(body) < 0:
            static.append(s)
        elif int(body) in kinematic:
            moved.setdefault(int(body), []).append(s)
    return static, moved


def filter_fixed_pairs(b: newton.ModelBuilder) -> int:
    """Filter every kinematic-vs-static and kinematic-vs-kinematic (other body) colliding shape pair. Returns the
    pairs added."""
    static, moved = fixed_shapes(b)
    if not moved:
        return 0
    bodies = sorted(moved)
    pairs = [(k, s) for body in bodies for k in moved[body] for s in static]
    pairs += [(k, o) for i, body in enumerate(bodies) for other in bodies[i + 1:]
              for k in moved[body] for o in moved[other]]
    for a, c in pairs:
        b.add_shape_collision_filter_pair(a, c)
    log.info("fixed pairs: %d kinematic-vs-static / kinematic-vs-kinematic shape pairs filtered (%d kinematic bodies)",
             len(pairs), len(bodies))
    return len(pairs)


__all__ = ["filter_fixed_pairs", "fixed_shapes"]
