# SPDX-License-Identifier: Apache-2.0
"""Per-body soft-contact list size for the `vbd` entry (SCR-WP10-6), Newton's
``rigid_body_particle_contact_buffer_size``.

SolverVBD keeps, per rigid body, a list of the soft contacts touching it (Newton's default: 256 entries). A body
whose contacts overflow the list makes each of its lanes scan every soft contact of the model, across all worlds, and
Newton warns every step. A towel draped on a Panda link holds up to 3,146 records; WP13's 20-object pile 548 over a
256-vertex cloth ([internal note] notes/mixed-speed-1107.md, lever 4b).

The list is sized from the model: the most records one body can hold in one world is its shapes times its world's
particles, edges and triangles (Newton emits one record per particle-shape pair, and with full-surface contact one
per edge-shape and triangle-shape pair). As a power of two, from Newton's 256 up to ``LIST_MAX`` and to what
``LIST_BYTES`` holds for all bodies' lists together. Larger lists cost memory only: the kernels walk each body's
stored count.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LIST_MIN = 256                # Newton's per-body list
LIST_BYTES = 256 << 20        # memory all bodies' lists may take together (the film entry's budget)
LIST_MAX = 8192               # per body: 2.6x the most seen (3,146 on a Panda link under a towel)


def _most_per_world(world: np.ndarray, world_count: int) -> int:
    """Largest count of entries in one world; global entries (world -1) are in every world."""
    if world.size == 0:
        return 0
    world = np.asarray(world, dtype=np.int64)
    local = np.bincount(world[world >= 0], minlength=max(world_count, 1))
    return int(local.max(initial=0)) + int((world < 0).sum())


def records_per_body(model: Any) -> int:
    """Upper bound on one body's soft-contact records in one world (0 without bodies, shapes or particles)."""
    if not (model.body_count and model.shape_count and model.particle_count):
        return 0
    shape_body = model.shape_body.numpy()
    shapes = int(np.bincount(shape_body[shape_body >= 0], minlength=model.body_count).max(initial=0))
    worlds = int(getattr(model, "world_count", 1) or 1)
    pworld = model.particle_world.numpy()
    soft = _most_per_world(pworld, worlds)
    if getattr(model, "edge_count", 0):
        soft += _most_per_world(pworld[model.edge_indices.numpy()[:, 2]], worlds)
    if getattr(model, "tri_count", 0):
        soft += _most_per_world(pworld[model.tri_indices.numpy()[:, 0]], worlds)
    return shapes * soft


def list_size(model: Any, budget: int = LIST_BYTES) -> int:
    """Per-body list entries for ``model``: ``records_per_body`` as a power of two, within ``budget`` bytes."""
    want = max(int(records_per_body(model)), 1)
    room = budget // (4 * max(int(model.body_count), 1))
    room_pow2 = 1 << max(int(room).bit_length() - 1, 0)
    return max(LIST_MIN, min(1 << (want - 1).bit_length(), room_pow2, LIST_MAX))
