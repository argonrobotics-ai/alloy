# SPDX-License-Identifier: Apache-2.0
"""A container part that sets an MPM object's voxel: both particle counts in the route, and one build warning.

The router sizes an MPM object's voxel as a quarter of the narrowest feature the material can meet
(``alloy.route.phase.feature_size``): its own thinnest extent, the passages it meets, and the middle extent of each
container part, taken as that container's inner size. A container built from parts does not count as one shape
there. Each wall slab of ``add_open_cylinder`` counts by the larger of its thickness and its width: 4 mm for the
default 4 mm wall and 24 slabs around a 10 mm radius. So a 12 mm cube of sand in such a vial gets a 1 mm voxel
instead of 3 mm: 13,824 particles instead of 512 (the voxel ratio cubed), and the build and every step slow by about
that factor.

``note`` runs when a container part, not the material or a passage, sets the voxel. It adds both counts to the
route decision (``mpm_particles``, ``mpm_particles_free``) and a reason line, which ``sim.routes()`` and
``alloy.estimate(...)["routes"]`` show. When the part multiplies the count by ``FACTOR`` or more, it also warns
once (``alloy.api._say``: printed when no log handler shows it), naming the part and the fix.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from alloy.cards.types import ParticleSource, ResolvedObject
from alloy.route.checks import mpm_voxel

log = logging.getLogger("alloy.integration")

__all__ = ["FACTOR", "note", "particle_count"]

FACTOR = 4.0                    # warn when a container part multiplies the particle count this much or more
_WARNED: set[tuple[str, str, int, int]] = set()


def particle_count(obj: ResolvedObject, voxel: float) -> int:
    """MPM particles ``emit_mpm`` seeds for ``obj`` at ``voxel`` (2 per voxel per axis; a round primitive keeps its
    lattice points inside the shape). Past the card's particle budget, the lattice box (the build refuses it anyway).
    Example: ``particle_count(sand, 0.003)`` -> 512 for a 12 mm cube."""
    from alloy.scene.emitters.deformable import _inside, _region  # noqa: PLC0415 (the emitter's own lattice)

    lo, hi = _region(obj)
    spacing = 0.5 * voxel
    n = np.maximum(1, np.floor((hi - lo) / spacing + 1e-6).astype(int))
    total = int(np.prod(n))
    shape = obj.card.shape
    if isinstance(shape, ParticleSource) or shape.kind == "box" or total > obj.card.budget.particles:
        return total
    axes = [lo[i] + (np.arange(n[i]) + 0.5) * spacing for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return int(np.count_nonzero(_inside(shape, pts)))


def note(obj: ResolvedObject, p: Any, parts: Mapping[str, float], free: float, voxel_max: float) -> bool:
    """Record both particle counts when the container ``parts`` ({name: middle extent, m}) set ``obj``'s voxel below
    the one its own size and passages give (``free``, m); warn once when that multiplies the count by ``FACTOR`` or
    more. ``p``: the router's path (``numbers``, ``reasons``). True when it warned. Example:
    ``note(sand, p, {"vial_wall0": 0.004}, 0.012, 0.005)`` -> True."""
    voxel, free_voxel = float(p.numbers["voxel"]), mpm_voxel(free, voxel_max)
    if not parts or voxel >= free_voxel:
        return False
    name, width = min(parts.items(), key=lambda kv: kv[1])
    more = sum(1 for w in parts.values() if w <= width) - 1
    n, n_free = particle_count(obj, voxel), particle_count(obj, free_voxel)
    p.numbers.update(mpm_particles=float(n), mpm_particles_free=float(n_free))
    across = f"{width * 1e3:.3g} mm across"
    part = f"container part '{name}'" + (f" and {more} more, each {across}" if more else f", {across}")
    p.reasons.append(f"mpm voxel {voxel * 1e3:.3g} mm, set by {part}: {n:,} particles; {n_free:,} at the voxel its "
                     f"own size and passages give ({free_voxel * 1e3:.3g} mm, from {free * 1e3:.3g} mm)")
    key = (obj.card.name, name, n, n_free)
    if n < FACTOR * n_free or key in _WARNED:
        return False
    _WARNED.add(key)
    from alloy.api import _say  # noqa: PLC0415 (the user-facing warning; alloy.api imports the scene layers)

    _say.warn("'%s' (MPM material): its voxel is %.3g mm, set by %s, not by its own size and passages (%.3g mm): %s "
              "particles instead of %s (%.0fx), and the build and every step slow by about that much. Make that "
              "container's parts at least %.3g mm across (a thicker wall=, or fewer segments= on add_open_cylinder: "
              "wider slabs), or use a container of fewer, larger parts", obj.card.name, voxel * 1e3, part, free * 1e3,
              f"{n:,}", f"{n_free:,}", n / n_free, free * 1e3)
    log.info("mpm_wall: %s warned (%d vs %d particles)", obj.card.name, n, n_free)
    return True
