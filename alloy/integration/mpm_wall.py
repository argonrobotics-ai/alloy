# SPDX-License-Identifier: Apache-2.0
"""A container part that sets an MPM object's voxel: both particle counts in the route, and one build warning.

The router sizes an MPM object's voxel as a quarter of the narrowest feature the material can meet
(``alloy.route.phase.feature_size``): its own thinnest extent, the passages it meets, and each container part's
inner size. A fixture part (``add_open_box``, ``add_open_cylinder``, ``add_block_with_hole``) gives its fixture's
inner width up to what its wall holds (``alloy.integration.mpm_cavity``); any other container part gives its own
middle extent, as every part did through 0.1.7. A container built from parts is then not one shape: through 0.1.7
each 4 mm wall slab of a default ``add_open_cylinder`` (24 slabs around a 10 mm radius) counted by the larger of its
thickness and width, 4 mm, so a 12 mm cube of sand inside got a 1 mm voxel instead of 3 mm: 13,824 particles instead
of 512 (the voxel ratio cubed), and the build and every step slowed by about that factor. Its 4 mm walls now hold a
1.33 mm voxel (5,832 particles), and a 9 mm wall gives 512. Containers built from ``add_box`` parts by hand, and
cards saved without their fixture, still route by each part's own size.

``note`` runs when a container part, not the material or a passage, sets the voxel. It adds both counts to the
route decision (``mpm_particles``, ``mpm_particles_free``) and a reason line, which ``sim.routes()`` and
``alloy.estimate(...)["routes"]`` show. When the part multiplies the count by ``FACTOR`` or more, it also warns
once (``alloy.api._say``: printed when no log handler shows it), naming the part and the fix. ``wall_capped`` adds a
reason line when fixture walls hold the voxel to a third of their thickness, below what their fixture's width gives.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from alloy.cards.types import ParticleSource, ResolvedObject
from alloy.route.checks import mpm_voxel

log = logging.getLogger("alloy.integration")

__all__ = ["FACTOR", "caveats", "note", "particle_count", "wall_capped"]

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


def note(obj: ResolvedObject, p: Any, parts: Mapping[str, float], free: float, voxel_max: float,
         fixtures: Collection[str] = ()) -> bool:
    """Record both particle counts when the container ``parts`` ({name: inner size, m}) set ``obj``'s voxel below
    the one its own size and passages give (``free``, m); warn once when that multiplies the count by ``FACTOR`` or
    more. ``fixtures``: the parts that are a fixture's (their fix is a thicker wall). ``p``: the router's path
    (``numbers``, ``reasons``). True when it warned. Example: ``note(sand, p, {"vial_wall0": 0.004}, 0.012, 0.005)``
    -> True."""
    voxel, free_voxel = float(p.numbers["voxel"]), mpm_voxel(free, voxel_max)
    if not parts or voxel >= free_voxel:
        return False
    name, width = min(parts.items(), key=lambda kv: kv[1])
    more = sum(1 for w in parts.values() if w <= width) - 1
    n, n_free = particle_count(obj, voxel), particle_count(obj, free_voxel)
    p.numbers.update(mpm_particles=float(n), mpm_particles_free=float(n_free))
    across = f"{width * 1e3:.3g} mm across"
    part = f"container part '{name}'" + (f" and {more} more, each {across}" if more else f", {across}")
    fix = _fix(free, name in fixtures)
    p.reasons.append(f"mpm voxel {voxel * 1e3:.3g} mm, set by {part}: {n:,} particles; {n_free:,} at the voxel its "
                     f"own size and passages give ({free_voxel * 1e3:.3g} mm, from {free * 1e3:.3g} mm), which {fix} "
                     "would give")
    key = (obj.card.name, name, n, n_free)
    if n < FACTOR * n_free or key in _WARNED:
        return False
    _WARNED.add(key)
    from alloy.api import _say  # noqa: PLC0415 (the user-facing warning; alloy.api imports the scene layers)

    _say.warn("'%s' (MPM material): its voxel is %.3g mm, set by %s, not by its own size and passages (%.3g mm): %s "
              "particles instead of %s (%.0fx), and the build and every step slow by about that much. Give that "
              "container %s", obj.card.name, voxel * 1e3, part, free * 1e3, f"{n:,}", f"{n_free:,}", n / n_free, fix)
    log.info("mpm_wall: %s warned (%d vs %d particles)", obj.card.name, n, n_free)
    return True


def _fix(free: float, fixture: bool) -> str:
    """The change that gives a material of ``free`` m its own size's voxel: for a fixture, walls of ``WALL_VOXELS``
    voxels (``alloy.integration.mpm_cavity``) or wider parts; for other containers, wider parts."""
    if not fixture:
        return f"parts at least {free * 1e3:.3g} mm across (a container of fewer, larger parts)"
    from alloy.integration.mpm_cavity import WALL_VOXELS  # noqa: PLC0415

    return (f"walls at least {WALL_VOXELS * free / 4.0 * 1e3:.3g} mm thick (wall={WALL_VOXELS * free / 4.0:.3g}) or "
            f"parts at least {free * 1e3:.3g} mm across (fewer segments= on add_open_cylinder: wider slabs)")


def wall_capped(p: Any, parts: Mapping[str, tuple[float, float, float]], free: float) -> bool:
    """A reason line when fixture ``parts`` ({name: (fixture inner width, wall thickness, own middle extent), m}) hold
    the voxel to a fraction of their wall (``mpm_cavity.part_size``: the wall term, not the part's own middle extent,
    set it), below what their fixture's width and its own size and passages (``free``, m) give; True when one does.
    Example:
    ``wall_capped(p, {"vial_wall0": (0.02, 0.004, 0.004)}, 0.012)`` -> True."""
    from alloy.integration.mpm_cavity import part_size  # noqa: PLC0415

    voxel = float(p.numbers["voxel"])
    sizes = {k: (part_size(w, t, own), w, t) for k, (w, t, own) in parts.items()}
    capped = {k: t for k, (size, w, t) in sizes.items() if size < min(w, free) and 0.25 * size <= voxel * 1.0001
              and size > parts[k][2]}            # the wall term set it, not the part's own size
    if not capped:
        return False
    name, t = min(capped.items(), key=lambda kv: kv[1])
    more = f" and {len(capped) - 1} more" if len(capped) > 1 else ""
    p.numbers["wall_capped"] = t
    from alloy.integration.mpm_cavity import WALL_VOXELS  # noqa: PLC0415

    p.reasons.append(f"mpm voxel {voxel * 1e3:.3g} mm: held to 1/{WALL_VOXELS:g} of the wall of fixture part '{name}'"
                     f"{more}, {t * 1e3:.3g} mm thick (a thinner wall lets material through), below what the fixture's "
                     f"{parts[name][0] * 1e3:.3g} mm inner width gives")
    return True


def caveats(report: Any) -> list[str]:
    """One sentence per MPM object whose voxel a container part sets with ``FACTOR`` or more particles (the objects
    ``note`` warns about), for ``alloy.estimate(...)["caveats"]``. Example: ``caveats(built.report)`` ->
    ``["'sand': mpm voxel 1 mm, set by container part 'vial_wall0' ..."]``."""
    out = []
    for name, d in report.decisions.items():
        n, free = d.numbers.get("mpm_particles"), d.numbers.get("mpm_particles_free")
        if n is None or free is None or n < FACTOR * free:
            continue
        why = next((r for r in d.reasons if "set by container part" in r), "")
        out.append(f"'{name}': {why} ({n / free:.0f}x the particles)")
    return out
